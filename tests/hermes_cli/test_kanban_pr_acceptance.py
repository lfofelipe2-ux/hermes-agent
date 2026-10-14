"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance_store as store
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,json,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "value=json.loads(urllib.request.urlopen(u).read().decode())\n"
                  "if '--paginate' in sys.argv and '--slurp' not in sys.argv:\n"
                  " for page in value: print(json.dumps(page))\n"
                  "else: print(json.dumps(value))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "isolated-native-kanban.db"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platforms("linux")
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.platforms("linux")
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, result="done", expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


# --- #122689: acceptance must read the repo as the ASSIGNEE profile's gh login ---

@pytest.mark.platforms("posix")
def test_acceptance_runs_gh_as_the_assignee_profile(tmp_path, monkeypatch):
    """The gh child env carries the assignee's own GH credentials (its .env),
    never the ambient/launch residue, and an invisible repo is classified
    `auth` naming the repository — not a retryable `infra` failure."""
    from pathlib import Path

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=b-token\n", encoding="utf-8")
    # Ambient residue that must NOT decide the login.
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os\n"
                  f"json.dump(dict(os.environ), open({str(env_dump)!r}, 'w'))\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["classification"] == "auth"
        assert "acme/repo" in receipts[-1]["detail"]
    captured = json.loads(env_dump.read_text())
    assert captured["GH_TOKEN"] == "b-token"
    assert captured.get("GH_CONFIG_DIR") != "/nonexistent/launch/gh"
    assert "credentials" in (kb.get_task(conn, tid).last_failure_error or "")


@pytest.mark.platforms("posix")
def test_assignee_without_own_gh_login_never_falls_through_to_ambient_login(tmp_path, monkeypatch):
    """An assignee profile with no GH_TOKEN/GH_CONFIG_DIR of its own must not inherit the
    launch user's ~/.config/gh (HOME/XDG_CONFIG_HOME stay the launch process's): gh is pinned
    to a profile-owned config dir, its 'not logged in' exit is classified `auth` naming the profile."""
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "launch-xdg"))

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh: GH_CONFIG_DIR wins; a config dir without hosts.yml means "not logged in" (exit 4).
    gh.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                  f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n"
                  "if 'GH_CONFIG_DIR' in os.environ and not os.path.exists(os.environ['GH_CONFIG_DIR']):\n"
                  "    sys.exit(4)\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'b'" in receipt["detail"] and "no login" in receipt["detail"]
    captured = json.loads(env_dump.read_text(encoding="utf-8-sig"))
    assert captured["GH_CONFIG_DIR"] == str(assignee_home / "gh")
    assert "GH_TOKEN" not in captured and "GITHUB_TOKEN" not in captured


def test_assigned_card_with_unresolvable_profile_is_auth_not_ambient(tmp_path, monkeypatch):
    """A card assigned to a profile that no longer exists must not run gh as the completing
    process's ambient login: classification `auth` naming the profile, gh never invoked."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "isolated-native-kanban.db"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # any gh spawn would fail as infra
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-ghost", completion_contract="acme/repo", assignee="ghost")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'ghost'" in receipt["detail"] and "cannot be resolved" in receipt["detail"]


@pytest.fixture
def live_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "isolated-kanban.db"))
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: tmp_path / "isolated-kanban.db")
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="bound", assignee="external-root", completion_contract="acme/repo")
        kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
        if task.status != "running":
            raise AssertionError("fixture requires running task")
        run_id = task.current_run_id
        yield conn, tid, run_id


def bind(conn, tid, run_id, *, owner=None, repo="acme/repo", assignee="external-root"):
    kb._append_event(conn, tid, "external_github_identity_bound", {
        "run_id": run_id, "owner": owner or kb.get_task(conn, tid).claim_lock,
        "profile": "existing", "expected_login": "authorized-login", "repo": repo,
        "assignee": assignee,
    }, run_id=run_id)


def gh_ok(monkeypatch, login="authorized-login"):
    monkeypatch.setattr(store, "_gh_env", lambda home: {"TEST_BOUND_IDENTITY": "fixed"})
    monkeypatch.setattr(store, "_api", lambda endpoint, **kw: {"login": login} if endpoint == "user" else {"full_name": "acme/repo"})
    monkeypatch.setattr(store, "collect_acceptance", lambda *a, **kw: {"ok": True, "classification": "success"})
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "normalize_profile_name", lambda p: p)
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda p: "/mock/profile")


def test_bound_identity_accepts_without_reassigning_external_assignee(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    assert prepared and prepared[1]["ok"]
    assert kb.get_task(conn, tid).assignee == "external-root"
    assert store.record_acceptance(conn, tid, prepared)
    assert kb.get_task(conn, tid).assignee == "external-root"


@pytest.mark.parametrize("mutation", ["owner", "run", "assignee", "repo"])
def test_bad_binding_rejected_and_assignee_unchanged(live_task, monkeypatch, mutation):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id, **({"owner": "other"} if mutation == "owner" else {"repo": "other/repo"} if mutation == "repo" else {}))
    gh_ok(monkeypatch)
    if mutation == "run":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET current_run_id=current_run_id+1 WHERE id=?", (tid,))
    if mutation == "assignee":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET assignee='changed' WHERE id=?", (tid,))
    assert store.prepare_acceptance(conn, tid, run_id, {}) is False
    assert kb.get_task(conn, tid).assignee == ("changed" if mutation == "assignee" else "external-root")


def test_missing_profile_and_mismatched_gh_identity_fail_closed(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch, login="somebody-else")
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    assert prepared[1]["classification"] == "auth" and not prepared[1]["ok"]
    assert kb.get_task(conn, tid).assignee == "external-root"
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda p: (_ for _ in ()).throw(FileNotFoundError()))
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    assert prepared[1]["classification"] == "auth"


def test_changed_identity_guard_rejects_terminal_receipt(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET assignee='changed' WHERE id=?", (tid,))
    assert not store.record_acceptance(conn, tid, prepared)
    assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
    assert kb.get_task(conn, tid).assignee == "changed"


def test_changed_claim_after_prepare_rejects_receipt(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock='different-owner' WHERE id=?", (tid,))
    assert not store.record_acceptance(conn, tid, prepared)
    assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
    assert kb.get_task(conn, tid).assignee == "external-root"




def test_old_gh_pagination_keeps_all_pages_without_slurp(monkeypatch):
    import types
    import hermes_cli.kanban_pr_acceptance as pr
    commands=[]
    monkeypatch.setattr(pr,"_gh_env",lambda p: {})
    def run(command,**kwargs):
        commands.append(command);return types.SimpleNamespace(stdout='[{"id":1}]\n[{"id":2}]\n')
    monkeypatch.setattr(pr.subprocess,"run",run)
    assert pr._api("rules",paginate=True,profile_home="existing") == [[{"id":1}],[{"id":2}]]
    assert "--paginate" in commands[0] and "--slurp" not in commands[0]

def test_old_gh_truncated_page_rejected(monkeypatch):
    import types
    import hermes_cli.kanban_pr_acceptance as pr
    monkeypatch.setattr(pr,"_gh_env",lambda p: {})
    monkeypatch.setattr(pr.subprocess,"run",lambda *a,**kw:types.SimpleNamespace(stdout='[]\n{'))
    with pytest.raises(ValueError):pr._api("rules",paginate=True,profile_home="existing")

def test_nonpaginated_graphql_shape_preserved(monkeypatch):
    import types
    import hermes_cli.kanban_pr_acceptance as pr
    monkeypatch.setattr(pr,"_gh_env",lambda p: {})
    monkeypatch.setattr(pr.subprocess,"run",lambda *a,**kw:types.SimpleNamespace(stdout='{"data":{"repository":{}}}'))
    assert pr._api("graphql",query="query",profile_home="existing") == {"data":{"repository":{}}}


def test_native_review_handoff_retains_bound_identity(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    assert kb.request_review(conn, tid, expected_run_id=run_id)
    prepared = store.prepare_acceptance(conn, tid, None, {})
    assert prepared and prepared[1]["ok"]
    assert store.record_acceptance(conn, tid, prepared)
    assert kb.get_task(conn, tid).assignee == "external-root"


def test_review_without_native_handoff_does_not_reuse_binding(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    conn.execute("UPDATE tasks SET status='review',claim_lock=NULL WHERE id=?", (tid,))
    assert store.prepare_acceptance(conn, tid, run_id, {}) is False


@pytest.mark.parametrize("field", ["profile", "expected_login"])
def test_malformed_binding_produces_auth_receipt(live_task, monkeypatch, field):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    row = conn.execute("SELECT id,payload FROM task_events WHERE kind='external_github_identity_bound'").fetchone()
    payload = json.loads(row["payload"])
    del payload[field]
    conn.execute("UPDATE task_events SET payload=? WHERE id=?", (json.dumps(payload), row["id"]))
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    assert prepared[1]["classification"] == "auth"
    assert store.record_acceptance(conn, tid, prepared) is False
    assert kb.get_task(conn, tid).status == "running"


@pytest.mark.parametrize("fault", [OSError("transport unavailable"), __import__("subprocess").TimeoutExpired("gh", 30)])
def test_bound_preflight_infrastructure_failure_has_durable_receipt(live_task, monkeypatch, fault):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    def fail(*args, **kwargs):
        raise fault
    monkeypatch.setattr(store, "_api", fail)
    prepared = store.prepare_acceptance(conn, tid, run_id, {})
    assert prepared[1]["classification"] == "infra"
    assert store.record_acceptance(conn, tid, prepared) is False
    assert conn.execute("SELECT count(*) FROM task_events WHERE kind='pr_acceptance'").fetchone()[0] == 1
    assert kb.get_task(conn, tid).status == "running"


def test_identity_environment_is_captured_once_for_entire_acceptance(live_task, monkeypatch):
    conn, tid, run_id = live_task
    bind(conn, tid, run_id)
    gh_ok(monkeypatch)
    captured = []
    frozen = {"TEST_BOUND_IDENTITY": "one"}
    def environment(home):
        captured.append(home)
        return frozen
    monkeypatch.setattr(store, "_gh_env", environment)
    observed = []
    def api(endpoint, **kwargs):
        observed.append(kwargs["child_env"])
        return {"login": "authorized-login"} if endpoint == "user" else {"full_name": "acme/repo"}
    def collect(*args, **kwargs):
        observed.append(kwargs["child_env"])
        return {"ok": True, "classification": "success"}
    monkeypatch.setattr(store, "_api", api)
    monkeypatch.setattr(store, "collect_acceptance", collect)
    assert store.prepare_acceptance(conn, tid, run_id, {})[1]["ok"]
    assert captured == ["/mock/profile"]
    assert len(observed) == 3 and all(env is frozen for env in observed)


def test_task_deleted_during_contract_upgrade_fails_closed(live_task, monkeypatch):
    conn, tid, run_id = live_task
    original = store._snapshot
    calls = []
    def snapshot(connection, task_id):
        result = original(connection, task_id)
        calls.append(task_id)
        if len(calls) == 2:
            connection.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        return result
    monkeypatch.setattr(store, "_snapshot", snapshot)
    assert store.prepare_acceptance(conn, tid, run_id,
        {"published_pr": "https://github.com/acme/repo/pull/7"}) is False
