"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

import subprocess

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, _GateAuthError, _api, _gh_env, collect_acceptance


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract = snapshot
    if not contract or contract == "local-only":
        return None
    if status not in {"running", "ready", "blocked", "review"} or (expected_run_id is not None and run_id != expected_run_id):
        return False
    published_pr = metadata.get("published_pr") if isinstance(metadata, dict) else None
    match = _PR.fullmatch(published_pr) if isinstance(published_pr, str) else None
    # Publication binds once. Retrying cannot replace the task's PR with a green sibling.
    if match and contract == match[1]:
        with write_txn(conn):
            if _snapshot(conn, task_id) != snapshot:
                return False
            conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (published_pr, task_id))
        snapshot = (run_id, status, published_pr)
        contract = published_pr
    row = conn.execute("SELECT current_run_id,status,claim_lock,assignee,completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        return False
    assignee = row["assignee"]
    bound = _latest_binding(conn, task_id)
    guard = (row["current_run_id"], row["status"], row["claim_lock"], assignee, row["completion_contract"], bound[0] if bound else None)
    if bound:
        _event_id, binding = bound
        if (not _binding_owns_transition(conn, task_id, row, binding) or binding.get("assignee") != assignee
                or binding.get("repo") != _contract_repo(contract)):
            return False
        try:
            from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
            if not all(isinstance(binding.get(key), str) and binding[key].strip()
                       for key in ("profile", "expected_login")):
                raise _GateAuthError("bound identity fields are missing")
            profile = normalize_profile_name(binding["profile"])
            home = resolve_profile_env(profile)
            child_env = _gh_env(home)
            user = _api("user", profile_home=home, child_env=child_env)
            if not isinstance(user, dict) or user.get("login") != binding.get("expected_login"):
                raise _GateAuthError("bound profile login does not match expected identity")
            repo_info = _api(f"repos/{binding['repo']}", profile_home=home, child_env=child_env)
            if not isinstance(repo_info, dict) or repo_info.get("full_name", "").casefold() != binding["repo"].casefold():
                raise _GateAuthError("bound identity cannot access contract repository")
            receipt = collect_acceptance(contract, published_pr, assignee=profile, child_env=child_env)
        except (FileNotFoundError, ValueError, _GateAuthError):
            receipt = {"ok":False,"classification":"auth","head_sha":None,"pr_url":published_pr,"checks":[],
                      "detail":"Bound GitHub identity is unavailable, mismatched, or cannot access the contract repository.",
                      "recovery":"Restore the explicitly bound profile identity and repository access, then retry completion."}
        except (OSError, subprocess.SubprocessError, KeyError, TypeError):
            receipt = {"ok":False,"classification":"infra","head_sha":None,"pr_url":published_pr,"checks":[],
                       "detail":"Bound GitHub identity evidence could not be read; retry the acceptance read.",
                       "recovery":"Restore GitHub API availability and retry completion without changing the bound identity."}
        return snapshot, receipt, guard
    return snapshot, collect_acceptance(contract, published_pr, assignee=assignee), guard


def _binding_owns_transition(conn, task_id, row, binding):
    if row["status"] == "running":
        return (binding.get("run_id") == row["current_run_id"]
                and bool(row["claim_lock"]) and binding.get("owner") == row["claim_lock"])
    if row["status"] != "review" or row["claim_lock"] is not None or row["current_run_id"] is not None:
        return False
    # Only the native handoff of this same bound run may clear its own claim.
    from hermes_cli.kanban_db import _json_dict
    event = conn.execute("SELECT run_id,payload FROM task_events WHERE task_id=? "
                         "AND kind='review_requested' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    run = conn.execute("SELECT profile,outcome,claim_lock FROM task_runs WHERE id=? AND task_id=?",
                       (binding.get("run_id"), task_id)).fetchone()
    payload = _json_dict(event["payload"]) if event else {}
    return (event is not None and event["run_id"] == binding.get("run_id")
            and run is not None and run["outcome"] == "review_requested"
            and run["claim_lock"] == binding.get("owner")
            and run["profile"] == binding.get("assignee") == payload.get("implementer")
            and payload.get("reviewer") in (None, row["assignee"]))


def _contract_repo(contract):
    import re
    match = _PR.fullmatch(contract or "")
    repo = match[1] if match else contract
    return repo if isinstance(repo, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) else None


def _latest_binding(conn, task_id):
    import json
    row = conn.execute("SELECT id,run_id,payload FROM task_events WHERE task_id=? AND kind='external_github_identity_bound' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if not row:
        return None
    try:
        data = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return None
    return (row["id"], data) if isinstance(data, dict) and data.get("run_id") == row["run_id"] else None


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt, guard = acceptance
    row = conn.execute("SELECT current_run_id,status,claim_lock,assignee,completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    bound = _latest_binding(conn, task_id)
    current_guard = (row["current_run_id"], row["status"], row["claim_lock"], row["assignee"], row["completion_contract"], bound[0] if bound else None) if row else None
    if _snapshot(conn, task_id) != snapshot or current_guard != guard:
        return False
    _append_event(conn, task_id, "pr_acceptance", receipt, run_id=snapshot[0])
    if not receipt["ok"]:
        detail = f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]
