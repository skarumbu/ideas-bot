#!/usr/bin/env python3
"""
ideas-bot: reads an idea from ideas-api, clones the target repo, uses GPT-4o
to implement it via a tool-use agent loop, opens a draft PR, and writes the
result back to ideas-api.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path

import openai
import requests
from openai import AzureOpenAI
from shared_logging import get_logger

log = get_logger("ideas-bot")

# ── Environment variables ──────────────────────────────────────────────────────
IDEA_ID               = os.environ["IDEA_ID"]
IDEAS_API_URL         = os.environ["IDEAS_API_URL"].rstrip("/")
IDEAS_WRITE_KEY       = os.environ["IDEAS_WRITE_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "o3-mini")
GITHUB_PAT            = os.environ["GITHUB_PAT"]
GITHUB_USERNAME       = os.environ.get("GITHUB_USERNAME", "skarumbu")


# ── Azure OpenAI client ────────────────────────────────────────────────────────
oai = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-01-preview",
    max_retries=8,       # retry up to 8x with exponential backoff on 429/5xx
    timeout=120.0,       # per-request timeout in seconds
)

def _chat_complete(**kwargs):
    """Wrapper around oai.chat.completions.create with app-level 429 retry."""
    for attempt in range(4):
        try:
            return oai.chat.completions.create(**kwargs)  # noqa: direct call inside wrapper
        except openai.RateLimitError as exc:
            if attempt == 3:
                raise
            wait = 60 * (attempt + 1)
            log.warning(f"Rate limited (attempt {attempt + 1}/4) — sleeping {wait}s: {exc}")
            time.sleep(wait)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the repository working directory. "
                "Returns combined stdout and stderr (capped at 8000 chars)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file by path relative to the repo root. Returns its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (relative to repo root). Creates or overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_idea",
            "description": (
                "Use this when you discover work that must happen in a different repository. "
                "Do NOT clone or modify that repo — call this tool to file a dependency idea. "
                "Provide enough context in 'body' that the next bot can act without referring back. "
                "The current idea will be marked as blocked until the dependency is completed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Exact project name from the available projects list.",
                    },
                    "title": {"type": "string"},
                    "body": {
                        "type": "string",
                        "description": "What to change, why it is needed, and a reference to this idea.",
                    },
                },
                "required": ["project_name", "title", "body"],
            },
        },
    },
]


# ── clean exit with log-flush delay ───────────────────────────────────────────
def _exit(code: int, delay: int = 12) -> None:
    """Sleep before exit so Azure Container Apps log-shipping has time to complete."""
    if code != 0:
        time.sleep(delay)
    sys.exit(code)


# ── ideas-api helpers ──────────────────────────────────────────────────────────
def _machine_headers() -> dict:
    return {"Content-Type": "application/json", "X-Ideas-Key": IDEAS_WRITE_KEY}


def fetch_updates(idea_id: str) -> list[dict]:
    resp = requests.get(
        f"{IDEAS_API_URL}/api/ideas/{idea_id}/updates",
        headers=_machine_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("updates", [])


def post_bot_update(idea_id: str, text: str) -> None:
    requests.post(
        f"{IDEAS_API_URL}/api/ideas/{idea_id}/updates",
        json={"content": text, "author": "bot"},
        headers=_machine_headers(),
        timeout=15,
    ).raise_for_status()


def set_bot_status(status: str, pr_url: str | None = None, error: str | None = None, bot_blocked_by: str | None = None) -> None:
    payload: dict = {"bot_status": status}
    if pr_url is not None:
        payload["bot_pr_url"] = pr_url
    if error is not None:
        payload["bot_error"] = error[:500]
    if bot_blocked_by is not None:
        payload["bot_blocked_by"] = bot_blocked_by
    try:
        resp = requests.patch(
            f"{IDEAS_API_URL}/api/ideas/{IDEA_ID}/bot",
            json=payload,
            headers=_machine_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        log.info(f"set_bot_status → {status}")
    except Exception as exc:
        log.error(f"Failed to write bot status: {exc}", extra={
            "event": "error",
            "endpoint": f"/api/ideas/{IDEA_ID}/bot",
            "method": "PATCH",
            "status": 500,
            "message": f"Failed to write bot status: {exc}",
            "error_type": type(exc).__name__,
            "duration_ms": 0,
        })


def fetch_project_repo(project_name: str) -> str:
    resp = requests.get(
        f"{IDEAS_API_URL}/api/projects",
        headers=_machine_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    for p in resp.json().get("projects", []):
        if p["name"] == project_name:
            return p.get("repo") or ""
    return ""


def fetch_all_projects() -> list[dict]:
    resp = requests.get(
        f"{IDEAS_API_URL}/api/projects",
        headers=_machine_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("projects", [])


def fetch_idea() -> dict:
    resp = requests.get(
        f"{IDEAS_API_URL}/api/ideas",
        headers=_machine_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    for idea in resp.json().get("ideas", []):
        if idea["id"] == IDEA_ID:
            return idea
    raise ValueError(f"Idea {IDEA_ID} not found in response")


# ── subprocess helpers ─────────────────────────────────────────────────────────
def run_cmd(args: list[str], cwd: str, extra_env: dict | None = None, timeout: int = 300) -> None:
    env = {**os.environ, **(extra_env or {})}
    log.info("$ " + " ".join(args))
    result = subprocess.run(args, cwd=cwd, env=env, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(args)}")


def capture_cmd(args: list[str], cwd: str) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


# ── agent tool dispatch ────────────────────────────────────────────────────────
def dispatch_tool(tool_call, repo_dir: str, all_projects: list[dict] | None = None, created_ideas: list[dict] | None = None) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "Error: could not parse tool arguments as JSON"

    try:
        if name == "bash":
            result = subprocess.run(
                args["command"],
                shell=True,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            output = result.stdout + result.stderr
            return output[:8000] if output else "(no output)"

        elif name == "read_file":
            path = Path(repo_dir) / args["path"]
            if not path.exists():
                return f"Error: file not found: {args['path']}"
            content = path.read_text(errors="replace")
            return content[:8000] if len(content) > 8000 else content

        elif name == "write_file":
            path = Path(repo_dir) / args["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args["content"].replace('\x00', '')
            path.write_text(content)
            return f"Written {len(content)} chars to {args['path']}"

        elif name == "create_idea":
            proj_name = args.get("project_name", "")
            title = args.get("title", "")
            body = args.get("body", "")
            projects = all_projects or []
            target = next((p for p in projects if p["name"] == proj_name), None)
            if not target:
                available = [p["name"] for p in projects]
                return f"Error: project '{proj_name}' not found. Available: {available}. Retry with an exact name."
            resp = requests.post(
                f"{IDEAS_API_URL}/api/ideas",
                headers=_machine_headers(),
                json={
                    "project": proj_name,
                    "project_id": target["id"],
                    "title": title,
                    "body": body,
                    "source": f"bot:cross-repo:{IDEA_ID}",
                },
                timeout=15,
            )
            if resp.status_code == 201:
                new_idea = resp.json()
                if created_ideas is not None:
                    created_ideas.append({"id": new_idea["id"], "project": proj_name, "title": title})
                return (
                    f"Dependency idea created (ID: {new_idea['id']}) in project '{proj_name}'. "
                    "The current idea will be blocked until this dependency is completed."
                )
            return f"Error creating idea: HTTP {resp.status_code} — {resp.text[:200]}"

        else:
            return f"Error: unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"
    except Exception as exc:
        return f"Error: {exc}"


# ── pre-flight clarity check ───────────────────────────────────────────────────
def assess_idea_clarity(idea: dict) -> tuple[bool, str | None]:
    response = _chat_complete(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You assess software feature ideas for autonomous implementation clarity. "
                    "The project is a personal web app (React + TypeScript frontend, Python Azure "
                    "Functions backend). Be permissive — most UI tweaks, new pages, and small "
                    "features have an obvious implementation. Only ask when the answer would "
                    "meaningfully fork the code path (e.g. which data to fetch, what the primary "
                    "action is, whether backend changes are needed)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Feature idea:\n"
                    f"Title: {idea['title']}\n"
                    f"Project: {idea.get('project', '')}\n"
                    f"Description: {idea.get('body', '') or '(none)'}\n\n"
                    f"Is this implementable without clarification?\n\n"
                    f"Rules:\n"
                    f"- Return clear=true if a skilled developer could make a reasonable "
                    f"implementation decision on their own\n"
                    f"- Only return clear=false when missing info would cause significantly "
                    f"different code — e.g. unknown data source, ambiguous primary action, "
                    f"unclear scope (1 page vs 5 pages)\n"
                    f"- Do NOT ask about style, color, copy, or other preferences with obvious defaults\n"
                    f"- If asking, make the question concrete and answerable in 1-2 sentences\n\n"
                    f"Reply with JSON only:\n"
                    f"{{\"clear\": true}} OR {{\"clear\": false, \"question\": \"<specific question>\"}}"
                ),
            },
        ],
        max_tokens=300,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    return data.get("clear", True), data.get("question")


# ── idea decomposition ────────────────────────────────────────────────────────
def decompose_idea(idea: dict, repo: str) -> list[dict] | None:
    """Returns [{title, body},...] if the idea should be split, or None if simple."""
    if (idea.get("source") or "").startswith("bot:subtask:"):
        return None  # never decompose a child task

    response = _chat_complete(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer planning implementation work. "
                    "Decide whether a feature idea should be implemented in one focused session "
                    "or broken into 2–4 sequential sub-tasks. Only split when genuinely necessary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Feature idea:\n"
                    f"Title: {idea['title']}\n"
                    f"Project: {idea.get('project', '')}\n"
                    f"Repo: {repo}\n"
                    f"Description: {idea.get('body', '') or '(none)'}\n\n"
                    "Keep as ONE task if: small UI change, single endpoint, minor refactor, "
                    "or a focused change in one area.\n"
                    "SPLIT into 2–4 tasks if: the idea mixes unrelated subsystems "
                    "(e.g. DB schema + API + UI), or would require 5+ unrelated commits, "
                    "or involves both backend and frontend changes.\n\n"
                    "Each sub-task title must be specific and self-contained. "
                    f"Each sub-task body must include what to implement and end with: "
                    f"'Part of: {idea['title']} (idea #{IDEA_ID})'\n\n"
                    "Reply with JSON only:\n"
                    '{"decompose": false} OR '
                    '{"decompose": true, "subtasks": [{"title": "...", "body": "..."}]}'
                ),
            },
        ],
        max_tokens=800,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    if not data.get("decompose"):
        return None
    subtasks = data.get("subtasks") or []
    return subtasks if subtasks else None


# ── CI watching + repair ───────────────────────────────────────────────────────
CI_TIMEOUT_SECS = int(os.environ.get("BOT_CI_TIMEOUT_SECS", "600"))


def _wait_for_checks(repo_slug: str, pr_number: str, repo_dir: str) -> tuple[bool, str]:
    """Poll gh pr checks until all complete. Returns (all_passed, failure_log)."""
    deadline = time.time() + CI_TIMEOUT_SECS
    poll_secs = 30
    while time.time() < deadline:
        time.sleep(poll_secs)
        result = subprocess.run(
            ["gh", "pr", "checks", pr_number, "--repo", repo_slug,
             "--json", "name,state,conclusion"],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning(f"gh pr checks error: {result.stderr[:200]}")
            continue

        checks = json.loads(result.stdout or "[]")
        if not checks:
            continue

        pending = [c for c in checks if c["state"] not in ("COMPLETED", "SUCCESS", "FAILURE", "ERROR", "CANCELLED", "SKIPPED", "TIMED_OUT")]
        if pending:
            log.info(f"CI: {len(pending)} check(s) still running…")
            continue

        failures = [c for c in checks if c.get("conclusion") not in ("SUCCESS", "SKIPPED", "NEUTRAL", None)]
        if not failures:
            log.info("CI: all checks passed")
            return True, ""

        log.warning(f"CI: {len(failures)} check(s) failed — fetching logs")
        failure_log = _fetch_failure_logs(repo_slug, repo_dir)
        return False, failure_log

    return False, f"CI checks did not complete within {CI_TIMEOUT_SECS}s"


def _fetch_failure_logs(repo_slug: str, repo_dir: str) -> str:
    """Return failed-step logs from the most recent workflow run on the current branch."""
    result = subprocess.run(
        ["gh", "run", "list", "--repo", repo_slug, "--limit", "1",
         "--json", "databaseId,conclusion"],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "Could not retrieve CI run list"
    runs = json.loads(result.stdout)
    if not runs:
        return "No CI runs found"
    run_id = str(runs[0]["databaseId"])
    log_result = subprocess.run(
        ["gh", "run", "view", run_id, "--repo", repo_slug, "--log-failed"],
        cwd=repo_dir, capture_output=True, text=True, timeout=60,
    )
    output = (log_result.stdout + log_result.stderr)
    return output[-4000:] if output else "No log output retrieved"


def _repair_from_ci(repo_dir: str, failure_log: str) -> bool:
    """Spawn a targeted repair agent using CI failure output. Returns True if commits were made."""
    repair_messages: list = [
        {
            "role": "system",
            "content": (
                "You are an expert software engineer. CI failed on your pull request. "
                "Diagnose the failure from the logs, fix the code using the provided tools, "
                "then commit: git add -A && git commit -m 'bot: fix CI failure'"
            ),
        },
        {
            "role": "user",
            "content": f"CI failed with these logs:\n\n```\n{failure_log}\n```",
        },
    ]
    commits_before = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
    ).stdout.strip()

    for round_num in range(20):
        log.info(f"Repair round {round_num + 1}")
        response = _chat_complete(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=repair_messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8096,
        )
        msg = response.choices[0].message
        repair_messages.append(msg)
        if not msg.tool_calls:
            log.info(f"Repair agent done: {msg.content[:200] if msg.content else '(no text)'}")
            break
        for tc in msg.tool_calls:
            tool_result = dispatch_tool(tc, repo_dir)
            repair_messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})

    commits_after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
    ).stdout.strip()
    return commits_after != commits_before


def watch_ci_and_repair(repo_slug: str, pr_number: str, repo_dir: str, branch: str) -> None:
    """Watch GitHub Actions on the PR. On failure: repair, push, watch once more."""
    passed, failure_log = _wait_for_checks(repo_slug, pr_number, repo_dir)
    if passed:
        return

    log.warning("CI failed — running repair agent")
    made_commits = _repair_from_ci(repo_dir, failure_log)
    if not made_commits:
        raise RuntimeError(f"CI failed and repair agent made no commits.\n{failure_log[-1000:]}")

    run_cmd(
        ["git", "push", "origin", branch],
        cwd=repo_dir,
        extra_env={"GIT_TERMINAL_PROMPT": "0"},
    )
    log.info("Repair pushed — waiting for CI again")
    passed2, failure_log2 = _wait_for_checks(repo_slug, pr_number, repo_dir)
    if not passed2:
        raise RuntimeError(f"CI still failing after repair.\n{failure_log2[-1000:]}")


# ── agent loop ─────────────────────────────────────────────────────────────────
def run_agent(idea: dict, repo_dir: str, prior_updates: list[dict], all_projects: list[dict] | None = None, created_ideas: list[dict] | None = None) -> None:
    system = (
        "You are an expert software engineer implementing a feature from a backlog idea.\n"
        "Use the provided tools to explore the repo structure, understand the codebase, "
        "then implement the feature described by the user.\n"
        "Requirements:\n"
        "- Make production-quality changes only to files relevant to this feature\n"
        "- Do NOT refactor unrelated code\n"
        "- Write or update tests where the project already has them\n"
        "- If you add a new import or dependency, check package.json first to confirm it is "
        "already listed; if not, run: npm install <package> --save (or --save-dev for types)\n"
        "- When done, commit: git add -A && git commit -m 'bot: <concise summary>'\n"
        "- Do NOT push — the orchestrator handles that\n"
        "- After committing, stop calling tools and give a short plain-text summary\n"
        "- If completing this idea requires work in a different repository, call create_idea to "
        "file that work as a dependency — do not clone or modify other repos yourself"
    )
    user_msg = (
        f"Project: {idea.get('project') or idea.get('feature_name', '')}\n"
        f"Title: {idea['title']}\n\n"
        f"{idea.get('body', '')}"
    )
    if prior_updates:
        thread = "\n".join(f"[{u.get('author_name') or u.get('author_email', 'unknown')}]: {u['content']}" for u in prior_updates)
        user_msg += f"\n\n## Prior conversation\n{thread}\n\nUse the user's answers above to guide your implementation."
    if all_projects:
        proj_list = "\n".join(f"  - {p['name']} (repo: {p.get('repo') or 'none'})" for p in all_projects)
        user_msg += f"\n\n## Available projects for cross-repo dependencies\n{proj_list}"
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    for round_num in range(40):
        log.info(f"Agent round {round_num + 1}")
        # Keep system + user prompt + last N exchanges to bound context size.
        # Tool results can be large; trimming older ones reduces 429 risk.
        context = messages[:2] + messages[2:][-30:] if len(messages) > 32 else messages
        response = _chat_complete(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=context,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8096,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            log.info(f"Agent finished: {msg.content[:200] if msg.content else '(no text)'}")
            break

        for tc in msg.tool_calls:
            log.info(f"  tool: {tc.function.name}({tc.function.arguments[:120]})")
            result = dispatch_tool(tc, repo_dir, all_projects=all_projects, created_ideas=created_ideas)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        raise RuntimeError("Agent exceeded 40 rounds without finishing")


# ── PR body ────────────────────────────────────────────────────────────────────
def build_pr_body(idea: dict) -> str:
    return (
        f"## AI-generated implementation\n\n"
        f"This PR was created automatically by **ideas-bot** from the Ideas board.\n\n"
        f"**Idea:** {idea['title']}  \n"
        f"**Project:** {idea.get('project') or idea.get('feature_name', '')}  \n"
        f"**Idea ID:** `{idea['id']}`\n\n"
        f"### Description\n\n"
        f"{idea.get('body', '_No description provided._')}\n\n"
        f"---\n\n"
        f"### Review checklist\n\n"
        f"- [ ] Implementation matches the intent above\n"
        f"- [ ] No unrelated files were modified\n"
        f"- [ ] Tests pass (if applicable)\n"
        f"- [ ] Code style is consistent with the rest of the codebase\n\n"
        f"/cc @{GITHUB_USERNAME}"
    )


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info(f"ideas-bot starting for idea {IDEA_ID}")

    try:
        idea = fetch_idea()
    except Exception as exc:
        log.error(f"Failed to fetch idea: {exc}", extra={
            "event": "error",
            "endpoint": "/api/ideas",
            "method": "GET",
            "status": 500,
            "message": f"Failed to fetch idea: {exc}",
            "error_type": type(exc).__name__,
            "duration_ms": 0,
        })
        set_bot_status("failed", error=f"Could not fetch idea: {exc}")
        _exit(1)

    log.info(f"Idea: {idea['title']} (project: {idea.get('project') or idea.get('feature_name', '')})")

    project_name = idea.get("project") or idea.get("feature_name", "")
    repo = fetch_project_repo(project_name)
    if not repo:
        set_bot_status("failed", error=f"Project '{project_name}' has no repo configured. Set a repo in the Ideas app under Manage Projects.")
        log.error(f"No repo configured for project '{project_name}'")
        _exit(1)
    log.info(f"Target repo: {repo}")

    try:
        all_projects = fetch_all_projects()
    except Exception as exc:
        log.warning(f"Could not fetch all projects ({exc}), cross-repo tool will be unavailable")
        all_projects = []

    # If previously blocked, check whether all dependencies are now complete
    blocked_by_raw = idea.get("bot_blocked_by") or ""
    if blocked_by_raw:
        try:
            blocking = json.loads(blocked_by_raw)
        except Exception:
            blocking = []
        if blocking:
            try:
                all_ideas_resp = requests.get(f"{IDEAS_API_URL}/api/ideas", headers=_machine_headers(), timeout=15)
                all_ideas_resp.raise_for_status()
                all_ideas_map = {i["id"]: i for i in all_ideas_resp.json().get("ideas", [])}
            except Exception as exc:
                log.error(f"Could not fetch ideas for dependency check: {exc}")
                set_bot_status("blocked", error="Could not verify dependencies", bot_blocked_by=blocked_by_raw)
                _exit(0)
            still_blocking = [b for b in blocking if all_ideas_map.get(b["id"], {}).get("bot_status") != "completed"]
            if still_blocking:
                names = ", ".join(f"[{b['project']}] {b['title']}" for b in still_blocking)
                log.info(f"Still blocked by: {names}")
                set_bot_status("blocked", error=f"Waiting on: {names}", bot_blocked_by=json.dumps(still_blocking))
                _exit(0)
            else:
                log.info("All dependencies completed — clearing block and proceeding")
                all_were_subtasks = all(b.get("type") == "subtask" for b in blocking)
                if all_were_subtasks:
                    child_refs = ", ".join(f"{b['title']} (#{b['id'][:8]})" for b in blocking)
                    set_bot_status(
                        "completed",
                        bot_blocked_by="",
                        error=f"Completed via {len(blocking)} sub-task(s): {child_refs}",
                    )
                    log.info("All sub-tasks done — parent marked complete")
                    _exit(0)
                else:
                    # Cross-repo deps: clear block and proceed to agent loop
                    set_bot_status("running", bot_blocked_by="")

    created_ideas: list[dict] = []

    if not blocked_by_raw:
        set_bot_status("running")

    try:
        is_clear, question = assess_idea_clarity(idea)
    except Exception as exc:
        log.warning(f"Clarity check failed ({exc}), proceeding anyway")
        is_clear, question = True, None

    if not is_clear and question:
        post_bot_update(
            IDEA_ID,
            f"I need more information before I can implement this:\n\n{question}\n\nPlease reply and re-trigger the bot.",
        )
        set_bot_status("needs_info")
        log.info("Idea needs clarification — bot pausing")
        _exit(0)

    # Decompose complex ideas into focused sub-tasks to avoid rate limits
    is_subtask = (idea.get("source") or "").startswith("bot:subtask:")
    if not is_subtask:
        try:
            subtasks = decompose_idea(idea, repo)
        except Exception as exc:
            log.warning(f"Decompose check failed ({exc}), proceeding as single task")
            subtasks = None

        if subtasks:
            this_project = next((p for p in all_projects if p["name"] == project_name), None)
            child_ideas = []
            for st in subtasks:
                resp = requests.post(
                    f"{IDEAS_API_URL}/api/ideas",
                    headers=_machine_headers(),
                    json={
                        "project": project_name,
                        "project_id": this_project["id"] if this_project else idea.get("project_id"),
                        "title": st["title"],
                        "body": st["body"],
                        "source": f"bot:subtask:{IDEA_ID}",
                    },
                    timeout=15,
                )
                if resp.status_code == 201:
                    new_idea = resp.json()
                    child_ideas.append({
                        "id": new_idea["id"],
                        "project": project_name,
                        "title": st["title"],
                        "type": "subtask",
                    })

            if child_ideas:
                lines = "\n".join(f"  {i+1}. {c['title']} (ID: {c['id']})" for i, c in enumerate(child_ideas))
                post_bot_update(
                    IDEA_ID,
                    f"Decomposed into {len(child_ideas)} sub-tasks to reduce API load:\n{lines}\n\n"
                    "Each will run as a separate bot job. Re-trigger this idea once all are completed.",
                )
                set_bot_status(
                    "blocked",
                    error=f"Waiting for {len(child_ideas)} sub-task(s) to complete",
                    bot_blocked_by=json.dumps(child_ideas),
                )
                log.info(f"Decomposed into {len(child_ideas)} sub-tasks")
                _exit(0)

    safe_title = re.sub(r"[^a-z0-9]+", "-", idea["title"].lower())[:40].strip("-")
    model_slug = re.sub(r"[^a-z0-9]+", "-", AZURE_OPENAI_DEPLOYMENT.lower())
    branch = f"bot/{date.today().isoformat()}-{model_slug}-{safe_title}"
    prior_updates = fetch_updates(IDEA_ID)

    with tempfile.TemporaryDirectory(prefix="ideas-bot-") as work_dir:
        repo_dir = str(Path(work_dir) / "repo")

        try:
            log.info(f"Cloning {repo}…")
            run_cmd(
                ["git", "clone",
                 f"https://x-access-token:{GITHUB_PAT}@github.com/{repo}.git",
                 "repo"],
                cwd=work_dir,
                timeout=120,
            )

            run_cmd(["git", "config", "user.name", "ideas-bot[bot]"], cwd=repo_dir)
            run_cmd(["git", "config", "user.email", "ideas-bot[bot]@users.noreply.github.com"], cwd=repo_dir)
            run_cmd(["git", "checkout", "-b", branch], cwd=repo_dir)

            log.info(f"Running agent ({AZURE_OPENAI_DEPLOYMENT}) on {repo}…")
            run_agent(idea, repo_dir, prior_updates, all_projects=all_projects, created_ideas=created_ideas)

            # If the agent created cross-repo dependencies, block and don't push
            if created_ideas:
                dep_lines = "\n".join(f"  - [{d['project']}] {d['title']} (ID: {d['id']})" for d in created_ideas)
                post_bot_update(IDEA_ID, f"Blocked — created the following dependency ideas:\n{dep_lines}\n\nRe-trigger once these are completed.")
                set_bot_status("blocked", error=f"Waiting on {len(created_ideas)} dependency idea(s)", bot_blocked_by=json.dumps(created_ideas))
                log.info(f"Blocked on {len(created_ideas)} dependency idea(s)")
                _exit(0)

            uncommitted = capture_cmd(["git", "status", "--porcelain"], cwd=repo_dir)
            if uncommitted:
                run_cmd(["git", "add", "-A"], cwd=repo_dir)
                run_cmd(
                    ["git", "commit", "-m", f"bot: implement '{idea['title']}' (AI-generated)"],
                    cwd=repo_dir,
                )

            ahead = capture_cmd(["git", "rev-list", "--count", "HEAD", "--not", "--remotes"], cwd=repo_dir)
            if ahead == "0":
                raise RuntimeError(f"Agent made no commits in {repo} — nothing to push")

            run_cmd(
                ["git", "push", "origin", branch],
                cwd=repo_dir,
                extra_env={"GIT_TERMINAL_PROMPT": "0"},
            )

            pr_url = capture_cmd(
                [
                    "gh", "pr", "create",
                    "--draft",
                    "--base", "main",
                    "--head", branch,
                    "--title", f"bot [{AZURE_OPENAI_DEPLOYMENT}]: {idea['title']}",
                    "--body", build_pr_body(idea),
                ],
                cwd=repo_dir,
            ).strip()
            log.info(f"PR created: {pr_url}")

            pr_number = pr_url.rstrip("/").split("/")[-1]
            watch_ci_and_repair(repo, pr_number, repo_dir, branch)

        except Exception as exc:
            log.error(str(exc), extra={
                "event": "error",
                "endpoint": "/job/ideas-bot",
                "method": "JOB",
                "status": 500,
                "message": str(exc),
                "error_type": type(exc).__name__,
                "stack_trace": traceback.format_exc()[:2000],
                "duration_ms": 0,
            })
            set_bot_status("failed", error=str(exc))
            _exit(1)

    set_bot_status("completed", pr_url=pr_url)


if __name__ == "__main__":
    main()
