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
import traceback
from datetime import date
from pathlib import Path

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
)

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
]


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


def set_bot_status(status: str, pr_url: str | None = None, error: str | None = None) -> None:
    payload: dict = {"bot_status": status}
    if pr_url is not None:
        payload["bot_pr_url"] = pr_url
    if error is not None:
        payload["bot_error"] = error[:500]
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


def fetch_project_repos(project_name: str) -> list[str]:
    resp = requests.get(
        f"{IDEAS_API_URL}/api/projects",
        headers=_machine_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    for p in resp.json().get("projects", []):
        if p["name"] == project_name:
            return p.get("repos") or []
    return []


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
def dispatch_tool(tool_call, repo_dir: str) -> str:
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

        else:
            return f"Error: unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"
    except Exception as exc:
        return f"Error: {exc}"


# ── pre-flight clarity check ───────────────────────────────────────────────────
def assess_idea_clarity(idea: dict) -> tuple[bool, str | None]:
    response = oai.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You assess software feature ideas for autonomous implementation clarity.",
            },
            {
                "role": "user",
                "content": (
                    f"Can this feature idea be implemented autonomously without clarification?\n\n"
                    f"Title: {idea['title']}\n"
                    f"Project: {idea.get('project', '')}\n"
                    f"Description: {idea.get('body', '') or '(none)'}\n\n"
                    f"Reply with JSON only: {{\"clear\": true}} if implementable as-is, "
                    f"or {{\"clear\": false, \"question\": \"your specific question\"}} if not. "
                    f"Be permissive — only flag when truly ambiguous."
                ),
            },
        ],
        max_tokens=256,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    return data.get("clear", True), data.get("question")


# ── agent loop ─────────────────────────────────────────────────────────────────
def run_agent(idea: dict, repo_dir: str, prior_updates: list[dict]) -> None:
    system = (
        "You are an expert software engineer implementing a feature from a backlog idea.\n"
        "Use the provided tools to explore the repo structure, understand the codebase, "
        "then implement the feature described by the user.\n"
        "Requirements:\n"
        "- Make production-quality changes only to files relevant to this feature\n"
        "- Do NOT refactor unrelated code\n"
        "- Write or update tests where the project already has them\n"
        "- When done, run: git add -A && git commit -m 'bot: <concise summary>'\n"
        "- Do NOT push — the orchestrator handles that\n"
        "- After committing, stop calling tools and give a short plain-text summary"
    )
    user_msg = (
        f"Project: {idea.get('project') or idea.get('feature_name', '')}\n"
        f"Title: {idea['title']}\n\n"
        f"{idea.get('body', '')}"
    )
    if prior_updates:
        thread = "\n".join(f"[{u.get('author_name') or u.get('author_email', 'unknown')}]: {u['content']}" for u in prior_updates)
        user_msg += f"\n\n## Prior conversation\n{thread}\n\nUse the user's answers above to guide your implementation."
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    for round_num in range(40):
        log.info(f"Agent round {round_num + 1}")
        response = oai.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
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
            result = dispatch_tool(tc, repo_dir)
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
        sys.exit(1)

    log.info(f"Idea: {idea['title']} (project: {idea.get('project') or idea.get('feature_name', '')})")

    project_name = idea.get("project") or idea.get("feature_name", "")
    repos = fetch_project_repos(project_name)
    if not repos:
        set_bot_status("failed", error=f"Project '{project_name}' has no repos configured. Add repos in the Ideas app.")
        log.error(f"No repos configured for project '{project_name}'")
        sys.exit(1)
    log.info(f"Target repos: {repos}")

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
        sys.exit(0)

    safe_title = re.sub(r"[^a-z0-9]+", "-", idea["title"].lower())[:40].strip("-")
    model_slug = re.sub(r"[^a-z0-9]+", "-", AZURE_OPENAI_DEPLOYMENT.lower())
    branch = f"bot/{date.today().isoformat()}-{model_slug}-{safe_title}"
    prior_updates = fetch_updates(IDEA_ID)
    pr_urls = []

    for repo_slug in repos:
        with tempfile.TemporaryDirectory(prefix="ideas-bot-") as work_dir:
            repo_dir = str(Path(work_dir) / "repo")

            try:
                # Clone
                log.info(f"Cloning {repo_slug}…")
                run_cmd(
                    ["git", "clone",
                     f"https://x-access-token:{GITHUB_PAT}@github.com/{repo_slug}.git",
                     "repo"],
                    cwd=work_dir,
                    timeout=120,
                )

                # Git identity
                run_cmd(["git", "config", "user.name", "ideas-bot[bot]"], cwd=repo_dir)
                run_cmd(["git", "config", "user.email", "ideas-bot[bot]@users.noreply.github.com"], cwd=repo_dir)

                # Feature branch
                run_cmd(["git", "checkout", "-b", branch], cwd=repo_dir)

                # Run agent
                log.info(f"Running agent ({AZURE_OPENAI_DEPLOYMENT}) on {repo_slug}…")
                run_agent(idea, repo_dir, prior_updates)

                # Catch any uncommitted stragglers
                uncommitted = capture_cmd(["git", "status", "--porcelain"], cwd=repo_dir)
                if uncommitted:
                    run_cmd(["git", "add", "-A"], cwd=repo_dir)
                    run_cmd(
                        ["git", "commit", "-m", f"bot: implement '{idea['title']}' (AI-generated)"],
                        cwd=repo_dir,
                    )

                # Guard: ensure agent actually made commits
                ahead = capture_cmd(["git", "rev-list", "--count", "HEAD", "--not", "--remotes"], cwd=repo_dir)
                if ahead == "0":
                    raise RuntimeError(f"Agent made no commits in {repo_slug} — nothing to push")

                # Push
                run_cmd(
                    ["git", "push", "origin", branch],
                    cwd=repo_dir,
                    extra_env={"GIT_TERMINAL_PROMPT": "0"},
                )

                # Open draft PR
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
                )
                log.info(f"PR created: {pr_url}")
                pr_urls.append(pr_url.strip())

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
                sys.exit(1)

    set_bot_status("completed", pr_url=pr_urls[0])


if __name__ == "__main__":
    main()
