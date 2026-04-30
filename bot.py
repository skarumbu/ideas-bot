#!/usr/bin/env python3
"""
ideas-bot: reads an idea from ideas-api, clones the target repo, uses GPT-4o
to implement it via a tool-use agent loop, opens a draft PR, and writes the
result back to ideas-api.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import requests
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ideas-bot")

# ── Environment variables ──────────────────────────────────────────────────────
IDEA_ID               = os.environ["IDEA_ID"]
IDEAS_API_URL         = os.environ["IDEAS_API_URL"].rstrip("/")
IDEAS_WRITE_KEY       = os.environ["IDEAS_WRITE_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
GITHUB_PAT            = os.environ["GITHUB_PAT"]
GITHUB_USERNAME       = os.environ.get("GITHUB_USERNAME", "skarumbu")

# ── Project → repo mapping ─────────────────────────────────────────────────────
REPO_MAP = {
    "Digits":       "digits",
    "NBA Games":    "momentum_finder",
    "Trail Finder": "trail_finder",
    "Ideas":        "ideas-api",
    "Dashboard":    "dashboard_api",
}

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
        log.error(f"Failed to write bot status: {exc}")


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
            path.write_text(args["content"])
            return f"Written {len(args['content'])} chars to {args['path']}"

        else:
            return f"Error: unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"
    except Exception as exc:
        return f"Error: {exc}"


# ── GPT-4o agent loop ──────────────────────────────────────────────────────────
def run_agent(idea: dict, repo_dir: str) -> None:
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
            max_tokens=4096,
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
        log.error(f"Failed to fetch idea: {exc}")
        set_bot_status("failed", error=f"Could not fetch idea: {exc}")
        sys.exit(1)

    log.info(f"Idea: {idea['title']} (project: {idea.get('project') or idea.get('feature_name', '')})")

    repo_name = REPO_MAP.get(idea.get("project") or idea.get("feature_name", ""), "my-website")
    repo_slug = f"{GITHUB_USERNAME}/{repo_name}"
    log.info(f"Target repo: {repo_slug}")

    set_bot_status("running")

    with tempfile.TemporaryDirectory(prefix="ideas-bot-") as work_dir:
        repo_dir = str(Path(work_dir) / "repo")
        safe_title = re.sub(r"[^a-z0-9]+", "-", idea["title"].lower())[:40].strip("-")
        branch = f"bot/{date.today().isoformat()}-{safe_title}"

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

            # Run GPT-4o agent
            log.info("Running GPT-4o agent…")
            run_agent(idea, repo_dir)

            # Catch any uncommitted stragglers
            uncommitted = capture_cmd(["git", "status", "--porcelain"], cwd=repo_dir)
            if uncommitted:
                run_cmd(["git", "add", "-A"], cwd=repo_dir)
                run_cmd(
                    ["git", "commit", "-m", f"bot: implement '{idea['title']}' (AI-generated)"],
                    cwd=repo_dir,
                )

            # Guard: ensure agent actually made commits
            ahead = capture_cmd(["git", "rev-list", "--count", "origin/main..HEAD"], cwd=repo_dir)
            if ahead == "0":
                raise RuntimeError("Agent made no commits — nothing to push")

            # Push
            run_cmd(
                ["git", "push", "origin", branch],
                cwd=repo_dir,
                extra_env={"GIT_TERMINAL_PROMPT": "0"},
            )

            # Open draft PR via gh CLI (GH_TOKEN env var set from GITHUB_PAT in container)
            pr_url = capture_cmd(
                [
                    "gh", "pr", "create",
                    "--draft",
                    "--base", "main",
                    "--head", branch,
                    "--title", f"bot: {idea['title']}",
                    "--body", build_pr_body(idea),
                ],
                cwd=repo_dir,
            )
            log.info(f"PR created: {pr_url}")

            set_bot_status("completed", pr_url=pr_url.strip())

        except Exception as exc:
            log.error(f"Bot failed: {exc}", exc_info=True)
            set_bot_status("failed", error=str(exc))
            sys.exit(1)


if __name__ == "__main__":
    main()
