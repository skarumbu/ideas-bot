# Ideas Bot — Agent Guidelines

## Posting to the Writing API

The `posts-api/post.py` CLI lets you (or the bot) create posts on quixotry.me
without going through the browser.

### Required env vars

| Variable | Example | Purpose |
|---|---|---|
| `POSTS_API_BASE_URL` | `https://posts-api-xxx.azurewebsites.net` | posts-api host |
| `POSTS_AZURE_TENANT_ID` | `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` | Azure AD tenant GUID |
| `POSTS_TOKEN_CACHE` | `/home/user/.posts-cli-cache.json` | Token cache path (optional, defaults to `~/.posts-cli-cache.json`) |

### One-time setup (human must do this before the bot can use it)

```bash
pip install -r posts-api/requirements-cli.txt
export POSTS_API_BASE_URL=https://...
export POSTS_AZURE_TENANT_ID=<tenant-guid>
python posts-api/post.py login   # opens device-code browser prompt
```

The `login` command writes a token cache to `POSTS_TOKEN_CACHE`. MSAL
refreshes the access token silently on each call — the browser prompt only
appears on the first login or if the refresh token expires (~90 days of
inactivity).

### Calling the CLI from the bot (headless, cache must be warm)

```python
import subprocess
import os

def post_to_writing_api(title: str, description: str, body: str, published: bool = False) -> str:
    """Creates a post via the CLI. Returns the new post slug."""
    cmd = [
        "python", "posts-api/post.py", "create",
        "--title", title,
        "--description", description,
        "--body", body,
    ]
    if published:
        cmd.append("--published")
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        raise RuntimeError(f"post.py failed: {result.stderr.strip()}")
    # output: "Created: <slug>"
    return result.stdout.strip().removeprefix("Created: ")
```

### Available sub-commands

```bash
python posts-api/post.py login                          # re-auth (force new device-code flow)
python posts-api/post.py list                           # list published posts
python posts-api/post.py create --title T --description D [--body B] [--published]
python posts-api/post.py update <slug> --title T --description D [--body B] [--published]
python posts-api/post.py delete <slug>
```
