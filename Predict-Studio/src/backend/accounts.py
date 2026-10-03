"""
accounts.py — user accounts, passwords, login sessions and access tokens.

Accounts and sessions live in one small SQLite file, data/accounts.db. It
records WHO may log in, never WHAT they own: a user's scans and results are
simply the folder paths.user_root(user_id). One source of truth, nothing to
keep in sync.

Access tokens are handed out by the administrator in the environment variable
PREDICT_ACCESS_TOKENS (comma-separated, one per person or group). A token is
needed to create an account AND at every login; each session remembers which
token it used and stops working as soon as that token is removed from the
variable (after a server restart). The admin account needs no token, so it can
never be locked out.

Security notes, all deliberate:
  * passwords: scrypt + a random salt per user; compared in constant time
  * session tokens: random, sent to the browser once; only a SHA-256 is stored,
    so a copied database file holds no usable session
  * an unknown username costs the same hashing time as a known one, and every
    failure gives the same message, so attempts do not reveal which names exist
  * ids are never reused (AUTOINCREMENT), so a new account can never inherit
    a deleted account's folder

Does NOT: speak HTTP (server.py does) or touch scans/results, except when the
admin deletes an account or adopts the old shared data from the CLI.
Called by: server.py, and as a CLI:

    python -m src.backend.accounts create-admin <username>
    python -m src.backend.accounts list
    python -m src.backend.accounts reset-password <username>
    python -m src.backend.accounts delete <username>
    python -m src.backend.accounts adopt-legacy <username>
"""
import getpass
import hashlib
import hmac
import os
import re
import secrets
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:          # allow `python src/backend/accounts.py`
    sys.path.insert(0, str(_project_root))

from src.backend import paths

TOKENS_ENV = "PREDICT_ACCESS_TOKENS"
SESSION_DAYS = 7
MAX_ACCOUNTS_PER_TOKEN = 50     # a leaked token can create at most this many accounts
FAILED_LOGIN_DELAY = 0.5        # seconds; the same for every kind of failure
MIN_PASSWORD = 8
USERNAME_RE = re.compile(r"[a-z0-9._@+-]{3,64}")   # emails work

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,   -- folder name: data/users/<id>
    username    TEXT UNIQUE NOT NULL,
    salt        BLOB NOT NULL,
    pw_hash     BLOB NOT NULL,
    is_admin    INTEGER NOT NULL DEFAULT 0,
    access_hash TEXT,                                -- token used at sign-up
    created     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    access_hash TEXT,                                -- token used at login; NULL for admin
    expires     TEXT NOT NULL
);
"""


class AuthError(Exception):
    """A refusal the user may read: bad password, closed sign-up, taken name."""


# ── storage ───────────────────────────────────────────────────────────────
def db_path() -> Path:
    return paths.DATA / "accounts.db"


@contextmanager
def db():
    """One short connection per call: safe across server threads and the CLI."""
    db_path().parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path(), timeout=10)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        with con:                     # one transaction: commit, or roll back on error
            yield con
    finally:
        con.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pw_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


_DUMMY_SALT = secrets.token_bytes(16)    # for unknown usernames: same work, no match


# ── access tokens ─────────────────────────────────────────────────────────
def access_hashes() -> set[str]:
    """Hashes of the tokens currently granted in PREDICT_ACCESS_TOKENS."""
    raw = os.environ.get(TOKENS_ENV, "")
    return {_sha(t.strip()) for t in raw.split(",") if t.strip()}


def token_ok(access_token: str) -> bool:
    return _sha((access_token or "").strip()) in access_hashes()


# ── accounts ──────────────────────────────────────────────────────────────
def clean_username(username: str) -> str:
    name = (username or "").strip().lower()
    if not USERNAME_RE.fullmatch(name):
        raise AuthError("Username must be 3-64 characters: letters, digits and . _ @ + -")
    return name


def _check_password(password: str):
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")


def create_user(username: str, password: str, access_token: str = "", is_admin: bool = False) -> int:
    """Create an account and return its id. Users need a granted token; the
    admin (created from the CLI) does not."""
    name = clean_username(username)
    _check_password(password)
    access = None
    if not is_admin:
        if not access_hashes():
            raise AuthError("Sign-up is closed: no access tokens are configured.")
        if not token_ok(access_token):
            raise AuthError("Invalid access token.")
        access = _sha(access_token.strip())
    salt = secrets.token_bytes(16)
    with db() as con:
        if access:
            used = con.execute("SELECT COUNT(*) FROM users WHERE access_hash = ?", (access,)).fetchone()[0]
            if used >= MAX_ACCOUNTS_PER_TOKEN:
                raise AuthError("This access token has reached its account limit. "
                                "Ask the administrator for a new one.")
        try:
            cur = con.execute(
                "INSERT INTO users (username, salt, pw_hash, is_admin, access_hash, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, salt, _pw_hash(password, salt), int(is_admin), access, _now()))
        except sqlite3.IntegrityError:
            raise AuthError("That username is taken.")
    return cur.lastrowid


def _public(row) -> dict:
    return {"id": row["id"], "username": row["username"], "is_admin": bool(row["is_admin"])}


def login(username: str, password: str, access_token: str = "") -> tuple[str, dict]:
    """Check credentials and open a session. Returns (session token, user).

    Every failure (unknown name, wrong password, missing/revoked token) takes
    the same time and gives the same message."""
    name = (username or "").strip().lower()
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
    expected = row["pw_hash"] if row else b"\0" * 32
    got = _pw_hash(password or "", row["salt"] if row else _DUMMY_SALT)
    ok = row is not None and hmac.compare_digest(got, expected)

    access = None
    if ok and not row["is_admin"]:
        ok = token_ok(access_token)
        access = _sha((access_token or "").strip())
    if not ok:
        time.sleep(FAILED_LOGIN_DELAY)
        raise AuthError("Invalid username, password or access token.")

    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")
    with db() as con:
        con.execute("DELETE FROM sessions WHERE expires < ?", (_now(),))
        con.execute("INSERT INTO sessions (token_hash, user_id, access_hash, expires) VALUES (?, ?, ?, ?)",
                    (_sha(token), row["id"], access, expires))
    return token, _public(row)


def user_for_session(token: str | None) -> dict | None:
    """The user behind a session token, or None if it is unknown, expired, or
    its access token has been revoked since login."""
    if not token:
        return None
    with db() as con:
        r = con.execute(
            "SELECT u.id, u.username, u.is_admin, s.access_hash, s.expires "
            "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
            (_sha(token),)).fetchone()
    if r is None or r["expires"] < _now():
        return None
    if not r["is_admin"] and r["access_hash"] not in access_hashes():
        return None
    return _public(r)


def logout(token: str | None):
    if token:
        with db() as con:
            con.execute("DELETE FROM sessions WHERE token_hash = ?", (_sha(token),))


def user_exists(user_id: int) -> bool:
    with db() as con:
        return con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is not None


def _find(username: str):
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE username = ?",
                          ((username or "").strip().lower(),)).fetchone()
    if row is None:
        sys.exit(f"No account named {username!r}. See: python -m src.backend.accounts list")
    return row


def set_password(username: str, password: str):
    """New password, and every open session of that account is ended."""
    _check_password(password)
    row = _find(username)
    salt = secrets.token_bytes(16)
    with db() as con:
        con.execute("UPDATE users SET salt = ?, pw_hash = ? WHERE id = ?",
                    (salt, _pw_hash(password, salt), row["id"]))
        con.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))


# ── CLI ───────────────────────────────────────────────────────────────────
def _ask_password() -> str:
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return first


def cmd_create_admin(username: str):
    try:
        uid = create_user(username, _ask_password(), is_admin=True)
    except AuthError as e:
        sys.exit(str(e))
    print(f"Admin {clean_username(username)!r} created (id {uid}, folder {paths.user_root(uid)}).")


def cmd_list():
    with db() as con:
        rows = con.execute("SELECT id, username, is_admin, created FROM users ORDER BY id").fetchall()
    if not rows:
        print("No accounts yet. Create the admin: python -m src.backend.accounts create-admin <username>")
    for r in rows:
        role = "admin" if r["is_admin"] else "user "
        print(f"{r['id']:>4}  {role}  {r['username']:<40} created {r['created']}  {paths.user_root(r['id'])}")


def cmd_reset_password(username: str):
    set_password(username, _ask_password())
    print(f"Password changed for {username!r}; all of its sessions were ended.")


def cmd_delete(username: str):
    row = _find(username)
    folder = paths.user_root(row["id"])
    print(f"This deletes account {row['username']!r} and everything in {folder}.")
    if input("Type the username again to confirm: ").strip().lower() != row["username"]:
        sys.exit("Not confirmed; nothing deleted.")
    with db() as con:
        con.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
        con.execute("DELETE FROM users WHERE id = ?", (row["id"],))
    if folder.exists():
        shutil.rmtree(folder)
    print("Deleted.")


def cmd_adopt_legacy(username: str):
    """One-time move of the pre-accounts shared data into one account:
    data/raw, data/work, data/out -> data/users/<id>/..., and data/uploads/<name>
    (scans for CLI runs) -> raw. Existing targets are never overwritten."""
    row = _find(username)
    root = paths.user_root(row["id"])
    moves = [(paths.DATA / area / d.name, root / area / d.name)
             for area in ("raw", "work", "out") if (paths.DATA / area).is_dir()
             for d in sorted((paths.DATA / area).iterdir()) if d.is_dir()]
    uploads = paths.DATA / "uploads"
    leftovers = []
    if uploads.is_dir():
        for d in sorted(uploads.iterdir()):
            if d.is_dir() and d.name.startswith("temp_"):
                leftovers.append(d)                  # unfinished uploads from before
            elif d.is_dir():
                moves.append((d, root / "raw" / d.name))
    if not moves and not leftovers:
        print("Nothing to adopt: no shared data found.")
        return
    for src, dst in moves:
        if dst.exists():
            print(f"skip   {src}  (already exists: {dst})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"moved  {src}  ->  {dst}")
    for d in leftovers:
        shutil.rmtree(d)
        print(f"removed unfinished upload {d}")
    print(f"Done. {row['username']!r} now owns the old shared studies.")


COMMANDS = {
    "create-admin": cmd_create_admin,
    "list": cmd_list,
    "reset-password": cmd_reset_password,
    "delete": cmd_delete,
    "adopt-legacy": cmd_adopt_legacy,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        sys.exit("Commands:" + __doc__.split("Called by: server.py, and as a CLI:")[1])
    wanted = 0 if args[0] == "list" else 1          # list takes nothing, the rest a username
    if len(args) - 1 != wanted:
        sys.exit(f"Usage: python -m src.backend.accounts {args[0]}" + (" <username>" if wanted else ""))
    COMMANDS[args[0]](*args[1:])
