"""
test_accounts.py — accounts, passwords, sessions and access tokens.

Each test uses a fresh temporary data folder, so the real data/accounts.db is
never touched. The failed-login delay is set to 0 to keep the suite fast.

Run:  python -m pytest tests/test_accounts.py -v
      python tests/test_accounts.py          (no pytest needed)
"""
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Predict-Studio/

from src.backend import accounts, paths

TOKEN = "grant-alpha"
_REAL_DATA, _REAL_DELAY = paths.DATA, accounts.FAILED_LOGIN_DELAY


def teardown_module():
    """Leave paths/accounts as other test files expect them."""
    paths.DATA, accounts.FAILED_LOGIN_DELAY = _REAL_DATA, _REAL_DELAY
    os.environ.pop(accounts.TOKENS_ENV, None)


def fresh(tokens=TOKEN):
    """A new empty data folder and the given access tokens."""
    paths.DATA = Path(tempfile.mkdtemp())
    os.environ[accounts.TOKENS_ENV] = tokens
    accounts.FAILED_LOGIN_DELAY = 0


def refused(fn, *args):
    try:
        fn(*args)
    except accounts.AuthError:
        return True
    return False


def test_signup_needs_a_granted_token():
    fresh()
    assert refused(accounts.create_user, "alice", "password1", "wrong-token")
    assert refused(accounts.create_user, "alice", "password1", "")
    assert accounts.create_user("alice", "password1", TOKEN) > 0


def test_signup_closed_when_no_tokens_configured():
    fresh(tokens="")
    assert refused(accounts.create_user, "alice", "password1", "anything")


def test_usernames_and_passwords_are_validated():
    fresh()
    for bad in ["ab", "has space", "../x", "a/b", "x" * 65, ""]:
        assert refused(accounts.create_user, bad, "password1", TOKEN), bad
    assert refused(accounts.create_user, "carol", "short", TOKEN)
    accounts.create_user("Mail.Someone@Example.com", "password1", TOKEN)   # emails, any case
    assert refused(accounts.create_user, "mail.someone@example.com", "password1", TOKEN)  # taken


def test_password_is_never_stored():
    fresh()
    accounts.create_user("alice", "password1", TOKEN)
    raw = accounts.db_path().read_bytes()
    assert b"password1" not in raw and TOKEN.encode() not in raw


def test_login_needs_password_and_token_for_users():
    fresh()
    accounts.create_user("alice", "password1", TOKEN)
    assert refused(accounts.login, "alice", "wrong-pass", TOKEN)
    assert refused(accounts.login, "alice", "password1", "")
    assert refused(accounts.login, "nobody", "password1", TOKEN)
    token, user = accounts.login("ALICE", "password1", TOKEN)
    assert user["username"] == "alice" and not user["is_admin"]
    assert accounts.user_for_session(token)["username"] == "alice"


def test_admin_needs_no_token():
    fresh(tokens="")                         # even with every token revoked
    accounts.create_user("boss@example.com", "adminpass1", is_admin=True)
    token, user = accounts.login("boss@example.com", "adminpass1")
    assert user["is_admin"] and accounts.user_for_session(token)["is_admin"]


def test_revoking_a_token_ends_its_sessions():
    fresh(tokens=f"{TOKEN},grant-beta")
    accounts.create_user("alice", "password1", TOKEN)
    accounts.create_user("bob", "password1", "grant-beta")
    a, _ = accounts.login("alice", "password1", TOKEN)
    b, _ = accounts.login("bob", "password1", "grant-beta")
    os.environ[accounts.TOKENS_ENV] = "grant-beta"          # alice's token removed
    assert accounts.user_for_session(a) is None
    assert accounts.user_for_session(b)["username"] == "bob"
    assert refused(accounts.login, "alice", "password1", TOKEN)


def test_logout_and_expiry_end_a_session():
    fresh()
    accounts.create_user("alice", "password1", TOKEN)
    t1, _ = accounts.login("alice", "password1", TOKEN)
    accounts.logout(t1)
    assert accounts.user_for_session(t1) is None
    t2, _ = accounts.login("alice", "password1", TOKEN)
    past = (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds")
    con = sqlite3.connect(accounts.db_path())
    con.execute("UPDATE sessions SET expires = ?", (past,))
    con.commit(); con.close()
    assert accounts.user_for_session(t2) is None
    assert accounts.user_for_session("not-a-session") is None
    assert accounts.user_for_session(None) is None


def test_session_token_is_stored_only_as_a_hash():
    fresh()
    accounts.create_user("alice", "password1", TOKEN)
    token, _ = accounts.login("alice", "password1", TOKEN)
    assert token.encode() not in accounts.db_path().read_bytes()


def test_one_token_can_create_only_so_many_accounts():
    fresh()
    accounts.MAX_ACCOUNTS_PER_TOKEN, saved = 3, accounts.MAX_ACCOUNTS_PER_TOKEN
    try:
        for i in range(3):
            accounts.create_user(f"user{i}", "password1", TOKEN)
        assert refused(accounts.create_user, "user3", "password1", TOKEN)
    finally:
        accounts.MAX_ACCOUNTS_PER_TOKEN = saved


def test_ids_are_never_reused():
    fresh()
    first = accounts.create_user("alice", "password1", TOKEN)
    with accounts.db() as con:
        con.execute("DELETE FROM users WHERE id = ?", (first,))
    assert accounts.create_user("bob", "password1", TOKEN) != first   # bob never gets alice's folder


def test_password_reset_ends_existing_sessions():
    fresh()
    accounts.create_user("alice", "password1", TOKEN)
    token, _ = accounts.login("alice", "password1", TOKEN)
    accounts.set_password("alice", "password2")
    assert accounts.user_for_session(token) is None
    assert refused(accounts.login, "alice", "password1", TOKEN)
    assert accounts.login("alice", "password2", TOKEN)[0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} passed")
