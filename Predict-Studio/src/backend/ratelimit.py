"""
ratelimit.py — the token bucket ("a box of balls") that caps abuse.

A bucket holds up to `capacity` balls. Every action takes one ball and one
ball comes back every `refill_seconds`. An empty bucket means "too many, wait".
Refill is worked out from timestamps whenever the bucket is used, so there is
no background thread and nothing to schedule.

    signups = TokenBucket(*SIGNUPS)
    wait = signups.take()        # 0.0 = allowed, else seconds until a ball is back

Does NOT: know about HTTP, users or files. server.py decides what to limit;
the numbers are here so every limit can be read in one place.
Called by: server.py.
"""
import threading
import time

# ── limits: (capacity, seconds until one ball comes back) ──────────────────
SIGNUPS          = (20, 180)   # all sign-ups together: a burst of 20, then 1 per 3 min
LOGINS_PER_USER  = (5, 60)     # WRONG passwords on one username (a good login costs nothing)
LOGINS_GLOBAL    = (60, 1)     # guessing spread across many usernames
UPLOADS_PER_USER = (10, 120)   # repeated large uploads by one account


class TokenBucket:
    def __init__(self, capacity: int, refill_seconds: float, clock=time.monotonic):
        self.capacity = capacity
        self.refill_seconds = refill_seconds
        self.clock = clock               # injectable so tests need not sleep
        self.balls = float(capacity)
        self.stamp = clock()
        self.lock = threading.Lock()

    def _refill(self):
        now = self.clock()
        self.balls = min(self.capacity, self.balls + (now - self.stamp) / self.refill_seconds)
        self.stamp = now

    def take(self) -> float:
        """Take one ball. Returns 0.0 if allowed, else the seconds until one is back."""
        with self.lock:
            self._refill()
            if self.balls >= 1:
                self.balls -= 1
                return 0.0
            return (1 - self.balls) * self.refill_seconds

    def wait(self) -> float:
        """Like take() but only looks: 0.0 if a ball is there, else seconds to wait."""
        with self.lock:
            self._refill()
            return 0.0 if self.balls >= 1 else (1 - self.balls) * self.refill_seconds

    def is_full(self) -> bool:
        with self.lock:
            self._refill()
            return self.balls >= self.capacity


class KeyedBuckets:
    """One bucket per key (a username, a user id).

    Made-up keys must not grow memory without limit: when the table is full,
    buckets that have refilled completely (idle keys) are dropped, and if that
    is not enough the table is cleared. Global buckets still apply meanwhile.
    """
    MAX_KEYS = 10_000

    def __init__(self, capacity: int, refill_seconds: float, clock=time.monotonic):
        self.args = (capacity, refill_seconds, clock)
        self.buckets = {}
        self.lock = threading.Lock()

    def take(self, key) -> float:
        with self.lock:
            if key not in self.buckets and len(self.buckets) >= self.MAX_KEYS:
                self.buckets = {k: b for k, b in self.buckets.items() if not b.is_full()}
                if len(self.buckets) >= self.MAX_KEYS:
                    self.buckets.clear()
            bucket = self.buckets.setdefault(key, TokenBucket(*self.args))
        return bucket.take()

    def wait(self, key) -> float:
        """Seconds until `key` may act again (0.0 = now), without taking a ball."""
        with self.lock:
            bucket = self.buckets.get(key)
        return bucket.wait() if bucket else 0.0
