"""
test_ratelimit.py — the token bucket that caps sign-ups, logins and uploads.

Time is faked (a clock we move by hand), so these run instantly and exactly.

Run:  python -m pytest tests/test_ratelimit.py -v
      python tests/test_ratelimit.py          (no pytest needed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Predict-Studio/

from src.backend.ratelimit import TokenBucket, KeyedBuckets


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_bucket_allows_capacity_then_refuses():
    clock = Clock()
    b = TokenBucket(3, 10, clock)
    assert [b.take() for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = b.take()
    assert 9.9 < wait <= 10.0                 # next ball is ~10 s away


def test_one_ball_comes_back_per_refill_period():
    clock = Clock()
    b = TokenBucket(2, 10, clock)
    b.take(); b.take()
    clock.t += 5
    assert b.take() > 0                       # half a ball is not a ball
    clock.t += 5
    assert b.take() == 0.0                    # one full period: one ball
    assert b.take() > 0


def test_refill_never_exceeds_capacity():
    clock = Clock()
    b = TokenBucket(2, 1, clock)
    clock.t += 1_000_000                      # idle for a long time
    assert [b.take() for _ in range(3)][:2] == [0.0, 0.0]
    assert b.take() > 0


def test_keyed_buckets_are_independent():
    clock = Clock()
    kb = KeyedBuckets(1, 60, clock)
    assert kb.take("alice") == 0.0
    assert kb.take("alice") > 0               # alice is out of balls...
    assert kb.take("bob") == 0.0              # ...bob is not affected


def test_wait_only_looks():
    clock = Clock()
    kb = KeyedBuckets(1, 60, clock)
    assert kb.wait("alice") == 0.0            # unknown key: free to act
    assert kb.wait("alice") == 0.0            # looking twice takes nothing
    kb.take("alice")
    assert kb.wait("alice") > 0               # a failure was spent: now wait
    clock.t += 60
    assert kb.wait("alice") == 0.0


def test_keyed_buckets_cannot_grow_without_limit():
    clock = Clock()
    kb = KeyedBuckets(1, 60, clock)
    kb.MAX_KEYS = 50
    for i in range(500):                      # a flood of made-up usernames
        kb.take(f"fake{i}")
    assert len(kb.buckets) <= 50


def test_idle_keys_are_pruned_first():
    clock = Clock()
    kb = KeyedBuckets(1, 60, clock)
    kb.MAX_KEYS = 3
    kb.take("a"); kb.take("b"); kb.take("c")  # all three empty
    clock.t += 120                            # they refill completely (idle)
    kb.take("d")
    assert set(kb.buckets) == {"d"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} passed")
