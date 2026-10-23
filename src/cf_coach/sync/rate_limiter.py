"""A token-bucket rate limiter stored in Redis, shared by every process.

The idea:
  - A bucket holds up to `capacity` tokens and refills at `rate` tokens per second.
  - Every request must take 1 token first. Empty bucket -> wait for the next token.

Why Redis: the old limiter kept its state in a module variable, so every Python process
had its own copy, and two processes together could break Codeforces' limit.
With the state in Redis, every process shares one bucket.

Why a Lua script: "read the bucket, refill it, take a token, save it" is 4 steps. If two
processes run those steps at the same time, both can read "1 token left" and both take
it. Redis runs a Lua script as ONE step that nothing else can interrupt, so that can't
happen. Design reasons: docs/decisions.md #16.
"""

import time

import redis

# Runs inside Redis. KEYS[1] = bucket key, ARGV[1] = capacity, ARGV[2] = rate (tokens/s).
# Returns 0 if a token was taken, otherwise how many milliseconds to wait before retrying.
TOKEN_BUCKET_LUA = """
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])

-- Use Redis's own clock, so every process (even on different machines) agrees on "now".
local t   = redis.call('TIME')                      -- {seconds, microseconds}
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local saved  = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(saved[1]) or capacity       -- a brand-new bucket starts full
local ts     = tonumber(saved[2]) or now

-- Refill: add the tokens that dripped in since the last visit, capped at capacity.
tokens = math.min(capacity, tokens + (now - ts) * rate)

local wait_ms = 0
if tokens >= 1 then
  tokens = tokens - 1                                -- take a token: request allowed
else
  wait_ms = math.ceil((1 - tokens) / rate * 1000)    -- time until one full token
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 60)   -- an idle bucket disappears (it would be full again anyway)
return wait_ms
"""


class TokenBucket:
    def __init__(self, client: redis.Redis, key: str, capacity: int, rate: float):
        self.key = key
        self.capacity = capacity
        self.rate = rate
        # register_script sends the Lua code to Redis once; later calls reuse it by its hash.
        self._script = client.register_script(TOKEN_BUCKET_LUA)

    def acquire(self) -> None:
        """Block until a token is available, then take it."""
        while True:
            wait_ms = self._script(keys=[self.key], args=[self.capacity, self.rate])
            if wait_ms == 0:
                return
            time.sleep(wait_ms / 1000)


# ---------- Manual check: run this in two terminals at once ----------
#   uv run python -m cf_coach.sync.rate_limiter
# Each process takes 3 tokens. Together, all the "took a token" times are about 2 s apart.

if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    bucket = TokenBucket(
        redis.Redis.from_url(os.environ["REDIS_URL"]),
        key="ratelimit:demo", capacity=1, rate=0.5,
    )
    for i in range(3):
        bucket.acquire()
        print(f"pid {os.getpid()} took token {i + 1} at {time.time() % 100:.2f}", flush=True)
