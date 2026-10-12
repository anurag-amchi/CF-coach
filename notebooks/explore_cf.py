import httpx
import json

HANDLE = "TitanCoder1258"

resp = httpx.get(
    "https://codeforces.com/api/user.status",
    params={"handle": HANDLE, "from": 1, "count": 5},
    timeout=30,
)

resp.raise_for_status() # checks for errors
data = resp.json()

print(data["status"])
print(len(data["result"]))
print(json.dumps(data["result"], indent=2))