from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

# Runs the data build on a RunPod CPU pod, which gets datacenter bandwidth to database.lichess.org. The pod
# serves /w over HTTP the whole time, so its logs can be read mid-run and out.tgz fetched at the end.
# The caller deletes it.
API = "https://rest.runpod.io/v1"
UA = {"User-Agent": "levelrl/1"}
SCRIPT = r"""
set -e
mkdir -p /w/levelrl && cd /w
(python -m http.server 8000 > /dev/null 2>&1 &)
apt-get update -qq && apt-get install -y -qq zstd curl > apt.log 2>&1
pip install -q python-chess > pip.log 2>&1
echo "$DATA_PY" | base64 -d > levelrl/data.py && touch levelrl/__init__.py
for m in $MONTHS; do python -m levelrl.data --scan $m > scan_$m.log 2>&1 & done
wait
python -m levelrl.data --months $MONTHS > build.log 2>&1
tar czf out.tgz data/positions data/scan/*/done *.log
touch FINISHED
sleep infinity
"""


def call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                          "Content-Type": "application/json", **UA})
    raw = urllib.request.urlopen(req, timeout=60).read()
    return json.loads(raw) if raw else None


def get(pod: str, name: str) -> bytes | None:
    try:
        return urllib.request.urlopen(urllib.request.Request(f"https://{pod}-8000.proxy.runpod.net/{name}", headers=UA),
                                      timeout=60).read()
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "watch", "fetch", "delete"])
    ap.add_argument("--pod")
    ap.add_argument("--months", default="2025-09 2026-03 2026-09")
    ap.add_argument("--vcpu", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "create":
        src = Path(__file__).resolve().parent.parent / "levelrl" / "data.py"
        pod = call("POST", "/pods", {
            "name": "levelrl-data", "computeType": "CPU", "cpuFlavorIds": ["cpu3g"], "vcpuCount": a.vcpu,
            "imageName": "python:3.12-slim", "containerDiskInGb": 40, "ports": ["8000/http"],
            "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [SCRIPT],
            "env": {"DATA_PY": base64.b64encode(src.read_bytes()).decode(), "MONTHS": a.months},
        })
        print(json.dumps({k: pod.get(k) for k in ("id", "costPerHr", "machineId")}))
    elif a.cmd == "watch":
        for name in ("FINISHED", *[f"scan_{m}.log" for m in a.months.split()], "build.log"):
            body = get(a.pod, name)
            print(f"== {name}: " + ("(none)" if body is None else body.decode(errors="replace").strip()[-400:]))
    elif a.cmd == "fetch":
        Path("out.tgz").write_bytes(get(a.pod, "out.tgz"))
        print("out.tgz", Path("out.tgz").stat().st_size)
    elif a.cmd == "delete":
        call("DELETE", f"/pods/{a.pod}")
        time.sleep(3)
        print([p["id"] for p in call("GET", "/pods")])
