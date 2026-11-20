from __future__ import annotations

import argparse
import time
from pathlib import Path

import libtorrent as lt


def main():
    ap = argparse.ArgumentParser(description="Download a torrent with libtorrent and resume existing files")
    ap.add_argument("torrent", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--poll", type=float, default=3.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    session = lt.session({"enable_dht": True, "enable_lsd": True, "enable_natpmp": True, "enable_upnp": True})
    handle = session.add_torrent({"ti": lt.torrent_info(str(args.torrent)), "save_path": str(args.out)})
    started = time.monotonic()
    while True:
        status = handle.status()
        done = status.total_done
        rate = status.download_rate
        print(f"{100 * status.progress:.2f}% · {done / 1e9:.2f} GB · {rate / 1e6:.2f} MB/s", flush=True)
        if status.is_finished:
            break
        if time.monotonic() - started > 7 * 24 * 3600:
            raise TimeoutError("torrent download exceeded one week")
        time.sleep(args.poll)
    session.pause()
    print("download complete", flush=True)


if __name__ == "__main__":
    main()
