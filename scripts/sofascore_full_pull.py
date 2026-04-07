#!/usr/bin/env python3
"""
ball_gpt — full pull: Premier League (or any tournament/season) match list, then per-match
detail + statistics. Respects RapidAPI rate limit (default 2 req/s; use --rate to tune).

Usage:
  python3 scripts/sofascore_full_pull.py --out-dir ./data/pl2526

Optional:
  --max-requests N   Stop after N HTTP calls (quota safety / testing)
  --dry-run          Only fetch paginated match list, no detail/stats

Env:
  RAPIDAPI_KEY       Required (see .env.example)
  SOFASCORE_HOST     default: sofascore.p.rapidapi.com
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


HOST = os.environ.get("SOFASCORE_HOST", "sofascore.p.rapidapi.com")
BASE = f"https://{HOST}"


@dataclass
class Budget:
    """Hard cap on total HTTP requests (optional)."""

    max_requests: int | None
    used: int = 0

    def consume(self, n: int = 1) -> None:
        self.used += n
        if self.max_requests is not None and self.used > self.max_requests:
            raise RuntimeError(
                f"Exceeded --max-requests={self.max_requests} (would be {self.used})."
            )


@dataclass
class RateLimiter:
    """At most `max_per_second` requests per rolling 1-second window."""

    max_per_second: float = 5.0
    timestamps: list[float] = field(default_factory=list)

    def wait(self) -> None:
        now = time.monotonic()
        window = 1.0
        # Drop timestamps outside window
        self.timestamps = [t for t in self.timestamps if now - t < window]
        if len(self.timestamps) >= int(self.max_per_second):
            sleep_for = window - (now - self.timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self.timestamps = [t for t in self.timestamps if now - t < window]
        self.timestamps.append(time.monotonic())


def http_get(
    path: str,
    params: dict[str, Any],
    api_key: str,
    budget: Budget,
    limiter: RateLimiter,
    retries: int = 12,
) -> Any:
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{BASE}{path}?{q}" if q else f"{BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "x-rapidapi-host": HOST,
            "x-rapidapi-key": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        budget.consume(1)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = resp.read()
                if resp.status == 204 or not body:
                    return None
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            err_body = ""
            try:
                raw = e.read() if hasattr(e, "read") else (e.fp.read() if e.fp else b"")
                err_body = raw.decode("utf-8", errors="replace")[:500]
            except OSError:
                pass
            if e.code == 429 and attempt < retries - 1:
                # RapidAPI can throttle harder than our client-side limiter; back off longer.
                wait = min(120.0, 5.0 * (2**attempt))
                print(f"  HTTP 429, sleeping {wait:.0f}s (attempt {attempt + 1}/{retries})…", flush=True)
                time.sleep(wait)
                continue
            if e.code == 429:
                raise RuntimeError("HTTP 429 Too Many Requests — back off or check quota.") from e
            if e.code in (401, 403):
                raise RuntimeError(
                    f"HTTP {e.code} {e.reason}. "
                    f"Subscribe this app to SofaScore on RapidAPI or check x-rapidapi-key. Body: {err_body}"
                ) from e
            raise
    assert last_err is not None
    raise last_err


def fetch_all_last_matches(
    tournament_id: int,
    season_id: int,
    api_key: str,
    budget: Budget,
    limiter: RateLimiter,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    page = 0
    while True:
        data = http_get(
            "/tournaments/get-last-matches",
            {"tournamentId": tournament_id, "seasonId": season_id, "pageIndex": page},
            api_key,
            budget,
            limiter,
        )
        if not data:
            break
        batch = data.get("events") or []
        if not batch:
            break
        events.extend(batch)
        page += 1
    return events


def main() -> int:
    p = argparse.ArgumentParser(description="SofaScore full pull (detail + statistics per match)")
    p.add_argument("--tournament-id", type=int, default=17, help="Premier League default: 17")
    p.add_argument("--season-id", type=int, default=76986, help="25/26 default: 76986")
    p.add_argument("--out-dir", type=str, default="./data/sofascore_pull")
    p.add_argument("--max-requests", type=int, default=None, help="Stop before exceeding N HTTP calls")
    p.add_argument("--dry-run", action="store_true", help="Only list matches, no detail/stats")
    p.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Max requests per second (default: 2; RapidAPI may still 429 at 5)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip matches that already have a non-empty matches/<id>.json",
    )
    p.add_argument(
        "--reuse-events-list",
        action="store_true",
        help="If events_list.json exists, skip re-fetching the paginated match list (saves HTTP calls)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="RapidAPI key (default: RAPIDAPI_KEY environment variable)",
    )
    args = p.parse_args()

    api_key = (args.api_key or os.environ.get("RAPIDAPI_KEY", "")).strip()
    if not api_key:
        print(
            "Missing API key: export RAPIDAPI_KEY=... or pass --api-key (see .env.example).",
            file=sys.stderr,
        )
        return 1

    budget = Budget(max_requests=args.max_requests)
    limiter = RateLimiter(max_per_second=args.rate)

    out_dir = os.path.abspath(args.out_dir)
    matches_dir = os.path.join(out_dir, "matches")
    os.makedirs(matches_dir, mode=0o755, exist_ok=True)

    events_path = os.path.join(out_dir, "events_list.json")
    if args.reuse_events_list and os.path.isfile(events_path):
        print("Loading match list from events_list.json …", flush=True)
        with open(events_path, encoding="utf-8") as f:
            events = json.load(f).get("events") or []
    else:
        print("Fetching paginated match list…", flush=True)
        events = fetch_all_last_matches(
            args.tournament_id, args.season_id, api_key, budget, limiter
        )
    ids = []
    for ev in events:
        mid = ev.get("id")
        if mid is not None:
            ids.append(int(mid))
    ids = sorted(set(ids))

    manifest = {
        "tournamentId": args.tournament_id,
        "seasonId": args.season_id,
        "matchCount": len(ids),
        "matchIds": ids,
        "httpUsedAfterList": budget.used,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "events_list.json"), "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=2)

    print(f"Listed {len(events)} event rows, {len(ids)} unique match ids. HTTP calls so far: {budget.used}")

    if args.dry_run:
        print("Dry run: skipping matches/detail and matches/get-statistics.")
        return 0

    def match_file_ok(path: str) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            if os.path.getsize(path) < 64:
                return False
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, dict) and "detail" in data and "statistics" in data
        except (OSError, json.JSONDecodeError):
            return False

    if args.resume:
        before = len(ids)
        ids = [mid for mid in ids if not match_file_ok(os.path.join(matches_dir, f"{mid}.json"))]
        skipped = before - len(ids)
        if skipped:
            print(f"Resume: skipping {skipped} matches already on disk, {len(ids)} remaining.", flush=True)

    for i, mid in enumerate(ids, start=1):
        detail = http_get(
            "/matches/detail",
            {"matchId": mid},
            api_key,
            budget,
            limiter,
        )
        stats = http_get(
            "/matches/get-statistics",
            {"matchId": mid},
            api_key,
            budget,
            limiter,
        )
        payload = {
            "matchId": mid,
            "detail": detail,
            "statistics": stats,
        }
        path = os.path.join(matches_dir, f"{mid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if i % 10 == 0 or i == len(ids):
            print(f"  [{i}/{len(ids)}] matchId={mid} … HTTP total: {budget.used}", flush=True)

    manifest["httpUsedTotal"] = budget.used
    manifest["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Output: {out_dir}  Total HTTP: {budget.used}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
