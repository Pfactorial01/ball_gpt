#!/usr/bin/env python3
"""
ball_gpt — build a text corpus from SofaScore match JSON (detail + statistics) for LM training.

Phase 1: Premier League 25/26 only — reads per-match files from data/pl2526_full/matches/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Stats to embed, in fixed order. (statistics key, short label in corpus)
# Values are always home-away using homeValue/awayValue (normalized).
STAT_ORDER: list[tuple[str, str]] = [
    ("ballPossession", "Poss"),
    ("expectedGoals", "xG"),
    ("bigChanceCreated", "BigChances"),
    ("totalShotsOnGoal", "Shots"),
    ("shotsOnGoal", "SOT"),
    ("shotsOffGoal", "OffTarget"),
    ("blockedScoringAttempt", "Blocked"),
    ("cornerKicks", "Corners"),
    ("fouls", "Fouls"),
    ("yellowCards", "Yellows"),
    ("redCards", "Reds"),
    ("goalkeeperSaves", "Saves"),
    ("passes", "Passes"),
    ("totalTackle", "Tackles"),
    ("freeKicks", "FreeKicks"),
]

END_TOKEN = "<|end|>"
SEASON_TAG = "<25/26>"


def _flatten_all_period(stats_root: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """First occurrence wins (overview comes before duplicate group names)."""
    out: dict[str, dict[str, Any]] = {}
    if not stats_root:
        return out
    for period in stats_root.get("statistics") or []:
        if period.get("period") != "ALL":
            continue
        for grp in period.get("groups") or []:
            for item in grp.get("statisticsItems") or []:
                k = item.get("key")
                if k and k not in out:
                    out[k] = item
    return out


def _fmt_pair(hv: Any, av: Any, *, as_float: bool) -> str:
    if as_float:
        return f"{float(hv):.2f}-{float(av):.2f}"
    return f"{int(round(float(hv)))}-{int(round(float(av)))}"


def _stat_segment(key: str, label: str, m: dict[str, dict[str, Any]]) -> str | None:
    item = m.get(key)
    if not item:
        return None
    hv, av = item.get("homeValue"), item.get("awayValue")
    if hv is None or av is None:
        return None
    as_float = key == "expectedGoals"
    return f"{label}: {_fmt_pair(hv, av, as_float=as_float)}"


def match_to_line(payload: dict[str, Any]) -> str | None:
    detail = payload.get("detail") or {}
    event = detail.get("event")
    if not event:
        return None
    st = event.get("status") or {}
    if st.get("type") != "finished":
        return None

    home = (event.get("homeTeam") or {}).get("name")
    away = (event.get("awayTeam") or {}).get("name")
    hs = event.get("homeScore") or {}
    aws = event.get("awayScore") or {}
    hc, ac = hs.get("current"), aws.get("current")
    if not home or not away or hc is None or ac is None:
        return None

    stats_block = payload.get("statistics") or {}
    stat_map = _flatten_all_period(stats_block)
    if not stat_map:
        return None

    parts = [
        SEASON_TAG,
        f"Home: {home}",
        f"Away: {away}",
        f"Score: {int(hc)}-{int(ac)}",
    ]
    for key, label in STAT_ORDER:
        seg = _stat_segment(key, label, stat_map)
        if seg:
            parts.append(seg)

    parts.append(END_TOKEN)
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build PL 25/26 text corpus from SofaScore JSON matches")
    ap.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "pl2526_full", "matches"),
        help="Directory containing <matchId>.json files",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "pl2526.txt"),
        help="Output text file (one match per line)",
    )
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_path = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    lines_out: list[str] = []
    skipped = 0
    for name in names:
        path = os.path.join(input_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        line = match_to_line(payload)
        if line is None:
            skipped += 1
            continue
        lines_out.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")

    print(f"Wrote {len(lines_out)} lines to {output_path}")
    print(f"Skipped {skipped} files (unfinished, missing stats, or parse errors).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
