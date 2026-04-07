#!/usr/bin/env python3
"""
ball_gpt — load checkpoint + vocab, fetch upcoming PL fixtures from SofaScore API,
and generate a continuation (not a real forecast — pattern completion on training text).

  source .venv/bin/activate
  PYTHONPATH=. python scripts/predict_upcoming.py --limit 5
  # Output is saved to data/predictions/latest.txt by default (-o to change).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
import urllib.parse
import urllib.request

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from ball_gpt.model import GPT, GPTConfig
from ball_gpt.tokenizer import CharTokenizer

HOST = os.environ.get("SOFASCORE_HOST", "sofascore.p.rapidapi.com")
BASE = f"https://{HOST}"


def http_get_json(path: str, params: dict, api_key: str) -> dict:
    q = urllib.parse.urlencode(params)
    url = f"{BASE}{path}?{q}"
    req = urllib.request.Request(
        url,
        headers={
            "x-rapidapi-host": HOST,
            "x-rapidapi-key": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        if resp.status == 204 or not body:
            return {}
        return json.loads(body.decode("utf-8"))


def fetch_upcoming_fixtures(
    api_key: str,
    tournament_id: int = 17,
    season_id: int = 76986,
    max_matches: int = 10,
) -> list[tuple[str, str]]:
    """Return list of (home_name, away_name) from paginated get-next-matches."""
    out: list[tuple[str, str]] = []
    page = 0
    while len(out) < max_matches:
        time.sleep(0.3)
        data = http_get_json(
            "/tournaments/get-next-matches",
            {"tournamentId": tournament_id, "seasonId": season_id, "pageIndex": page},
            api_key,
        )
        batch = data.get("events") or []
        if not batch:
            break
        for ev in batch:
            h = (ev.get("homeTeam") or {}).get("name")
            a = (ev.get("awayTeam") or {}).get("name")
            if h and a:
                out.append((h, a))
            if len(out) >= max_matches:
                break
        page += 1
    return out[:max_matches]


def build_prompt(home: str, away: str) -> str:
    """Must match training format in scripts/build_corpus.py."""
    return f"<25/26> | Home: {home} | Away: {away} | Score: "


@torch.no_grad()
def generate_completion(
    model: GPT,
    tok: CharTokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 900,
    temperature: float = 0.75,
) -> str:
    model.eval()
    ids = tok.encode(prompt)
    missing = set(prompt) - set(tok.stoi.keys())
    if missing:
        raise ValueError(f"Prompt contains characters not in vocab: {missing!r}")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    end_marker = "<|end|>"

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        text = tok.decode(idx[0].tolist())
        if end_marker in text:
            break

    return tok.decode(idx[0].tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict (generate) stats lines for upcoming fixtures")
    ap.add_argument("--checkpoint", default=os.path.join(_ROOT, "data", "checkpoints", "ckpt.pt"))
    ap.add_argument("--vocab", default=os.path.join(_ROOT, "data", "prepared", "vocab.json"))
    ap.add_argument("--limit", type=int, default=8, help="How many upcoming fixtures to show")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--api-key",
        default=None,
        help="RapidAPI key (default: RAPIDAPI_KEY env; see .env.example)",
    )
    ap.add_argument(
        "--output",
        "-o",
        default=os.path.join(_ROOT, "data", "predictions", "latest.txt"),
        help="Write full run log + generated lines to this file (default: data/predictions/latest.txt)",
    )
    ap.add_argument(
        "--also-timestamped",
        action="store_true",
        help="Also write data/predictions/predictions_YYYYMMDD_HHMMSS.txt",
    )
    args = ap.parse_args()

    api_key = (args.api_key or os.environ.get("RAPIDAPI_KEY", "")).strip()
    if not api_key:
        print(
            "Missing API key: export RAPIDAPI_KEY=... or pass --api-key (see .env.example).",
            file=sys.stderr,
        )
        return 1

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    out_path = os.path.abspath(args.output)

    print("Fetching upcoming Premier League fixtures from API…", flush=True)
    fixtures = fetch_upcoming_fixtures(api_key, max_matches=args.limit)
    if not fixtures:
        print("No upcoming fixtures returned (check subscription / season ids).", file=sys.stderr)
        return 1
    print(f"Got {len(fixtures)} fixture(s).\n", flush=True)

    tok = CharTokenizer.load(args.vocab)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"# ball_gpt model predictions\n"
        f"# generated_utc: {ts}\n"
        f"# checkpoint: {os.path.abspath(args.checkpoint)}\n"
        f"# vocab: {os.path.abspath(args.vocab)}\n"
        f"# temperature: {args.temperature}\n"
        f"# fixtures_requested: {args.limit}\n"
        f"#\n\n"
    )

    paths_to_write = [out_path]
    if args.also_timestamped:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        paths_to_write.append(
            os.path.join(_ROOT, "data", "predictions", f"predictions_{stamp}.txt")
        )

    body_parts: list[str] = [header]

    for i, (home, away) in enumerate(fixtures, 1):
        prompt = build_prompt(home, away)
        block = f"--- Match {i}: {home} vs {away} ---\nPrompt: {prompt!r}\n"
        print(block, end="")
        body_parts.append(block)
        try:
            text = generate_completion(
                model, tok, prompt, device, temperature=args.temperature
            )
        except ValueError as e:
            err = f"  Skip: {e}\n"
            print(err, end="")
            body_parts.append(err)
            continue
        gen = f"Generated:\n{text}\n\n"
        print(gen, end="")
        body_parts.append(gen)

    footer = (
        "Note: this is next-token continuation on the training distribution, not a real match engine.\n"
    )
    print(footer, end="")
    body_parts.append(footer)

    full_text = "".join(body_parts)
    for path in paths_to_write:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_text)
    print(f"\nSaved output to: {out_path}", flush=True)
    if args.also_timestamped and len(paths_to_write) > 1:
        print(f"Also saved: {paths_to_write[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
