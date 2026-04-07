# ball_gpt

Project name: **ball_gpt**. The trainable model and data utilities live in the **`ball_gpt/`** Python package (decoder GPT, tokenizer, datasets).

Small character-level GPT trained on Premier League match lines (scores + stats). Includes SofaScore API helpers and a prediction script that **generates text**, not real forecasts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your RapidAPI key (required for any script that calls SofaScore):

```bash
cp .env.example .env
# edit .env, then:
export RAPIDAPI_KEY="your_key"
```

Never commit `.env` or paste keys into source files.

## Repo layout

| Path | Purpose |
|------|---------|
| `ball_gpt/` | Core library: `model`, `tokenizer`, `dataset` |
| `data/corpus/pl2526.txt` | Training text (rebuild with `scripts/build_corpus.py`) |
| `data/pl2526_full/matches/` | Raw match JSON from API (large; gitignored) |
| `data/prepared/vocab.json` | Character tokenizer |
| `data/checkpoints/` | Training checkpoints (`*.pt` gitignored) |
| `data/predictions/` | Generated prediction logs |

## Commands

```bash
export PYTHONPATH=.

# Pull match data (needs RAPIDAPI_KEY)
python scripts/sofascore_full_pull.py --out-dir ./data/pl2526_full

# Build corpus from JSON
python scripts/build_corpus.py

# Dataset smoke test + save vocab
python scripts/prepare_dataset.py

# Train
python scripts/train.py --amp

# Predict (API + model; needs RAPIDAPI_KEY)
python scripts/predict_upcoming.py
```

## Security

- `.env` is listed in `.gitignore`. Use `.env.example` as a template only.
- Large raw JSON pulls and model weights are excluded from git; regenerate locally or use your own artifacts.
