# ball_gpt

A small **decoder-only GPT** (PyTorch) trained on **character-level** text: one line per Premier League match with score and team stats (possession, xG, shots, cards, etc.). The project includes scripts to pull data from the **SofaScore API** (via RapidAPI), build a corpus, train the model, and **generate** synthetic match lines for upcoming fixtures.

**What this is:** a learning project—next-token language modeling on a fixed text format.  
**What this is not:** a sportsbook-grade predictor. Generations are **stylized continuations** from training data, not causal forecasts of real matches.

---

## Requirements

- **Python** 3.10+ (3.12 recommended)
- **PyTorch** with CUDA optional (`pip install -r requirements.txt`)
- **RapidAPI** subscription to the [SofaScore API](https://rapidapi.com/apidojo/api/sofascore) for any script that calls the network

---

## Quick start

```bash
cd ball_gpt

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export PYTHONPATH=.
```

Configure your API key (never commit it):

```bash
cp .env.example .env
# Edit .env, then either:
export RAPIDAPI_KEY="your_rapidapi_key"
# or pass --api-key on CLI where supported
```

---

## End-to-end workflow

| Step | Script | Output |
|------|--------|--------|
| 1. Pull matches | [`scripts/sofascore_full_pull.py`](scripts/sofascore_full_pull.py) | `data/<out>/matches/*.json`, `events_list.json` |
| 2. Build corpus | [`scripts/build_corpus.py`](scripts/build_corpus.py) | [`data/corpus/pl2526.txt`](data/corpus/pl2526.txt) |
| 3. Prepare data | [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) | [`data/prepared/vocab.json`](data/prepared/vocab.json) |
| 4. Train | [`scripts/train.py`](scripts/train.py) | `data/checkpoints/ckpt.pt`, `config.json` |
| 5. Generate | [`scripts/predict_upcoming.py`](scripts/predict_upcoming.py) | [`data/predictions/latest.txt`](data/predictions/latest.txt) (default) |

Example (Premier League 25/26 defaults: `tournamentId=17`, `seasonId=76986`):

```bash
export PYTHONPATH=.

# 1) Fetch JSON (rate-limited; use --resume / --reuse-events-list for long runs)
python scripts/sofascore_full_pull.py --out-dir ./data/pl2526_full

# 2) One line per match for training
python scripts/build_corpus.py

# 3) Char tokenizer + train/val chunks
python scripts/prepare_dataset.py

# 4) Train (~6M params, 6 GB GPU friendly with --amp)
python scripts/train.py --amp

# 5) Upcoming fixtures from API + model completions
python scripts/predict_upcoming.py --limit 8
```

Useful flags:

- **`sofascore_full_pull.py`:** `--resume`, `--reuse-events-list`, `--rate`, `--max-requests`
- **`train.py`:** `--max-steps`, `--batch-size`, `--checkpoint-dir`, `--resume`
- **`predict_upcoming.py`:** `-o` / `--output`, `--temperature`, `--also-timestamped`

---

## Python package

The [`ball_gpt/`](ball_gpt/) package holds the core code:

| Module | Role |
|--------|------|
| [`tokenizer.py`](ball_gpt/tokenizer.py) | Character-level encode/decode, vocab JSON |
| [`dataset.py`](ball_gpt/dataset.py) | `CharLMDataset`, train/val split from corpus |
| [`model.py`](ball_gpt/model.py) | GPT (`GPTConfig`, causal transformer, `generate`) |

Run scripts from the repo root with `PYTHONPATH=.` so `import ball_gpt` resolves.

---

## Data layout

| Path | Purpose |
|------|---------|
| `ball_gpt/` | Library source |
| `scripts/` | CLI entrypoints |
| `data/corpus/` | Training text (`pl2526.txt`) |
| `data/pl2526_full/` | Raw API pulls (large; typically gitignored) |
| `data/prepared/` | `vocab.json` |
| `data/checkpoints/` | Training checkpoints (`*.pt` gitignored) |
| `data/predictions/` | Generated runs (`latest.txt` by default) |

See [`.gitignore`](.gitignore) for what stays out of version control.

---

## Security

- Store secrets in **`.env`** (ignored by git) or the environment only.
- Use [`.env.example`](.env.example) as a template without real keys.
- Rotate any key that was ever committed or pasted in chat.
