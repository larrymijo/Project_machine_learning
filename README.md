# Predicting CONMEBOL World Cup qualifier scores

**DA380A Machine Learning · Högskolan Kristianstad · HT2026 · Group 28**
Members: Lawrence Arciniegas, Gunnar Arias

A regression project in the **Sports** area: predict the **home and away score** of international football matches from pre-match information (team strength, form, head-to-head, venue, match importance), and evaluate it on the **CONMEBOL World Cup qualifiers** — the 10 South American national teams.

| | |
|---|---|
| **Problem type** | Regression (two targets: home goals, away goals) |
| **Algorithm 1** | Linear Regression **implemented from scratch** in NumPy |
| **Algorithm 2** | Random Forest Regressor (scikit-learn) |
| **Training data** | International matches between FIFA national teams, 1994 → Aug 2023 (25,266) |
| **Main test set** | The 90 CONMEBOL qualifiers of the 2026 World Cup cycle (Sep 2023 → Sep 2025) |

The full plan — decisions, features, timeline, work split — is in **[docs/PLAN.md](docs/PLAN.md)**.

## Key dates

| Date | What |
|---|---|
| 24 Sep 2026 | Step 1: group, topic and dataset submitted ✅ |
| **25 Oct 2026** | **Final submission: code + report on Canvas** |
| **27–28 Oct 2026** | **Presentation** (our slot is announced on Canvas) |

## Dataset

Mart Jürisoo, *International football results from 1872* — 49,945 matches with date, teams, score, tournament, city, country and a neutral-venue flag.

- Kaggle (submitted in Step 1): <https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017>
- Same files from the author on GitHub: <https://github.com/martj42/international_results>

The CSVs are **not committed**. Download them into `data/raw/` (PowerShell, from the repo folder):

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
foreach ($f in "results", "goalscorers", "shootouts", "former_names") {
    Invoke-WebRequest "https://raw.githubusercontent.com/martj42/international_results/master/$f.csv" -OutFile "data\raw\$f.csv"
}
```

The dataset is updated after new matches are played; our version was downloaded on **13 Oct 2026** (last match 6 Oct 2026).

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/larrymijo/Project_machine_learning.git
cd Project_machine_learning
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
nbstripout --install                # strips notebook outputs from commits
```

Then download the data (above), open the folder in VS Code, select the `.venv` interpreter and open `notebooks/project.ipynb`. Run the tests with `pytest`.

## Project structure

| Path | Contents |
|---|---|
| `data/raw/` | Downloaded CSVs (git-ignored) |
| `data/processed/` | Feature table built by the notebook (git-ignored) |
| `docs/PLAN.md` | Project plan |
| `notebooks/project.ipynb` | The submission notebook, following the guidelines' 11-step structure |
| `src/` | Data loading/cleaning, feature engineering, evaluation helpers |
| `tests/` | `pytest` checks: from-scratch linear regression and leakage |
| `reports/` | Report and presentation |

## Status

- [x] Phase 0 — Setup, data, plan (13 Oct)
- [x] Phase 1 — Load, clean, explore (15 Oct)
- [x] Phase 2 — Feature engineering + leakage tests (18 Oct)
- [x] Phase 3 — Linear Regression from scratch + tests (18 Oct)
- [x] Phase 4 — Chronological split, baselines, Algorithm 1 results (20 Oct)
- [ ] Phase 5 — Random Forest + tuning
- [ ] Phase 6 — Evaluation, comparison, figures
- [ ] Phase 7 — Report + final notebook
- [ ] Submit on Canvas (25 Oct)
- [ ] Phase 8 — Slides + presentation (27–28 Oct)

## Workflow rules

- Don't commit directly to `main` after setup: one branch per task, pull request, the other member reviews.
- Only one person edits the notebook at a time.
- Every feature uses only matches **before** the match it describes; the test period is never used for training or tuning.
- `random_state=42` everywhere so results are reproducible.

## Use of AI tools

AI tools may be used for ideas, explanations and debugging, but not to copy work directly.
Every group member must be able to explain all code and text submitted.
