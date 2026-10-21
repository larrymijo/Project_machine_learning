# Project plan — Predicting CONMEBOL World Cup qualifier scores

**DA380A Machine Learning · HT2026 · Group 28** — Lawrence Arciniegas, Gunnar Arias
Plan written 13 Oct 2026. Final submission **Sun 25 Oct**, presentations **27–28 Oct**.

---

## 1. What we are building

A **regression** model that predicts the **score of a football match** — the number of goals for the home team and for the away team — and evaluate it on the **CONMEBOL World Cup qualifiers** (the 10 South American national teams). It is trained on international matches since 1994 using only information that is known **before** kick-off: team strength (Elo ratings), recent form, head-to-head history, venue and the type of match.

Two algorithms, as the project guidelines require:

| | Algorithm | Implementation |
|---|---|---|
| **Algorithm 1** | **Linear Regression** | **From scratch** in NumPy — normal equation + gradient descent. No library regression model. |
| **Algorithm 2** | **Random Forest Regressor** | scikit-learn (allowed for algorithm 2) |

Example of the final output: *"Bolivia vs Brazil, El Alto, 9 Sep 2025 → predicted x.x – x.x"* next to the actual result, **1 – 0**.

## 2. Requirements → how we meet them

From the *Machine Learning Course Project Guidelines* and the Step 1 submission (see the Obsidian note *Project — Machine Learning*).

| Requirement | Our answer |
|---|---|
| Classification **or** regression | **Regression** — goals scored (submitted in Step 1) |
| Algorithm 1 from scratch, no ready-made model | `LinearRegressionScratch` — our own class, NumPy only (§7) |
| Algorithm 2, library allowed | `RandomForestRegressor` (§8) |
| Evaluate and **compare** both, explain which is better and why | MAE / MSE / RMSE / R² + baselines + error analysis (§9) |
| Readable, organised, executable code in the recommended 11-step order | One submission notebook following the 11 steps (§10) |
| Report: Introduction → Aim → Dataset → Preprocessing → Methods → Results → Comparison & Discussion → Conclusion | §12 |
| Presentation: topic, dataset, preprocessing, methods, results, comparison, **code** | §12 — both of us present and explain the code |
| AI use: allowed for support, no direct copying, must be able to explain everything | Every function reviewed and explained by both members (§11) |
| Deadline | Code + report on Canvas **25 Oct**; present **27–28 Oct** |

## 3. Timeline

Today is **Tue 13 Oct** — 12 days to submission.

| Dates | Phase | Output |
|---|---|---|
| **Tue 13 Oct** | **0 — Setup + plan** | Repo structure, `.venv`, data in `data/raw/`, this plan. Agree on the decisions in §5 |
| Wed 14 – Thu 15 | **1 — Load, clean, explore** | Notebook sections 1–3: data overview, team-name cleaning, CONMEBOL focus, score distributions, home advantage, altitude |
| Thu 15 – Sat 17 | **2 — Feature engineering** | `src/features.py`: Elo, form, head-to-head, venue, tournament weight + leakage tests |
| Fri 16 – Sun 18 *(in parallel)* | **3 — Linear Regression from scratch** | `LinearRegressionScratch` (normal equation + gradient descent + L2), unit tests, loss curve |
| Mon 19 | **4 — Split, baselines, Algorithm 1 results** | Chronological split, Dummy + home-advantage baselines, scratch-LR results and tuning of λ |
| Tue 20 | **5 — Random Forest + tuning** | Grid search with `TimeSeriesSplit`, before/after table |
| Wed 21 | **6 — Evaluate, compare, visualise** | Final tables and figures, error analysis per team / venue. **Freeze results** |
| Thu 22 – Sat 24 | **7 — Report + notebook polish** | Report (Word → PDF) in the Lab 1/2 style, notebook clean "Run All" |
| **Sun 25 Oct** | **Submit** | Code + report on Canvas |
| Mon 26 | **8 — Slides + rehearsal** | 10–12 slides, both practise explaining the scratch code |
| **Tue 27 / Wed 28** | **Presentation** | Our slot is announced on Canvas |

Buffer: Phases 2 and 3 run in parallel (one per person), which leaves Sat 24 as slack.

## 4. The data

**Source:** Mart Jürisoo, *International football results from 1872* — Kaggle `martj42/international-football-results-from-1872-to-2017` (the link in our Step 1 submission). The same files are published by the author on GitHub (`martj42/international_results`), which is where `data/raw/` was downloaded from on 13 Oct 2026.

| File | Rows | Use |
|---|---|---|
| `results.csv` | **49,945** matches, 1872-11-30 → 2026-10-06, **no missing values** | Main table: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral` |
| `former_names.csv` | 36 | Map old team names to current ones (e.g. Dahomey → Benin) so ratings follow a team through history |
| `goalscorers.csv` | 47,914 | Optional extra features (e.g. share of penalty goals) — **never for the match being predicted** |
| `shootouts.csv` | 700 | Not needed — penalty shoot-outs don't change the score |

Facts that shape the design (computed on 13 Oct):

| | Matches | Notes |
|---|---|---|
| All matches since 1994 | 30,435 | 324 teams |
| Training window 1994-01-01 → 2023-08-31 | 26,845 → **25,266** after the FIFA-team filter | |
| Test window 2023-09-01 → 2025-09-09 | 2,375 → **2,269** after the FIFA-team filter | all internationals in the 2026 qualifying period |
| CONMEBOL vs CONMEBOL World Cup qualifiers | 886 in total, 683 since 1996 | round-robin format since the 1998 qualifiers |
| **2026 qualifying cycle (main test set)** | **90** | Sep 2023 → Sep 2025, 18 rounds × 5 matches |
| After the test window (incl. 104 World Cup 2026 matches) | 1,215 | optional bonus check |

| Score statistics | Home goals | Away goals | Home win / draw / away win |
|---|---|---|---|
| All matches since 1994 | 1.66 | 1.12 | 48.5% / 23.2% / 28.2% |
| CONMEBOL qualifiers since 1996 | 1.64 | 0.90 | **54.0% / 25.0% / 20.9%** |
| 2026 cycle | 1.43 | 0.60 | 54.4% / 30.0% / 15.6% |

Home scores since 1994: 0 goals 26%, 1 goal 30%, 2 goals 22%, 3 goals 11%, 6+ goals 3% — **small, skewed counts** with a few extreme results (max 31).

## 5. Design decisions

| Decision | Choice | Why |
|---|---|---|
| **Target** | **Home score and away score** (two targets) | Agreed 13 Oct. Gives a real scoreline. Both algorithms handle the two targets at once: our scratch model fits a θ with 2 columns (one per target), `RandomForestRegressor` supports multi-output natively — same features, one fit, two predictions |
| **Training data** | **All international matches since 1994 between FIFA national teams** (teams that have played a World Cup or World Cup qualifier): **25,266** matches | Only ~600 CONMEBOL qualifiers exist before the test cycle — too few for a forest. Agreed 13 Oct. The FIFA-team filter (added 15 Oct) removes 120 regional/unrecognised teams (Catalonia, Tibet, …) and their extreme results |
| **Test data** | **The 90 CONMEBOL qualifiers of the 2026 cycle** (+ all 2,269 FIFA-team matches in the same window as a secondary, more stable test) | Answers our topic directly; the larger set shows whether the result holds in general |
| **Split** | **Chronological**: train < 1 Sep 2023 ≤ test | Matches are a time series. A random 80/20 split would let the model learn from *future* matches and give an over-optimistic score (Obsidian *Model Evaluation and Cross-Validation*: "never shuffle time series") |
| **Validation / tuning** | `TimeSeriesSplit(n_splits=5)` on the training window | Each fold trains on the past and validates on the following period |
| **Features** | Pre-match only (§6) | A feature that uses the match itself, or any later match, is leakage |
| **Metrics** | MAE, MSE, RMSE, R² per target and averaged; plus win/draw/loss accuracy and exact-score hit rate as secondary, presentation-friendly metrics | Guidelines list MAE/MSE/RMSE/R²; the derived metrics make the results understandable to an audience |

> [!note] Expect a low R²
> Goals are rare, noisy events. A model that explains a modest share of the variation in goals is normal for football. What matters is beating the baselines (§9) and explaining *why* one algorithm does better.

## 6. Feature engineering (`src/features.py`)

Every feature for a match on date *d* is computed **only from matches before *d***. Matches are processed in date order; a team's statistics are read *before* the current match is added.

| Feature (for home and away team) | Definition | Notes |
|---|---|---|
| **Elo rating** (`elo_home`, `elo_away`, `elo_diff`) | World Football Elo formula (eloratings.net): start 1500; expected result `We = 1 / (10^(−dr/400) + 1)` with `dr` = rating difference **+ 100 for the home team** unless neutral; update `R ← R + K·G·(W − We)` | `K` by importance: 60 World Cup finals · 50 continental championships · 40 World Cup / continental qualifiers · 30 other tournaments · 20 friendlies. `G` (goal-difference multiplier): 1 for a margin of 0–1, 1.5 for 2, (11 + N)/8 for N ≥ 3. Computed over **all matches since 1872** so ratings are "warmed up" by 1994 |
| **Recent form** | Over the team's last 10 matches: goals scored per match, goals conceded per match, points per match (3/1/0) | Fewer than 10 previous matches → use what exists + a `few_matches` flag |
| **Head-to-head** | Mean goal difference (home-team view) in the last 5 meetings | 0 when they have never met |
| **Venue** | `neutral` (0/1) | In CONMEBOL qualifiers since 1996 home teams average 1.64 goals and away teams 0.90 — a gap of ≈ 0.7 goals |
| **Match importance** | Elo `K` value of the tournament (20–60) or a one-hot of 5 groups | Friendlies are played differently from qualifiers |
| **Year** | Calendar year | Long-term trends in scoring |
| **Altitude** *(promoted from stretch after Phase 1)* | Venue altitude in metres for high cities (La Paz 3,640, El Alto ≈ 4,100, Quito ≈ 2,850, Bogotá ≈ 2,640, …), 0 otherwise | Phase 1: Bolivia takes 1.59 points per game at home vs 0.13 away, Ecuador 2.13 vs 0.72 — the largest home/away gaps in CONMEBOL. Small hand-made lookup table, documented |
| *Stretch:* **rest days** | Days since the team's previous match | |

**Status 18 Oct — implemented** in `src/features.py`: 17 model features (`FEATURES`): `elo_home`, `elo_away`, six form columns, `h2h_gd`, `h2h_n`, `neutral`, `altitude_km`, four tournament-group dummies (friendly = reference) and `year`. `elo_diff` is kept for analysis only (it is `elo_home − elo_away`, so including it would make the normal equation singular). Rest days were not added (stretch). 9 training rows (debut matches of new teams) have no form → filled with training medians in Phase 4.

**Leakage tests** (`tests/test_features.py`):
- The first ever match of a team gets the start values (Elo 1500, no form).
- Changing the score of a match changes **no** feature of that match or of earlier matches — only of later ones.
- No feature column correlates suspiciously with the target on the test set (e.g. |r| > 0.9).

## 7. Algorithm 1 — Linear Regression from scratch (`LinearRegressionScratch`)

Only NumPy (+ pandas for data handling). Shown in full in the notebook (section 7) because it is the graded part, with a short explanation of every step.

```python
class LinearRegressionScratch:
    """Multi-output linear regression: Y (m×2: home, away) ≈ X·θ + b."""
    def __init__(self, method="normal", learning_rate=0.05, n_iters=2000, l2=0.0, tol=1e-8): ...
    def fit(self, X, Y): ...        # adds the bias column, then normal equation or gradient descent
    def predict(self, X): ...
    # after fit: theta_ (n+1 × 2), loss_history_ (MSE per iteration for gradient descent)
```

| Part | Formula / rule |
|---|---|
| Model | `Ŷ = X_b · θ`, with `X_b` = X plus a column of 1s (bias) |
| Cost | MSE = (1/m) Σ (ŷ − y)², summed over the two targets |
| **Normal equation** | `θ = (X_bᵀX_b + λ·I*)⁻¹ X_bᵀ Y` — solved with `np.linalg.solve` (no matrix inverse); `I*` = identity with 0 for the bias (the bias is not regularised) |
| **Batch gradient descent** | `θ ← θ − η · [(2/m) X_bᵀ(X_bθ − Y) + 2λθ*]`; stop after `n_iters` or when the loss improves less than `tol` |
| Scaling | Standardise X with the **training** mean and std (gradient descent needs it; it also makes the coefficients comparable) |
| Hyperparameters to tune | `l2` (λ: 0, 0.01, 0.1, 1, 10) with our own `TimeSeriesSplit` loop; `learning_rate` / `n_iters` checked with the loss curve |
| Correctness tests (`tests/test_linear_regression.py`) | Synthetic data with a known θ → both methods recover it; gradient descent ≈ normal equation (`np.allclose`); the loss never increases for a small learning rate |
| Interpretation | Standardised coefficients per target (e.g. "+1 SD of Elo difference → +0.4 home goals") |

**Status 18 Oct — implemented** (`src/linear_regression.py`, 10 tests). On the training data: normal equation 0.03 s; gradient descent η = 0.1 → 414 iterations, θ within 7·10⁻⁶ of the normal equation; stable for η < 0.27 (largest eigenvalue 3.70), η = 0.3 diverges. Training R² 0.28 (home) / 0.23 (away). λ tuning follows in Phase 4.

Note for the report: predictions are continuous (e.g. 1.43 goals) and can in principle be negative — we clip at 0 for the scoreline and discuss it as a limitation of a linear model for count data.

## 8. Algorithm 2 — Random Forest Regressor

- `RandomForestRegressor(random_state=42)` fitted on the same features and both targets at once (multi-output).
- No scaling needed; same chronological split.
- Grid search with `TimeSeriesSplit(n_splits=5)`, scoring `neg_root_mean_squared_error`:

| Hyperparameter | Values |
|---|---|
| `n_estimators` | 200, 500 |
| `max_depth` | None, 10, 20 |
| `min_samples_leaf` | 1, 5, 20 |
| `max_features` | 1.0, 0.5, `"sqrt"` |

- Report before vs after tuning (like the labs). Run time should be manageable: ~27k rows × ~15 features (no huge one-hot like Lab 1). If not, drop `n_estimators = 500` first.

## 9. Evaluation and comparison

| What | How |
|---|---|
| **Baselines** | (1) Dummy: mean home and away score of the training window; (2) home advantage: mean scores of non-neutral training matches |
| **Main metrics** | MAE, MSE, RMSE, R² — for home goals, away goals and the average — on the **90 CONMEBOL qualifiers** and on all **2,269** test-window matches |
| **Secondary metrics** | Win/draw/loss accuracy from the predicted goal difference (draw band chosen on the training data) · exact-score hit rate after rounding |
| **Overfitting check** | Train vs test scores for both models |
| **Interpretation** | Scratch-LR coefficients · permutation importance for both models · residuals vs predicted · error per team and venue (altitude matches, big favourites) |
| **Figures** (≈ 6–8) | Score distribution · home advantage · Elo over time for the 10 CONMEBOL teams · gradient-descent loss curve · predicted vs actual · feature importance · error per team |
| *Bonus* | Predictions for the 2026 World Cup matches of CONMEBOL teams (data after the test window) |

**Status 20 Oct — Phase 4 results (Algorithm 1, test set):**

| | MAE | RMSE | R² home / away | W/D/L acc. |
|---|---|---|---|---|
| All 2,269 test matches — scratch LR (λ = 0.01) | 0.93 | 1.23 | 0.29 / 0.23 | 59.9% |
| All 2,269 test matches — best baseline | 1.07 | 1.43 | 0.01 / 0.00 | 47.0% |
| 90 CONMEBOL qualifiers — scratch LR | 0.91 | 1.11 | 0.19 / −0.14 | 50.0% |
| 90 CONMEBOL qualifiers — best baseline | 0.97 | 1.20 | −0.03 / −0.19 | 54.4% |

λ tuning: best λ = 0.01 but only 0.0002 better than λ = 0 (no overfitting). Main limitations found: away goals over-predicted in CONMEBOL qualifiers (0.98 vs 0.60 actual — stronger home advantage than average), no draws predicted (best draw band on training = 0), Bolivia at El Alto (never in training). **Not fixed with new features** — that would be tuning on the test set; goes into the discussion / future work.

**Status 20 Oct — Phase 5 results (Algorithm 2, Random Forest + tuning):**

Grid search over 54 combinations × 5 `TimeSeriesSplit` folds. Best configuration: `n_estimators=500`, `max_depth=10`, `min_samples_leaf=5`, `max_features=0.5`. Run time depends on the parallelism: the largest grid cells (500 trees, no depth limit, leaf size 1) need about 0.9 GB per forest, so the notebook caps the grid search at `GRID_N_JOBS = 3` parallel fits to stay within the memory of a 16 GB laptop. That takes roughly 25 minutes (about 5 minutes with all cores on a machine with enough memory); the results are identical.

| Model | Val RMSE (5 folds) | All Test RMSE | CONMEBOL Test RMSE | Negative Preds | Notes |
|---|---|---|---|---|---|
| Random Forest (default) | 1.2752 | 1.2484 | 1.1810 | 0 | Severe overfitting (train RMSE 0.4748, 11,682 leaves/tree) |
| **Random Forest (tuned)** | **1.2430** | 1.2254 | 1.1388 | 0 | Leaves drop to 399 (−97%), beats default in 5/5 folds |
| Scratch LR (λ = 0.01) | 1.2528 | 1.2271 | **1.1090** | 117 | Beats baselines, but outputs negative goal predictions |
| Best baseline | — | 1.4270 | 1.1960 | 0 | Venue baseline |

Key findings:
- **All 2,269 test matches:** the tuned forest (1.2254) and the scratch linear regression (1.2271) are effectively tied; the forest's lead is 0.0017 goals. The forest is clearly better than the baseline (1.4270) and never predicts a negative number of goals (0 vs 117 for the linear model).
- **90 CONMEBOL qualifiers:** the scratch linear regression is better than the tuned forest on every metric (RMSE 1.109 vs 1.139, MAE 0.907 vs 0.942, R² away −0.14 vs −0.27). The tuned forest still improves on the baseline (1.139 vs 1.196) and on the default forest (1.181).
- **Away goals:** the forest over-predicts away goals in the CONMEBOL qualifiers even more than the linear model (1.12 goals per match vs 0.98; actual 0.60). This is one of the main findings for our topic and goes into the discussion.
- Both algorithms are ready for the comprehensive comparison in Phase 6.

## 10. Repository and notebook structure

```
Project_machine_learning/
├── data/raw/              downloaded CSVs (git-ignored, see README)
├── data/processed/        feature table (git-ignored, rebuilt by the notebook)
├── docs/PLAN.md           this plan
├── notebooks/project.ipynb   THE submission notebook
├── src/
│   ├── data.py            load + clean (former names, CONMEBOL list, date filters)
│   ├── features.py        Elo, form, head-to-head, venue, importance
│   └── evaluation.py      metrics, baselines, plots
├── tests/                 pytest: from-scratch LR correctness + leakage tests
└── reports/               report (.docx) and slides
```

**Notebook = the guidelines' 11 steps:** 1 Import libraries · 2 Load the dataset · 3 Explore the data · 4 Preprocess · 5 Define features and target · 6 Chronological train/test split · **7 Algorithm 1 from scratch (class shown in full)** · 8 Random Forest · 9 Evaluate both · 10 Compare · 11 Visualisations. Data cleaning and feature code live in `src/` (imported, with docstrings) so the notebook stays readable. **Changed 18 Oct:** the from-scratch model lives in `src/linear_regression.py` so that `pytest` can test it; notebook section 7 explains the maths step by step and shows the **complete class code** (syntax-highlighted), so the grader still sees all of it.

## 11. Work split (proposal)

| | Lawrence | Gunnar |
|---|---|---|
| Phase 1 | Loading, team-name cleaning | EDA figures (score distribution, home advantage, altitude) |
| Phase 2 / 3 | **Feature engineering** (Elo, form, head-to-head) + leakage tests | **LinearRegressionScratch** (normal equation, gradient descent, L2) + tests |
| Phase 4 / 5 | Random Forest + grid search | Baselines + scratch-LR tuning + results |
| Phase 6 | Error analysis per team / venue | Comparison tables, final figures |
| Report | Intro, Aim, Dataset, Preprocessing | Methods, Results, Comparison & Discussion; Conclusion together |
| Slides | Topic, data, features | Methods (scratch code walkthrough), results |

**Both:** review every pull request of the other person and be able to explain *all* of the code — the instructor may ask either of us about any part. Branch per task, PR into `main`, one notebook owner at a time (as in the labs).

## 12. Report and presentation

**Report** (Word, same layout as our Lab 1/2 reports, ≈ 6–8 pages):

| Section | Content |
|---|---|
| Introduction | Football prediction, CONMEBOL qualifiers, why it is interesting (home advantage, altitude) |
| Project Aim | Predict home and away goals of CONMEBOL qualifiers from pre-match information; compare a from-scratch linear model with a random forest |
| Dataset Description | Source, 49,945 matches, columns, our training/test windows, score statistics |
| Data Preprocessing | Team-name cleaning, filters, feature engineering, leakage rules, scaling |
| Methods | **Linear regression from scratch in detail** (model, cost, normal equation, gradient descent, L2, our implementation + tests); random forest; tuning; chronological split |
| Results | Tables vs baselines, figures, train vs test |
| Comparison and Discussion | Which model is better and why, overfitting, limitations (noise, count data, small test set), dataset influence |
| Conclusion | Summary, best model, future work (Poisson regression, more features, player data) |
| References | Géron (2019) ch. 2, 4, 7 · Elo method · dataset · scikit-learn |

**Presentation** (10–12 slides, ≈ 10 min): topic → dataset → preprocessing + features → the two methods (**live walk through `LinearRegressionScratch`**) → results → comparison → example predictions (2026 cycle) → conclusion. Both present; both can answer questions about any slide.

## 13. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Weak predictive power (football is noisy) | Set expectations early; compare against baselines; add W/D/L accuracy so results are interpretable |
| Leakage through features | Strict pre-match computation, chronological split, leakage tests in `tests/` |
| Small main test set (90 matches) | Also report all 2,269 test-window matches; optional bootstrap confidence interval |
| Scratch LR bugs / numerical problems | Standardise features, `np.linalg.solve`, gradient descent cross-checked against the normal equation in tests |
| Time (12 days, labs in parallel) | Stretch features are optional; results frozen on Wed 21 |
| Unclear rules | Ask the teacher the questions in §14 early (this week) |

## 14. Questions for the teacher (ask this week)

1. Is a **chronological** train/test split acceptable instead of a random 80/20 split? (We think it's the correct choice for time-ordered matches.)
2. Inside the from-scratch implementation, is using **`np.linalg.solve`** for the normal equation fine, or should we rely on gradient descent only? (We plan to implement both.)
3. Is evaluating on the 90 CONMEBOL qualifiers while training on all international matches in line with our submitted topic?

## 15. Definition of done

- [ ] `notebooks/project.ipynb` runs top to bottom ("Run All") without errors, saved **with outputs**
- [ ] Algorithm 1 written from scratch, explained, tested; no library regression model used for it
- [ ] Algorithm 2 trained, tuned, compared before/after
- [ ] Both models evaluated on the same test sets against the baselines, with MAE/RMSE/R² (+ MSE)
- [ ] No leakage: chronological split, pre-match features, tests pass (`pytest`)
- [ ] Figures readable and referenced in the report
- [ ] Report covers all 8 sections, written in our own words; both of us have read all of it
- [ ] Code + report uploaded to Canvas by **25 Oct**
- [ ] Slides done; both of us can explain the scratch implementation and every result
