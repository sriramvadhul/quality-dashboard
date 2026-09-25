# Manufacturing Quality Dashboard — Defect Detection & Inspection Alert Prioritization

## The problem

A semiconductor production line has hundreds of sensors monitoring every unit that
comes off the line. Not every signal matters, and quality teams can't manually
inspect every unit — inspection capacity is limited. The question isn't just
*"can we detect defects?"* — it's *"given a fixed inspection budget, which units
should we actually pull aside to catch the most real defects?"*

## What this project does

1. **ETL (`src/etl.py`)** — loads raw sensor data (590 anonymized sensor readings
   per unit), drops sensors too unreliable to trust (>35% missing), imputes the
   rest, and removes zero-information sensors. 590 → 588 usable sensors.
2. **Modeling (`src/train_model.py`)** — trains an XGBoost classifier on a
   **chronological** train/test split (no shuffling — the model only ever sees
   the past when predicting the future, which avoids data leakage and mirrors
   how the model would actually be deployed).
3. **Alert prioritization** — instead of stopping at an accuracy metric, the
   model ranks every unit by risk and reports precision/recall **at a fixed
   inspection budget** (e.g., "if we can only inspect 100 units, we catch 80%
   of real defects — a 3x lift over random inspection").
4. **Dashboard (`app.py`)** — a Dash app with three panels aimed at a quality
   manager, not a data scientist: defect rate over time, which sensors are
   driving the risk score, and a ranked, sortable table of units flagged for
   inspection today.

## Why synthetic data

The real dataset this project is modeled on is the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom)
(1,567 semiconductor production units, 590 sensors, ~6.6% defect rate). It
requires a direct download from UCI or Kaggle. `data/generate_synthetic_secom.py`
generates a dataset with the **identical schema and statistical shape** (same
column names, same missingness pattern, same class imbalance, same handful of
sensors that actually matter) so every downstream script runs unmodified.

**To use the real data instead:** download `uci-secom.csv` from UCI or Kaggle,
rename it to `raw_secom.csv`, place it in `data/`, and rename its label column
to `Pass_Fail` and timestamp column to `Timestamp` if needed. No other code
changes required.

## Results (on synthetic data)

| Metric | Value |
|---|---|
| ROC-AUC | 0.89 |
| Average Precision | 0.55 |
| Precision @ 100-unit budget | 20% |
| Recall @ 100-unit budget | 80% |
| Lift over random inspection | 3.1x |

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the sensor data (or drop in the real UCI/Kaggle CSV — see above)
python3 data/generate_synthetic_secom.py

# 2. Clean it
python3 src/etl.py

# 3. Train the model and score every unit
python3 src/train_model.py

# 4. Launch the dashboard
python3 app.py
# then open http://127.0.0.1:8050
```

## Design choices worth calling out in an interview

- **Chronological split, not random** — same leakage-avoidance principle used
  in production fraud/quality systems, where you never get to see the future
  when scoring the present.
- **Alert prioritization framing, not just accuracy** — a quality team has a
  finite inspection budget; the model is evaluated against that real
  operational constraint, not an abstract metric.
- **Missingness handled explicitly** — real sensor data fails in specific,
  informative ways (some sensors are just flakier); dropping vs. imputing is a
  deliberate threshold, not a default.
- **Dashboard built for the stakeholder, not the model** — three panels
  answer three concrete questions a quality manager would ask, in plain
  language, not a wall of diagnostic plots.
