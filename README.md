# IndabaX-hackathon-
# 🌾 Dry Spell Prediction: From Starter Notebook to Top 10

![Rank](https://img.shields.io/badge/Rank-7th_Place-bronze?style=for-the-badge) ![F1 Score](https://img.shields.io/badge/F1_Score-0.59-blue?style=for-the-badge) ![Model](https://img.shields.io/badge/Model-XGBoost-green?style=for-the-badge)

> **7th Place Solution** | *IndabaX Sudan Dry Spell Prediction Hackathon*

This repository documents my journey to crack the code on agricultural drought prediction. Starting with the competition's **Starter Notebook**, I engineered a solution that climbed to **7th Place** by injecting domain-specific physics and creating a dynamic thresholding strategy.

---

## 🎯 The Challenge
The goal was to predict `dryspell_warn_7d`—a binary target indicating if a region would suffer a critical dry spell in the next week.

* **The Problem:** Droughts are "rare events" (imbalanced classes).
* **The Trap:** A model can get 90% accuracy by just predicting "No Drought," but that fails the farmer. We needed to maximize **F1 Score (Class 1)**, balancing Precision with Recall.

## 📉 The Rollercoaster (My Journey)

This wasn't a straight line to success. It was a series of pivots around the baseline code.

### 🐣 Phase 1: The Baseline
* **Approach:** I started with the provided `starter_notebook.ipynb`. It gave a decent baseline but struggled to differentiate between "normal dry" and "critical dry."
* **Result:** Moderate scores, lots of noise.

### 💀 Phase 2: The Crash
* **Approach:** I tried to "outsmart" the starter code by removing time features (thinking they confused the model) and trying complex deep learning architectures.
* **Result:** `F1: 0.09` (Catastrophic)
* **Lesson:** **Context is King.** A dry day in July (normal) is physically different from a dry day in October (harvest season). The starter notebook's time features were actually critical.

### 🚀 Phase 3: The Physics Pivot (The Turning Point)
I went back to the Starter Notebook but stopped treating the data as just numbers. I started thinking like an Agronomist. I engineered features that describe **stress**:
1.  **Drying Velocity:** Not just "is it dry?", but "how *fast* is it drying?"
2.  **Cumulative Thirst:** How much Vapor Pressure Deficit (VPD) has accumulated over 7 days?

* **Result:** `F1: 0.57` → **0.59** (Peak Performance)

---

## 🛠️ The Winning Solution (Rank 7)

My final submission is essentially the **Starter Notebook on Steroids**. I kept the core XGBoost structure but turbocharged the feature engineering and post-processing.

### 1. Feature Engineering: "The Physics Engine"
Instead of raw sensor data, I fed the model rates of change.


# The "Golden Features" that saved my score
df['drying_velocity_3d'] = df['5cm_soli_moist'].diff(3) / 3
df['drying_accel_3d'] = df['drying_velocity_3d'].diff(3)
df['cumulative_vpd_7d'] = df['vapor_pressure_deficit'].rolling(7).sum()
2. Strategy: Dynamic Seasonal Thresholds
I discovered that a global probability threshold (e.g., > 0.5) failed because the risk profile changes by month. I implemented a Sliding Net logic on top of the starter model's predictions:

July (Safe Season): Set threshold to 92nd percentile (Strict). Ignore weak signals.

October (Danger Season): Set threshold to 80th percentile (Loose). Catch every possible sign of drought.

3. Post-Processing (Gap Filling)
Logic: Used physics constraints to smooth predictions. If Day 1 is Dry and Day 3 is Dry, Day 2 must be Dry (soil doesn't heal overnight).

🔍 What I Missed (The Gap to Rank 1)
The top winners (F1 ~0.66) bridged the gap that I couldn't cross. In retrospect, here is what separated Rank 7 from Rank 1:

The "Consecutive Days" Feature:

My model looked at "velocity," but it didn't explicitly count "How many days has the soil been stressed?"

This duration feature is likely the strongest predictor of crop failure.

Ensembling:

I bet everything on a single XGBoost.

The winners likely stacked XGBoost + CatBoost + LightGBM. The ensemble "committee" smooths out the edge-case errors that trapped my single model.

Training Data Split:

I trained on all years (2002-2019). The climate in 2002 is different from 2025.

A better strategy would have been weighting recent years (2016-2019) higher to capture modern climate change trends.

📂 Repository Structure
├── data/               # Raw input files (Train/Test)
├── starter_notebook_tuned.ipynb  # The main solution file
└── README.md
🧠 Final Thoughts
This competition taught me that Domain Knowledge > Complex Algorithms.

I could have run a GridSearch for days, but realizing that "Soil drying velocity matters more than absolute moisture" is what jumped my score by 20 points. I finished 7th place solo against teams using complex ensembles, proving that a strong physical hypothesis can carry you far.
