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

```python
# The "Golden Features" that saved my score
df['drying_velocity_3d'] = df['5cm_soli_moist'].diff(3) / 3
df['drying_accel_3d'] = df['drying_velocity_3d'].diff(3)
df['cumulative_vpd_7d'] = df['vapor_pressure_deficit'].rolling(7).sum()
