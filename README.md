# NYC Taxi Trip Duration — ML Pipeline & Interactive Predictor

This project has two parts that work together:

1. **`NYC_Taxi.ipynb`** — a full machine-learning pipeline (in 7 steps) that cleans NYC taxi trip data, engineers features, handles a heavily skewed target with **SMOGN**, trains and compares several regression models, and saves the best one to disk.
2. **`streamlit_app.py`** — an interactive **Streamlit** web app where a user picks a pickup/dropoff location on a map and gets a predicted trip duration from the model trained in the notebook.

---

## Project Structure

```
project/
├── NYC_Taxi.ipynb          # Training pipeline (run this first)
├── streamlit_app.py        # Interactive predictor (run this second)
├── data.csv                # Raw NYC taxi trip dataset (required, see Requirements)
├── models/                 # Created by the notebook
│   ├── best_model_<Name>.pkl
│   ├── feature_scaler.pkl
│   ├── selected_features.pkl
│   └── model_metadata.json
└── plots/                  # Created by the notebook — static PNG charts (EDA, model comparison, etc.)
```

> ⚠️ **Important**: the Streamlit app does **not** look inside the `models/` folder — it expects `data.csv`, `model_metadata.json`, and the `.pkl` files to sit in the **same folder as `streamlit_app.py`**. See [Setup & Running](#setup--running) below.

---

## Part 1 — The Notebook (`NYC_Taxi.ipynb`)

A 7-step pipeline:

| Step | What it does |
|---|---|
| **1. Data Collection** | Loads `data.csv`, prints shape/stats, and runs an exploratory analysis (trip duration histogram, trips by hour/day of week, passenger count distribution, vendor share, pickup heatmap). |
| **2. Data Tidying & Cleaning** | Drops duplicates, parses datetimes, removes trips with `trip_duration` outside **60–10,800 seconds** (1 min–3 hrs), keeps only trips inside an NYC bounding box (`lat 40.5–40.9`, `lon -74.3 to -73.6`), and restricts `passenger_count` to **1–6**. |
| **3. Feature Extraction** | Computes `haversine_dist`, `manhattan_dist`, `bearing` between pickup/dropoff, plus temporal features (`pickup_hour`, `pickup_dayofweek`, `pickup_month`, `pickup_day`, `is_weekend`, `is_rush_hour`, `is_night`), encodes `store_and_fwd_flag`, and creates the modeling target `log_duration = log1p(trip_duration)`. |
| **4. Feature Selection** | Compares two methods — `SelectKBest` (F-regression, p < 0.05) and Random Forest feature importances — and merges the features picked by either method into a final `selected_features` list. |
| **5. SMOGN (target rebalancing)** | The raw target is heavily right-skewed; `log_duration` is far closer to normal. **SMOGN** (`smogn.smoter`, `k=5`, `rel_thres=0.9`) is applied to the **training set only** to synthetically oversample under-represented (rare/extreme) duration values — the test set stays 100% real data for honest evaluation. |
| **6. Model Comparison** | Trains and compares 5 models on SMOGN-augmented, scaled (`StandardScaler`) data: **Ridge**, **Random Forest**, **Gradient Boosting**, a **tuned GBM**, and a **tuned Random Forest**. Each is scored with MAE, RMSE, and R² (on the real, non-synthetic test set). The best model (highest R²) is saved with `joblib`, along with the fitted scaler, the selected feature list, and a `model_metadata.json` summarizing the run. |
| **7. Data Visualization** | Saves PNG charts to `plots/`: target distribution before/after log-transform & SMOGN, model comparison bars (MAE & R²), actual-vs-predicted scatter + residuals, and feature importance (for tree-based winners; skipped for Ridge). |

**Key implementation detail to remember:** the model is trained on `log1p(trip_duration)`, so any inference code must invert it with `expm1()` — this is exactly what the Streamlit app's `model_raw_to_trip_seconds()` does, clipping the result back to the same `[60, 10800]` second range used during cleaning.

---

## Part 2 — The Streamlit App (`streamlit_app.py`)

A dark-themed, map-driven UI (folium + `streamlit-folium`) built around the model from Part 1.

**Sidebar**
- Pick which saved `.pkl` model to use (auto-discovers every model file next to the script, excluding the scaler/feature files).
- Choose vendor (1 or 2) and passenger count (1–6) with quick-select buttons.
- Toggle "store & forward" flag.
- Reset the map to a sensible default pickup/dropoff pulled from `data.csv`.

**Main area**
- "Set Pickup" / "Set Dropoff" buttons, then click anywhere on the interactive map to place each marker (connected by a dashed line). Clicks are clamped to the lat/lon range seen in `data.csv`.
- Date & time picker for the pickup (defaults to the **next Wednesday at 08:00**, a deliberately "average" weekday/time).
- A collapsible **debug panel** showing the exact feature vector that will be sent to the model — useful for catching missing/misnamed features.
- **Predict** button that:
  - Rebuilds the same feature set used in training (`build_model_features`, mirroring the notebook's Step 3 features).
  - Scales the features with the saved `StandardScaler` (if valid) and runs the chosen model.
  - Converts the model's `log1p` output back to seconds and displays **duration, distance (km), average speed, passenger count, and a Rush-Hour/Night/Normal time badge**.
  - **Falls back to a baseline estimate** (median historical seconds-per-km for that pickup hour, computed once from `data.csv`) if the model/scaler/features can't be loaded or prediction throws an error — and shows an info message saying so, instead of crashing.

---

## Requirements

### For the notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib smogn
```
> `smogn` can be picky about its `numpy`/`scikit-learn` version compatibility — if installation fails, try installing it in a dedicated virtual environment.

### For the Streamlit app
```bash
pip install streamlit pandas numpy folium streamlit-folium joblib
```
(`joblib` is optional — the app falls back to plain `pickle` if it isn't installed, but `joblib` is what the notebook uses to save the model, so it's recommended.)

---

## Setup & Running

### 1. Run the notebook
Place `data.csv` (the raw NYC taxi trip dataset, with columns like `pickup_datetime`, `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`, `trip_duration`, `vendor_id`, `passenger_count`, `store_and_fwd_flag`) in the same folder as `NYC_Taxi.ipynb`, then run all cells top to bottom. This creates a `models/` folder containing:
- `best_model_<ModelName>.pkl`
- `feature_scaler.pkl`
- `selected_features.pkl`
- `model_metadata.json`

### 2. Flatten the artifacts next to the app
Copy `data.csv` and **everything from `models/`** into the same folder as `streamlit_app.py` (not in a subfolder):

```
app_folder/
├── streamlit_app.py
├── data.csv
├── best_model_<ModelName>.pkl
├── feature_scaler.pkl
├── selected_features.pkl
└── model_metadata.json
```

### 3. Launch the app
```bash
streamlit run streamlit_app.py
```

---

## Notes & Limitations

- The NYC bounding box (`40.5–40.9` lat, `-74.3 to -73.6` lon) is hardcoded in the notebook's cleaning step; trips outside it are dropped during training and map clicks outside the data's observed range are clamped in the app.
- Trip duration is capped to **1 minute – 3 hours** both during training-data cleaning and at inference time.
- SMOGN only ever touches the **training** split — all reported metrics (MAE/RMSE/R²) and the app's baseline fallback are computed from real, unaltered data.
- Some inline comments in the source (written by the original author) are in Arabic — they don't affect functionality, just developer notes.
- The app assumes a single best model is selected at training time; if you want to compare multiple models live in the UI, keep more than one `.pkl` file next to `streamlit_app.py` and use the sidebar's model dropdown.
