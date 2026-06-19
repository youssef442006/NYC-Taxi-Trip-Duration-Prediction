import json
import os
import pickle
from datetime import date, datetime, time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

try:
    import joblib
except ImportError:
    joblib = None

st.set_page_config(page_title="NYC Taxi Duration Predictor", page_icon="🚕", layout="wide")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")


# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 800px at 12% -10%, #1e3a5f 0%, #0b1220 45%, #070b12 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1b2e 0%, #0a1424 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }
    .block-container { padding-top: 1rem; max-width: 1200px; }
    .hero {
        background: linear-gradient(135deg, rgba(30,58,138,0.5), rgba(14,165,233,0.18));
        border: 1px solid rgba(125,211,252,0.28);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .hero h2 { margin: 0; font-size: 1.65rem; letter-spacing: -0.02em; color: #f8fafc; }
    .hero p  { margin: 8px 0 0 0; color: #94a3b8; font-size: 0.96rem; }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .sidebar-section {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin: 18px 0 6px 0;
    }
    .result-box {
        background: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(30,58,138,0.25));
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 16px;
        padding: 18px 22px;
        margin-top: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def next_midweek_date():
    """يرجع أقرب أربعاء جاي (أو النهارده لو أربعاء)"""
    today = date.today()
    days_ahead = (2 - today.weekday()) % 7   # 2 = Wednesday
    return today if days_ahead == 0 else date.fromordinal(today.toordinal() + days_ahead)


def load_metadata():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv_profile():
    profile = {
        "has_data": False,
        "default_pickup":  (40.7580, -73.9855),
        "default_dropoff": (40.7484, -73.9857),
        "lat_range": (40.4, 41.0),
        "lon_range": (-74.3, -73.6),
        "sec_per_km_by_hour": {h: 180.0 for h in range(24)},
        "global_sec_per_km": 180.0,
        "sample_date": None,
        "default_vendor_id": 2,
        "default_passenger_count": 1,
    }
    if not os.path.exists(DATA_PATH):
        return profile
    try:
        df = pd.read_csv(DATA_PATH)
        need = {"pickup_latitude","pickup_longitude","dropoff_latitude",
                "dropoff_longitude","pickup_datetime","trip_duration"}
        if not need.issubset(df.columns):
            return profile

        clean = df.dropna(subset=list(need)).copy()
        clean["pickup_datetime"] = pd.to_datetime(clean["pickup_datetime"], errors="coerce")
        clean["trip_duration"]   = pd.to_numeric(clean["trip_duration"], errors="coerce")
        clean = clean.dropna(subset=["pickup_datetime","trip_duration"])
        clean = clean[clean["trip_duration"] > 0]
        if clean.empty:
            return profile

        profile["has_data"] = True
        all_lat = pd.concat([clean["pickup_latitude"], clean["dropoff_latitude"]])
        all_lon = pd.concat([clean["pickup_longitude"], clean["dropoff_longitude"]])
        profile["lat_range"] = (float(all_lat.min())-0.05, float(all_lat.max())+0.05)
        profile["lon_range"] = (float(all_lon.min())-0.05, float(all_lon.max())+0.05)

        clean["dist_km"] = haversine_km(
            clean["pickup_latitude"].to_numpy(float),
            clean["pickup_longitude"].to_numpy(float),
            clean["dropoff_latitude"].to_numpy(float),
            clean["dropoff_longitude"].to_numpy(float),
        )
        clean = clean[clean["dist_km"] > 0.05]
        if clean.empty:
            return profile

        med = float(np.median(clean["dist_km"]))
        row = clean.iloc[int(np.abs(clean["dist_km"].to_numpy() - med).argmin())]
        profile["default_pickup"]  = (float(row["pickup_latitude"]),  float(row["pickup_longitude"]))
        profile["default_dropoff"] = (float(row["dropoff_latitude"]), float(row["dropoff_longitude"]))

        if "vendor_id" in clean.columns:
            try:   profile["default_vendor_id"] = int(row["vendor_id"])
            except: pass
        if "passenger_count" in clean.columns:
            try:   profile["default_passenger_count"] = int(np.clip(int(row["passenger_count"]), 1, 6))
            except: pass

        clean["sec_per_km"] = clean["trip_duration"] / clean["dist_km"]
        clean = clean.replace([np.inf,-np.inf], np.nan).dropna(subset=["sec_per_km"])
        clean = clean[(clean["sec_per_km"] > 20) & (clean["sec_per_km"] < 2500)]
        if clean.empty:
            return profile

        global_spk = float(clean["sec_per_km"].median())
        by_hour    = clean.groupby(clean["pickup_datetime"].dt.hour)["sec_per_km"].median()
        profile["global_sec_per_km"]   = global_spk
        profile["sec_per_km_by_hour"]  = {h: float(by_hour.get(h, global_spk)) for h in range(24)}
    except Exception:
        pass
    return profile


META          = load_metadata()
SCALER_PATH   = os.path.join(BASE_DIR, os.path.basename(META.get("scaler_file",   "feature_scaler.pkl")))
FEATURES_PATH = os.path.join(BASE_DIR, os.path.basename(META.get("features_file", "selected_features.pkl")))
FEATURES_FROM_META = META.get("selected_features") or []

EXCLUDED_MODEL_FILES = {
    os.path.basename(SCALER_PATH).lower(),
    os.path.basename(FEATURES_PATH).lower(),
    "feature_scaler.pkl",
    "selected_features.pkl",
    "logistic_model_smote.pkl",
}


@st.cache_data(show_spinner=False)
def discover_models():
    if not os.path.isdir(BASE_DIR):
        return []
    out = [f for f in os.listdir(BASE_DIR)
           if f.lower().endswith(".pkl") and f.lower() not in EXCLUDED_MODEL_FILES]
    out.sort()
    return out


@st.cache_resource(show_spinner=False)
def load_object(path):
    if not os.path.exists(path):
        return None
    if joblib is not None:
        try:   return joblib.load(path)
        except Exception: pass
    with open(path, "rb") as f:
        return pickle.load(f)


def is_valid_scaler(obj):
    return obj is not None and hasattr(obj, "transform") and callable(obj.transform)


def normalize_feature_list(obj):
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]
    return []


@st.cache_data(show_spinner=False)
def training_feature_columns():
    feats = normalize_feature_list(load_object(FEATURES_PATH))
    return feats if feats else list(FEATURES_FROM_META)


def build_model_features(pickup_dt, lat1, lon1, lat2, lon2, vendor_id, passenger_count,
                            store_and_fwd_flag=0):
    """
    FEATURES في النوت بوك:
        vendor_id, passenger_count, store_and_fwd_flag,
        pickup_hour, pickup_dayofweek, pickup_month, pickup_day,
        is_weekend, is_rush_hour, is_night,
        haversine_dist, manhattan_dist, bearing,
        pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude
    """
    haversine = float(np.asarray(haversine_km(lat1, lon1, lat2, lon2)).ravel()[0])

    manhattan = (abs(lat2 - lat1) + abs(lon2 - lon1))

    bearing = float(np.arctan2(lon2 - lon1, lat2 - lat1))

    hour = pickup_dt.hour
    dow  = pickup_dt.weekday()   # 0=Mon … 6=Sun

    is_weekend   = int(dow >= 5)
    is_rush_hour = int(hour in [7, 8, 9, 17, 18, 19])
    is_night     = int(hour in list(range(22, 24)) + list(range(0, 6)))

    feature_dict = {
        "vendor_id":          float(int(np.clip(int(vendor_id), 1, 2))),
        "passenger_count":    float(int(np.clip(int(passenger_count), 1, 6))),
        "store_and_fwd_flag": float(store_and_fwd_flag),
        "pickup_hour":        float(hour),
        "pickup_dayofweek":   float(dow),         
        "pickup_month":       float(pickup_dt.month),
        "pickup_day":         float(pickup_dt.day),
        "is_weekend":         float(is_weekend),
        "is_rush_hour":       float(is_rush_hour),
        "is_night":           float(is_night),
        "haversine_dist":     haversine,
        "manhattan_dist":     float(manhattan),
        "bearing":            bearing,
        "pickup_latitude":    float(lat1),
        "pickup_longitude":   float(lon1),
        "dropoff_latitude":   float(lat2),
        "dropoff_longitude":  float(lon2),
    }
    return feature_dict, haversine


def vector_for_model(feature_dict, col_names):
    return np.array([[float(feature_dict.get(c, 0.0)) for c in col_names]], dtype=np.float64)


def model_raw_to_trip_seconds(raw_pred):
    raw_pred = float(raw_pred)
    if not np.isfinite(raw_pred):
        raise RuntimeError("Model returned non-finite prediction.")
    sec = float(np.expm1(raw_pred))
    return float(np.clip(sec, 60.0, 10800.0))


def predict_duration_seconds(model, scaler, col_names, feature_dict):
    x = vector_for_model(feature_dict, col_names)
    if scaler is not None and is_valid_scaler(scaler):
        x = scaler.transform(x)
    if model is None:
        raise RuntimeError("No model loaded.")
    raw = float(np.asarray(model.predict(x)).ravel()[0])
    return model_raw_to_trip_seconds(raw), raw


def baseline_seconds(dist_km, pickup_dt, profile):
    spk = profile["sec_per_km_by_hour"].get(int(pickup_dt.hour), profile["global_sec_per_km"])
    return float(np.clip(float(dist_km) * float(spk), 60.0, 10800.0))


# ─── Session state ─────────────────────────────────────────────────────────────
_INIT_VER = 5

def init_state_from_csv(profile):
    p_lat, p_lon = profile["default_pickup"]
    d_lat, d_lon = profile["default_dropoff"]
    st.session_state.pickup_lat        = p_lat
    st.session_state.pickup_lon        = p_lon
    st.session_state.dropoff_lat       = d_lat
    st.session_state.dropoff_lon       = d_lon
    st.session_state.map_mode          = "pickup"
    st.session_state.last_click_key    = None
    st.session_state.vendor_id         = int(profile.get("default_vendor_id", 2))
    st.session_state.passenger_count   = int(profile.get("default_passenger_count", 1))
    st.session_state.store_and_fwd     = 0


def init_state():
    profile = load_csv_profile()
    if st.session_state.get("_init_ver") != _INIT_VER:
        st.session_state["_init_ver"] = _INIT_VER
        init_state_from_csv(profile)
        return
    for key, default in [("map_mode","pickup"), ("last_click_key",None), ("store_and_fwd",0)]:
        if key not in st.session_state:
            st.session_state[key] = default
    for coord_key in ("pickup_lat","pickup_lon","dropoff_lat","dropoff_lon"):
        if coord_key not in st.session_state:
            init_state_from_csv(profile)
            break
    if "vendor_id"       not in st.session_state: st.session_state.vendor_id       = int(profile.get("default_vendor_id",2))
    if "passenger_count" not in st.session_state: st.session_state.passenger_count = int(profile.get("default_passenger_count",1))


init_state()
csv_profile   = load_csv_profile()
model_files   = discover_models()
training_cols = training_feature_columns()

META          = load_metadata()
default_model = os.path.basename(META.get("model_file","")) if META.get("model_file") else None
default_index = (model_files.index(default_model)
                 if default_model and default_model in model_files else 0)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚕 Settings")

    st.markdown('<div class="sidebar-section">🤖 Prediction Model</div>', unsafe_allow_html=True)
    if not model_files:
        st.error("No model (.pkl) found in the models/ folder.")
        selected_model_file = None
    else:
        selected_model_file = st.selectbox(
            "Model", model_files,
            index=min(default_index, len(model_files)-1),
            format_func=lambda n: n.replace(".pkl","").replace("_"," ").title(),
            label_visibility="collapsed",
        )

    st.markdown('<div class="sidebar-section">🏢 Vendor</div>', unsafe_allow_html=True)
    vendor_options = {1: "Vendor 1 — Creative Mobile", 2: "Vendor 2 — VeriFone Inc."}
    vendor_choice = st.radio(
        "Vendor", options=[1,2],
        format_func=lambda v: vendor_options[v],
        index=0 if st.session_state.vendor_id == 1 else 1,
        label_visibility="collapsed",
    )
    st.session_state.vendor_id = vendor_choice

    st.markdown('<div class="sidebar-section">👥 Passengers</div>', unsafe_allow_html=True)
    pax_cols = st.columns(6)
    for i, col in enumerate(pax_cols, start=1):
        is_active = st.session_state.passenger_count == i
        if col.button(
            f"**{i}**" if is_active else str(i),
            key=f"pax_{i}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.passenger_count = i
            st.rerun()
    st.markdown(
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">'
        f'Selected: <strong style="color:#38bdf8;">{st.session_state.passenger_count} passenger'
        f'{"s" if st.session_state.passenger_count > 1 else ""}</strong></p>',
        unsafe_allow_html=True,
    )

    # Store & Forward Flag
    st.markdown('<div class="sidebar-section">📡 Store & Forward</div>', unsafe_allow_html=True)
    sfwd = st.toggle("Trip stored before sending to server", value=bool(st.session_state.store_and_fwd))
    st.session_state.store_and_fwd = int(sfwd)

    st.divider()
    if csv_profile["has_data"] and st.button("↺ Reset Map", use_container_width=True):
        init_state_from_csv(load_csv_profile())
        st.rerun()

    st.markdown(
        '<p style="color:#475569;font-size:0.8rem;margin-top:16px;">'
        'NYC Taxi Trip Duration Predictor<br>Powered by Random Forest + SMOGN</p>',
        unsafe_allow_html=True,
    )


# ─── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h2>🚕 NYC Taxi Trip Duration Predictor</h2>
  <p>Click <strong>Pickup</strong> then <strong>Dropoff</strong> on the map, set your departure time, and let the model predict your trip duration.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("📍 Set Pickup",
                 type="primary" if st.session_state.map_mode=="pickup" else "secondary",
                 use_container_width=True):
        st.session_state.map_mode = "pickup"; st.rerun()
with c2:
    if st.button("🏁 Set Dropoff",
                 type="primary" if st.session_state.map_mode=="dropoff" else "secondary",
                 use_container_width=True):
        st.session_state.map_mode = "dropoff"; st.rerun()
with c3:
    mode_color = "#38bdf8" if st.session_state.map_mode=="pickup" else "#f472b6"
    mode_label = ("📍 Placing PICKUP — tap anywhere on the map"
                  if st.session_state.map_mode=="pickup"
                  else "🏁 Placing DROPOFF — tap anywhere on the map")
    st.markdown(
        f'<div style="padding:8px 14px;background:rgba(15,23,42,0.6);border-radius:10px;'
        f'border:1px solid rgba(148,163,184,0.2);color:{mode_color};font-weight:600;font-size:0.9rem;">'
        f'{mode_label}</div>', unsafe_allow_html=True)

# Map
center_lat = (st.session_state.pickup_lat + st.session_state.dropoff_lat) / 2
center_lon = (st.session_state.pickup_lon + st.session_state.dropoff_lon) / 2
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

folium.Marker([st.session_state.pickup_lat, st.session_state.pickup_lon],
              tooltip="📍 Pickup", icon=folium.Icon(color="green", icon="plus")).add_to(fmap)
folium.Marker([st.session_state.dropoff_lat, st.session_state.dropoff_lon],
              tooltip="🏁 Dropoff", icon=folium.Icon(color="red", icon="flag")).add_to(fmap)
folium.PolyLine([[st.session_state.pickup_lat, st.session_state.pickup_lon],
                 [st.session_state.dropoff_lat, st.session_state.dropoff_lon]],
                color="#38bdf8", weight=4, opacity=0.85, dash_array="8 4").add_to(fmap)

map_data = st_folium(fmap, height=460, use_container_width=True)
clicked  = map_data.get("last_clicked") if map_data else None

if clicked:
    click_key = f"{clicked['lat']:.6f},{clicked['lng']:.6f}"
    if click_key != st.session_state.last_click_key:
        st.session_state.last_click_key = click_key
        if st.session_state.map_mode == "pickup":
            st.session_state.pickup_lat  = float(np.clip(clicked["lat"], *csv_profile["lat_range"]))
            st.session_state.pickup_lon  = float(np.clip(clicked["lng"], *csv_profile["lon_range"]))
            st.session_state.map_mode    = "dropoff"
        else:
            st.session_state.dropoff_lat = float(np.clip(clicked["lat"], *csv_profile["lat_range"]))
            st.session_state.dropoff_lon = float(np.clip(clicked["lng"], *csv_profile["lon_range"]))
        st.rerun()

coord_c1, coord_c2 = st.columns(2)
with coord_c1:
    st.markdown(
        f'<div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:8px 14px;'
        f'border:1px solid rgba(56,189,248,0.15);font-size:0.85rem;color:#94a3b8;">'
        f'📍 <strong style="color:#4ade80;">Pickup</strong> &nbsp;'
        f'{st.session_state.pickup_lat:.5f}, {st.session_state.pickup_lon:.5f}</div>',
        unsafe_allow_html=True)
with coord_c2:
    st.markdown(
        f'<div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:8px 14px;'
        f'border:1px solid rgba(56,189,248,0.15);font-size:0.85rem;color:#94a3b8;">'
        f'🏁 <strong style="color:#f472b6;">Dropoff</strong> &nbsp;'
        f'{st.session_state.dropoff_lat:.5f}, {st.session_state.dropoff_lon:.5f}</div>',
        unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─── Date / Time — default = أقرب أربعاء ──────────────────────────────────────
col_d, col_t = st.columns(2)
with col_d:
    selected_date = st.date_input("📅 Pickup Date", value=next_midweek_date())
with col_t:
    selected_time = st.time_input("🕗 Pickup Time", value=time(8, 0))
pickup_dt = datetime.combine(selected_date, selected_time)

# Debug info — بيوضح الـ features اللي هتتبعت للموديل
with st.expander("🔍 Feature Preview (debug)", expanded=False):
    if training_cols:
        tmp_feat, tmp_dist = build_model_features(
            pickup_dt,
            st.session_state.pickup_lat, st.session_state.pickup_lon,
            st.session_state.dropoff_lat, st.session_state.dropoff_lon,
            st.session_state.vendor_id, st.session_state.passenger_count,
            st.session_state.store_and_fwd,
        )
        debug_df = pd.DataFrame([{col: tmp_feat.get(col, "❌ MISSING") for col in training_cols}]).T
        debug_df.columns = ["value"]
        st.dataframe(debug_df, use_container_width=True)
    else:
        st.warning("selected_features.pkl not loaded yet.")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ─── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮  Predict Trip Duration", type="primary", use_container_width=True):
    lat1, lon1 = st.session_state.pickup_lat,  st.session_state.pickup_lon
    lat2, lon2 = st.session_state.dropoff_lat, st.session_state.dropoff_lon

    if abs(lat1-lat2) < 0.0005 and abs(lon1-lon2) < 0.0005:
        st.warning("⚠️ Pickup and dropoff are the same location.")
        st.stop()

    if not selected_model_file:
        st.error("❌ No prediction model available.")
        st.stop()

    model   = load_object(os.path.join(BASE_DIR, selected_model_file))
    scaler  = load_object(SCALER_PATH)
    scaler  = scaler if is_valid_scaler(scaler) else None
    columns = training_feature_columns()

    if not columns:
        st.error("selected_features.pkl not found in models/ folder.")
        st.stop()

    used_fallback = False
    try:
        feat_row, dist_km = build_model_features(
            pickup_dt, lat1, lon1, lat2, lon2,
            st.session_state.vendor_id,
            st.session_state.passenger_count,
            st.session_state.store_and_fwd,
        )
        pred_sec, _ = predict_duration_seconds(model, scaler, columns, feat_row)
    except Exception as e:
        _, dist_km = build_model_features(
            pickup_dt, lat1, lon1, lat2, lon2,
            st.session_state.vendor_id, st.session_state.passenger_count,
        )
        pred_sec      = baseline_seconds(dist_km, pickup_dt, csv_profile)
        used_fallback = True

    pred_min = pred_sec / 60.0
    speed    = dist_km / max(pred_sec / 3600.0, 1e-6)

    # Rush hour / night badge
    is_rush = pickup_dt.hour in [7,8,9,17,18,19]
    is_night = pickup_dt.hour in list(range(22,24)) + list(range(0,6))
    time_badge = ("🔴 Rush Hour" if is_rush else "🌙 Night" if is_night else "🟢 Normal")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Results")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("⏱ Duration",    f"{pred_min:.1f} min")
    m2.metric("📏 Distance",   f"{dist_km:.2f} km")
    m3.metric("🚀 Avg Speed",  f"{speed:.1f} km/h")
    m4.metric("👥 Passengers", str(st.session_state.passenger_count))
    m5.metric("🕐 Time Type",  time_badge)

    dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    st.markdown(
        f'<p style="margin-top:12px;color:#64748b;font-size:0.85rem;">'
        f'🏢 {vendor_options[st.session_state.vendor_id]} &nbsp;|&nbsp; '
        f'🗓 {dow_names[pickup_dt.weekday()]} &nbsp;|&nbsp; '
        f'🕗 {pickup_dt.strftime("%b %d, %Y at %H:%M")}'
        f'</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if used_fallback:
        st.info("ℹ️ Model could not run — showing baseline estimate from training data.")
