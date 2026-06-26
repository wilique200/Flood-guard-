# ============================================================
# FLOODGUARD — STREAMLIT APP (v4 model)
# ============================================================
# Flood Risk Intelligence System
# Multi-task DNN — 5 prediction heads, validated against 31 real
# post-cutoff flood events (see About tab for full methodology).
# Production threshold revised from 0.760 (F1-optimised) to 0.70
# (real-world validated) — see About tab for details.
# Neighbourhood-aware analysis (25-point grid).
# Dual map rendering: Plotly (dashboard) + Folium (street).
# ============================================================

import streamlit as st


def html(markup: str):
    """
    Renders raw HTML via st.markdown, safely.

    Markdown treats ANY line indented 4+ spaces as a code block. Multi-line
    f-strings written with Python-level indentation (for readability in the
    source file) trigger this unintentionally — every line's leading
    whitespace survives into the string. This strips leading whitespace
    from every line before rendering, so visual indentation in the source
    code never leaks into the rendered output.
    """
    flattened = "\n".join(line.lstrip() for line in markup.split("\n"))
    st.markdown(flattened, unsafe_allow_html=True)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium
from streamlit_local_storage import LocalStorage
import json

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FloodGuard — Flood Risk Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLES — dark navy "Crisis Operations Center" aesthetic
# ══════════════════════════════════════════════════════════════
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#080C18; color:#E2E8F0; }
.stApp { background-color:#080C18; }

[data-testid="stSidebar"] { background-color:#0F1623; border-right:1px solid #1E2A40; }
[data-testid="stSidebar"] * { color:#E2E8F0 !important; }

[data-testid="metric-container"] { background:#0F1623; border:1px solid #1E2A40; border-radius:12px; padding:16px; }

.stTextInput > div > div > input, .stSelectbox > div > div {
    background-color:#141B2D !important; border:1px solid #1E2A40 !important;
    color:#E2E8F0 !important; border-radius:8px !important; }

.stButton > button {
    background:linear-gradient(135deg,#0EA5E9,#38BDF8); color:#080C18;
    font-family:'JetBrains Mono',monospace; font-weight:700; border:none;
    border-radius:8px; padding:10px 24px; letter-spacing:0.5px; width:100%; }
.stButton > button:hover { background:linear-gradient(135deg,#38BDF8,#7DD3FC); transform:translateY(-1px); }

.fg-card { background:#0F1623; border:1px solid #1E2A40; border-radius:14px; padding:18px 20px; margin-bottom:16px; }
.fg-card-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:1.5px;
    color:#64748B; text-transform:uppercase; margin-bottom:12px; }

.hero-section { background:linear-gradient(135deg,#0a1628 0%,#0d1f3c 50%,#080C18 100%);
    border:1px solid #1E2A40; border-radius:16px; padding:28px 32px; margin-bottom:20px;
    position:relative; overflow:hidden; }
.hero-section::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,#38BDF8,transparent); }

.ai-card { background:linear-gradient(135deg,#0F1A2E,#111C30); border:1px solid rgba(56,189,248,0.2);
    border-radius:14px; padding:18px 20px; position:relative; overflow:hidden; }
.ai-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(56,189,248,0.5),transparent); }

.env-tile { background:#141B2D; border:1px solid #1E2A40; border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.env-name { font-family:'JetBrains Mono',monospace; font-size:9px; color:#64748B;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.env-value { font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:#E2E8F0; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#080C18; }
::-webkit-scrollbar-thumb { background:#1E2A40; border-radius:2px; }

.stTabs [data-baseweb="tab-list"] { background:#0F1623; border-radius:10px; padding:4px; border:1px solid #1E2A40; }
.stTabs [data-baseweb="tab"] { color:#64748B; font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.5px; }
.stTabs [aria-selected="true"] { background:#141B2D !important; color:#38BDF8 !important; border-radius:8px !important; }

hr { border-color:#1E2A40; }

.validation-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(34,197,94,0.1);
    border:1px solid rgba(34,197,94,0.3); border-radius:6px; padding:4px 10px;
    font-family:'JetBrains Mono',monospace; font-size:10px; color:#22C55E; }
</style>
""")


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# Risk bands re-centred around the v4 production threshold of 0.70
# (revised from the model's original F1-optimised 0.760 after the
# 31-case real-world validation battery — see About tab).
RISK_LEVELS = {
    (0.00,0.20):("SAFE","#22C55E","✅"),
    (0.20,0.40):("WATCH","#EAB308","👁"),
    (0.40,0.70):("WARNING","#F97316","⚠️"),
    (0.70,0.85):("DANGER","#EF4444","🚨"),
    (0.85,1.01):("EXTREME","#A855F7","🆘"),
}

SEV_LABELS  = {0:"None",1:"Low",2:"Medium",3:"High",4:"Extreme"}
SEV_COLORS  = {0:"#38BDF8",1:"#22C55E",2:"#EAB308",3:"#F97316",4:"#A855F7"}
DAYS_BUCKET_LABELS = {
    0:"0-3 days (imminent)", 1:"4-7 days (near-term)",
    2:"8-14 days (medium-term)", 3:"15-30 days (low immediacy)",
}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

GRID_OFFSETS = []
for ring in [0.25, 0.50, 0.75]:
    for dlat, dlon in [(ring,0),(-ring,0),(0,ring),(0,-ring),
                        (ring,ring),(ring,-ring),(-ring,ring),(-ring,-ring)]:
        GRID_OFFSETS.append((round(dlat,2), round(dlon,2)))


# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — must match notebook_03_model_training_v4.py
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim,dim), nn.BatchNorm1d(dim),
        )
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)
    def forward(self, x):
        return self.act(self.drop(x + self.block(x)))


class FloodGuardNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,128,64],
                 n_severity_classes=5, n_days_classes=4, dropout=0.3):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.backbone = nn.Sequential(
            ResidualBlock(hidden_dims[0], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(), nn.Dropout(dropout),
            ResidualBlock(hidden_dims[1], dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]), nn.BatchNorm1d(hidden_dims[2]),
            nn.GELU(), nn.Dropout(dropout*0.7),
        )
        d = hidden_dims[2]
        self.head_flood = nn.Sequential(
            nn.Linear(d,32), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(32,1))
        self.head_severity = nn.Sequential(
            nn.Linear(d,32), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(32,n_severity_classes))
        self.head_days = nn.Sequential(
            nn.Linear(d,32), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(32,n_days_classes))
        self.head_soil = nn.Sequential(
            nn.Linear(d,16), nn.GELU(), nn.Linear(16,1), nn.Sigmoid())
        self.head_discharge = nn.Sequential(
            nn.Linear(d,16), nn.GELU(), nn.Linear(16,1))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        s = self.backbone(x)
        return (
            self.head_flood(s).squeeze(-1),
            self.head_severity(s),
            self.head_days(s),
            self.head_soil(s).squeeze(-1),
            self.head_discharge(s).squeeze(-1),
        )


# ══════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = Path("models/floodguard_best_model_v4.pt")
    if not model_path.exists():
        return None, None, None, None, None
    checkpoint   = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    input_dim    = checkpoint["input_dim"]
    threshold    = checkpoint["threshold"]
    scaler       = checkpoint["scaler"]
    bucket_map   = checkpoint.get("days_bucket_map", DAYS_BUCKET_LABELS)
    model = FloodGuardNet(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, scaler, feature_cols, threshold, bucket_map


# ══════════════════════════════════════════════════════════════
# SAVED LOCATIONS — persisted in the user's BROWSER (localStorage)
# ══════════════════════════════════════════════════════════════
# Streamlit Cloud apps have no server-side database by default, and this
# app has no login/account system. Browser localStorage is the right fit
# for "let me save a few spots and check them again later" on a single
# device, with zero backend infrastructure required — it survives app
# restarts/redeploys because it lives on the USER's device, not ours.
#
# IMPORTANT LIMITATION, stated plainly: this does NOT enable push
# notifications or background monitoring. The app only re-checks saved
# locations when the user actually opens it — there is no way for a
# Streamlit Cloud app to run code or send an alert while nobody has the
# page open. What this DOES give you: every time you open the app, all
# your saved locations are automatically re-evaluated against the live
# model, and you see how each one's risk has changed since your last visit.
#
# KNOWN QUIRK (per the streamlit-local-storage library's own docs/issues):
# getItem() does not return synchronously on the very first script run
# after a fresh page load — the value lands in st.session_state on a
# follow-up rerun. The loader below handles that by treating "not yet
# loaded" as its own distinct state, not as "empty."

LOCAL_STORAGE_KEY = "floodguard_saved_locations"

def get_local_storage():
    if "_local_storage_instance" not in st.session_state:
        st.session_state["_local_storage_instance"] = LocalStorage()
    return st.session_state["_local_storage_instance"]


def _ls_get(localS, key, widget_key):
    """
    Calls LocalStorage.getItem() defensively. The package's own README
    documents getItem(key, key=widget_key) — but live deployment testing
    showed the actually-installed 0.0.25 wheel raises TypeError on that
    kwarg, meaning the published docs and the shipped package disagree.
    Try the documented form first, fall back to the bare call so this
    keeps working regardless of which signature is actually present.
    """
    try:
        return localS.getItem(key, key=widget_key)
    except TypeError:
        return localS.getItem(key)


def _ls_set(localS, key, value, widget_key):
    """Same defensive pattern as _ls_get, for setItem()."""
    try:
        localS.setItem(key, value, key=widget_key)
    except TypeError:
        localS.setItem(key, value)


def load_saved_locations():
    """
    Returns a list of saved-location dicts, or None if the browser
    hasn't returned a value yet (first-run quirk — caller should treat
    None as "still loading," not "empty list").
    """
    localS = get_local_storage()
    raw = _ls_get(localS, LOCAL_STORAGE_KEY, "fg_get_locations")
    if raw is None:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []


def save_locations_to_storage(locations):
    localS = get_local_storage()
    _ls_set(localS, LOCAL_STORAGE_KEY, json.dumps(locations), "fg_set_locations")


def add_saved_location(locations, label, lat, lon):
    locations = list(locations)
    locations.append({
        "label": label, "lat": lat, "lon": lon,
        "last_prob": None, "last_severity": None, "last_checked_at": None,
    })
    save_locations_to_storage(locations)
    return locations


def remove_saved_location(locations, index):
    locations = [l for i, l in enumerate(locations) if i != index]
    save_locations_to_storage(locations)
    return locations


def update_saved_location_result(locations, index, prob, severity):
    locations = list(locations)
    locations[index] = {
        **locations[index],
        "last_prob": prob, "last_severity": severity,
        "last_checked_at": datetime.utcnow().isoformat(),
    }
    return locations


# ══════════════════════════════════════════════════════════════
# GEOCODING & DATA FETCHING
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def geocode(location_str):
    """
    Geocodes via Open-Meteo's own Geocoding API
    (https://geocoding-api.open-meteo.com/v1/search) — the same provider
    already used elsewhere in this app for weather/discharge data, no API
    key required, and no separate restrictive usage policy to navigate.

    Switched away from Nominatim/OpenStreetMap after real-world testing
    showed unreliable behaviour for this app's deployment environment —
    Nominatim's public instance is also explicitly rate-limited to ~1
    request/second and prohibits client-side autocomplete, which made it
    a fragile single point of failure for a deployed app rather than a
    quick prototyping convenience.

    Tries the query as typed first; if that returns nothing, retries
    with just the first comma-separated segment (e.g. "Itakpe, Kogi
    State, Nigeria" -> "Itakpe") since Open-Meteo's matcher works best
    on the place name itself rather than a full free-text address.

    IMPORTANT: returns ALL meaningfully distinct candidates rather than
    silently picking the first result. Place names like "Derby" refer
    to genuinely different real locations (Kansas, Connecticut, New
    York, and the original Derby in England, among others) — for a
    flood-risk tool specifically, silently guessing wrong means showing
    someone irrelevant weather data for a real decision. Two results are
    treated as "the same place" (collapsed to one) only if they're within
    ~0.1° of each other AND share country+admin1 — small metadata/
    naming variants of one true match, not a real ambiguity.
    """
    location_str = location_str.strip()
    url = "https://geocoding-api.open-meteo.com/v1/search"

    candidates_to_try = [location_str]
    first_segment = location_str.split(",")[0].strip()
    if first_segment and first_segment != location_str:
        candidates_to_try.append(first_segment)

    for query in candidates_to_try:
        try:
            params = {"name": query, "count": 8, "format": "json"}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results")
            if not results:
                continue

            distinct = []
            for r in results:
                is_dup = False
                for d in distinct:
                    same_admin = (r.get("country")==d.get("country") and
                                  r.get("admin1")==d.get("admin1"))
                    close = (abs(r["latitude"]-d["latitude"])<0.1 and
                            abs(r["longitude"]-d["longitude"])<0.1)
                    if same_admin and close:
                        is_dup = True
                        break
                if not is_dup:
                    distinct.append(r)

            options = []
            for r in distinct:
                parts = [r.get("name", "")]
                if r.get("admin1"):
                    parts.append(r["admin1"])
                if r.get("country"):
                    parts.append(r["country"])
                options.append({
                    "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                    "display_name": ", ".join(p for p in parts if p),
                })

            if len(options) == 1:
                return {**options[0], "found": True, "ambiguous": False}
            else:
                return {"found": True, "ambiguous": True, "options": options}

        except Exception:
            pass

    return {"found": False}


@st.cache_data(ttl=1800)
def fetch_weather_data(lat, lon):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=35)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(start_date), "end_date": str(end_date),
        "daily": ",".join([
            "precipitation_sum", "rain_sum", "precipitation_hours",
            "temperature_2m_max", "temperature_2m_min",
            "windspeed_10m_max", "et0_fao_evapotranspiration",
        ]),
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm",
        "timezone": "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=1800)
def fetch_discharge_data(lat, lon):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=35)
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat, "longitude": lon, "daily": "river_discharge",
        "start_date": str(start_date), "end_date": str(end_date),
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "daily" in data and "river_discharge" in data["daily"]:
            return data["daily"]["river_discharge"]
    except Exception:
        pass
    return None


def get_elevation(lat, lon):
    try:
        resp = requests.get(
            f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}",
            timeout=8,
        )
        return resp.json()["results"][0]["elevation"]
    except Exception:
        return 50


def is_coastal(lat, lon):
    coastal_zones = [
        (-10,10,-20,20), (5,25,55,75), (-10,25,95,125),
        (25,45,120,145), (25,50,-100,-70), (-35,-10,-55,-30),
    ]
    for lat_min, lat_max, lon_min, lon_max in coastal_zones:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return 1
    return 0


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — mirrors notebook_03_model_training_v4.py
# ══════════════════════════════════════════════════════════════

def build_features(weather_data, discharge_vals, elevation, coastal_flag, feature_cols):
    if "error" in weather_data:
        return None, None

    daily  = weather_data["daily"]
    hourly = weather_data.get("hourly", {})

    rain_mm  = np.nan_to_num(np.array(daily.get("precipitation_sum",[0]*35),dtype=float), nan=0.0)
    rain_h   = np.nan_to_num(np.array(daily.get("precipitation_hours",[0]*35),dtype=float), nan=0.0)
    temp_max = np.array(daily.get("temperature_2m_max",[25]*35), dtype=float)
    temp_min = np.array(daily.get("temperature_2m_min",[15]*35), dtype=float)
    wind_max = np.array(daily.get("windspeed_10m_max",[10]*35), dtype=float)
    et0      = np.nan_to_num(np.array(daily.get("et0_fao_evapotranspiration",[3]*35),dtype=float), nan=3.0)

    sm07  = np.nan_to_num(np.array(hourly.get("soil_moisture_0_to_7cm",[0.3]*35*24),dtype=float), nan=0.3)
    sm728 = np.nan_to_num(np.array(hourly.get("soil_moisture_7_to_28cm",[0.3]*35*24),dtype=float), nan=0.3)

    n_days = len(rain_mm)
    n_hours = len(sm07)
    hrs_per_day = max(1, n_hours // n_days) if n_days > 0 else 24
    sm07_d  = sm07[:n_days*hrs_per_day].reshape(n_days, hrs_per_day).mean(axis=1)
    sm728_d = sm728[:n_days*hrs_per_day].reshape(n_days, hrs_per_day).mean(axis=1)

    FIELD_CAP = 0.40
    if discharge_vals is not None:
        disc = np.nan_to_num(np.array(discharge_vals[-n_days:], dtype=float), nan=0.0)
        for i in range(1, len(disc)):
            if disc[i] == 0 and disc[i-1] > 0:
                disc[i] = disc[i-1]
        disc_available = 1
    else:
        disc = np.zeros(n_days)
        disc_available = 0

    i = n_days - 1
    r3  = float(rain_mm[max(0,i-2):i+1].sum())
    r7  = float(rain_mm[max(0,i-6):i+1].sum())
    r14 = float(rain_mm[max(0,i-13):i+1].sum())
    r30 = float(rain_mm[max(0,i-29):i+1].sum())
    rmax7 = float(rain_mm[max(0,i-6):i+1].max())

    s07  = min(1.0, float(sm07_d[i]) / FIELD_CAP)
    s728 = min(1.0, float(sm728_d[i]) / FIELD_CAP)
    soil_sat = s07*0.4 + s728*0.6
    s07_7  = min(1.0, float(sm07_d[max(0,i-7)]) / FIELD_CAP)
    s728_7 = min(1.0, float(sm728_d[max(0,i-7)]) / FIELD_CAP)
    soil_sat_7ago = s07_7*0.4 + s728_7*0.6
    soil_gradient = s07 - s728
    soil_7d_trend = soil_sat - soil_sat_7ago

    wet = dry = 0
    for j in range(i, -1, -1):
        if rain_mm[j] > 1.0:
            if dry > 0: break
            wet += 1
        else:
            if wet > 0: break
            dry += 1

    disc_today    = float(disc[i])
    disc_7d_mean  = float(disc[max(0,i-6):i+1].mean())
    disc_p99      = float(np.percentile(disc, 99)) if disc.max() > 0 else 1.0
    disc_pct_max  = min(1.0, disc_today/disc_p99) if disc_p99 > 0 else 0.0
    disc_anomaly  = disc_today - disc_7d_mean
    disc_3d_delta = disc_today - float(disc[max(0,i-3)])
    disc_rising   = int(disc_anomaly > 0)

    dates = daily["time"]
    today_dt = datetime.strptime(dates[-1], "%Y-%m-%d")
    month = today_dt.month
    elev = max(1, elevation)
    dry_index = dry / (dry+wet+1)
    intensity_ratio = float(rain_mm[i]) / ((r7/7)+1)

    f = {
        "rain_mm": float(rain_mm[i]), "rain_3d": r3, "rain_7d": r7, "rain_14d": r14,
        "rain_30d": r30, "rain_max7": rmax7, "rain_hours": float(rain_h[i]),
        "intensity_ratio": intensity_ratio, "rain_30d_extreme": int(r30 > 250),
        "rain_7d_extreme": int(r7 > 200),

        "soil_sat": soil_sat, "soil_gradient": soil_gradient, "soil_7d_trend": soil_7d_trend,
        "soil_sat_critical": int(soil_sat > 0.85),

        "disc_pct_max": disc_pct_max*disc_available, "disc_anomaly": disc_anomaly*disc_available,
        "disc_3d_delta": disc_3d_delta*disc_available, "disc_rising": disc_rising*disc_available,
        "discharge_available": disc_available,
        "disc_pct_max_critical": int(disc_pct_max > 0.80)*disc_available,
        "disc_pct_max_x_rain_max7": disc_pct_max*rmax7*disc_available,
        "rain_30d_x_disc_pct_max": r30*disc_pct_max*disc_available,

        "cwi": r7*soil_sat/elev, "rain_soil": r7*soil_sat, "precond": r14*soil_sat/elev,
        "et_def": float(rain_mm[i]) - float(np.nan_to_num(et0[i], nan=3.0)),
        "dry_index": dry_index, "dry_index_high": int(dry_index > 0.60),

        "consec_wet": wet, "consec_dry": dry, "consec_wet_high": int(wet > 7),

        "sin_month": math.sin(2*math.pi*month/12), "cos_month": math.cos(2*math.pi*month/12),
        "month_rain": month*r7,

        "elevation": float(elevation), "is_coastal": float(coastal_flag), "month": float(month),
        "temp_max": float(np.nan_to_num(temp_max[i], nan=25.0)),
        "temp_min": float(np.nan_to_num(temp_min[i], nan=15.0)),
        "wind_max": float(np.nan_to_num(wind_max[i], nan=10.0)),
    }

    vec = np.array([f.get(col, 0.0) for col in feature_cols], dtype=np.float32)

    rain_series = {
        "dates": dates[-7:], "rain": rain_mm[-7:].tolist(), "r7d": r7, "r30d": r30,
        "discharge": disc[-7:].tolist() if disc_available else None,
        "disc_available": disc_available, "soil_sat": soil_sat,
        "temp_max": float(np.nan_to_num(temp_max[i], nan=25.0)),
        "wind_max": float(np.nan_to_num(wind_max[i], nan=10.0)),
        "et_deficit": float(rain_mm[i]) - float(np.nan_to_num(et0[i], nan=3.0)),
        "consec_wet": wet, "month": month,
    }
    return vec, rain_series


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════

def predict(model, scaler, threshold, feature_vec):
    if feature_vec is None:
        return None
    X = scaler.transform(feature_vec.reshape(1, -1))
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        p_flood, p_sev, p_days, p_soil, p_disc = model(X_t)
        flood_prob   = torch.sigmoid(p_flood).item()
        sev_probs    = torch.softmax(p_sev, dim=1).squeeze().numpy()
        days_probs   = torch.softmax(p_days, dim=1).squeeze().numpy()
        soil_sat     = float(p_soil.item())
        disc_anomaly = float(p_disc.item()*500)
    return {
        "flood_prob": flood_prob, "flood_flag": int(flood_prob >= threshold),
        "severity": int(np.argmax(sev_probs)), "sev_probs": sev_probs.tolist(),
        "days_bucket": int(np.argmax(days_probs)), "days_probs": days_probs.tolist(),
        "soil_sat": soil_sat, "disc_anomaly": disc_anomaly, "threshold": threshold,
    }


def get_risk_level(prob):
    for (lo, hi), (label, color, icon) in RISK_LEVELS.items():
        if lo <= prob < hi:
            return label, color, icon
    return "EXTREME", "#A855F7", "🆘"


# ══════════════════════════════════════════════════════════════
# NEIGHBOURHOOD ANALYSIS
# ══════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def get_direction(dlat, dlon):
    if abs(dlat) < 0.05 and abs(dlon) < 0.05:
        return "centre"
    angle = math.degrees(math.atan2(dlon, dlat))
    dirs = ["N","NE","E","SE","S","SW","W","NW","N"]
    return dirs[int((angle+202.5)/45) % 8]


def analyse_neighbourhood(lat, lon, model, scaler, feature_cols, threshold):
    results = []
    all_points = [(0.0, 0.0)] + GRID_OFFSETS
    progress = st.progress(0, text="Analysing neighbourhood grid…")

    for k, (dlat, dlon) in enumerate(all_points):
        pt_lat = round(lat+dlat, 4)
        pt_lon = round(lon+dlon, 4)

        w_data  = fetch_weather_data(pt_lat, pt_lon)
        d_vals  = fetch_discharge_data(pt_lat, pt_lon)
        elev    = get_elevation(pt_lat, pt_lon)
        coastal = is_coastal(pt_lat, pt_lon)

        feat_vec, rain_info = build_features(w_data, d_vals, elev, coastal, feature_cols)
        if feat_vec is not None:
            pred = predict(model, scaler, threshold, feat_vec)
        else:
            pred = {"flood_prob": 0.0, "severity": 0, "flood_flag": 0,
                    "days_bucket": 3, "soil_sat": 0.3}

        label, color, icon = get_risk_level(pred["flood_prob"])
        dist_km   = haversine_km(lat, lon, pt_lat, pt_lon)
        direction = get_direction(dlat, dlon)

        results.append({
            "dlat": dlat, "dlon": dlon, "lat": pt_lat, "lon": pt_lon, "elev": elev,
            "prob": pred["flood_prob"], "severity": pred["severity"],
            "days_bucket": pred["days_bucket"], "soil": pred["soil_sat"],
            "label": label, "color": color, "icon": icon, "dist_km": dist_km,
            "direction": direction, "is_centre": (dlat == 0.0 and dlon == 0.0),
            "rain_info": rain_info,
        })

        progress.progress((k+1)/len(all_points), text=f"Analysing grid point {k+1}/{len(all_points)}…")
        time.sleep(0.05)

    progress.empty()
    return results


def propagation_risk(user_pt, neighbours, threshold=0.55):
    alerts = []
    for nb in neighbours:
        if nb["is_centre"] or nb["prob"] < threshold:
            continue
        elev_diff = nb["elev"] - user_pt["elev"]
        dist_km   = nb["dist_km"]
        if elev_diff > 5 and dist_km < 60:
            factor = round(min(0.35, 0.35*(1-dist_km/60)), 2)
            alerts.append({
                "direction": nb["direction"], "dist_km": round(dist_km,1),
                "prob": nb["prob"], "label": nb["label"],
                "elev_diff": round(elev_diff,0), "factor": factor, "color": nb["color"],
            })
    alerts.sort(key=lambda x: -x["factor"])
    return alerts


# ══════════════════════════════════════════════════════════════
# MAP RENDERING
# ══════════════════════════════════════════════════════════════

def build_plotly_map(grid_results, user_name):
    fig = go.Figure()
    for row in grid_results:
        size = 18 if row["is_centre"] else 12
        symbol = "star" if row["is_centre"] else "square"
        label = (f"<b>{'📍 '+user_name if row['is_centre'] else row['direction'].upper()}</b><br>"
                 f"Flood prob: <b>{row['prob']*100:.1f}%</b><br>Risk: <b>{row['label']}</b><br>"
                 f"Severity: {SEV_LABELS[row['severity']]}<br>Elevation: {row['elev']}m<br>"
                 f"Distance: {row['dist_km']:.1f} km")
        fig.add_trace(go.Scatter(
            x=[row["dlon"]], y=[row["dlat"]], mode="markers",
            marker=dict(size=size, color=row["color"], symbol=symbol,
                       line=dict(color="#0F1623", width=2)),
            text=label, hoverinfo="text", showlegend=False))
    for label, color, icon in [("Safe","#22C55E","✅"),("Watch","#EAB308","👁"),
                                ("Warning","#F97316","⚠️"),("Danger","#EF4444","🚨"),
                                ("Extreme","#A855F7","🆘")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"), name=f"{icon} {label}"))
    fig.update_layout(
        title=dict(text=f"🗺️ Neighbourhood Flood Risk Grid — {user_name}",
                  font=dict(color="#E2E8F0", size=14, family="JetBrains Mono"),
                  y=0.97, yanchor="top"),
        paper_bgcolor="#0F1623", plot_bgcolor="#141B2D",
        font=dict(color="#E2E8F0", family="Inter"),
        xaxis=dict(title="← West | East →", gridcolor="#1E2A40", zeroline=True,
                  zerolinecolor="#38BDF8", zerolinewidth=1.5, tickfont=dict(color="#64748B")),
        yaxis=dict(title="← South | North →", gridcolor="#1E2A40", zeroline=True,
                  zerolinecolor="#38BDF8", zerolinewidth=1.5, tickfont=dict(color="#64748B")),
        legend=dict(bgcolor="#0F1623", bordercolor="#1E2A40", borderwidth=1,
                   font=dict(color="#E2E8F0", size=10), orientation="h",
                   yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=560, margin=dict(l=40, r=20, t=60, b=90))
    return fig


def build_folium_map(grid_results, user_name, user_lat, user_lon):
    m = folium.Map(location=[user_lat, user_lon], zoom_start=9, tiles="CartoDB dark_matter")
    for row in grid_results:
        color_hex = row["color"]
        popup_html = (
            f"<div style='font-family:monospace;background:#0F1623;color:#E2E8F0;"
            f"padding:10px;border-radius:8px;min-width:180px'>"
            f"<b style='color:{color_hex}'>{'📍 '+user_name if row['is_centre'] else row['direction'].upper()}</b><br>"
            f"<hr style='border-color:#1E2A40;margin:6px 0'>"
            f"Flood probability: <b style='color:{color_hex}'>{row['prob']*100:.1f}%</b><br>"
            f"Risk level: <b>{row['label']}</b><br>Severity: {SEV_LABELS[row['severity']]}<br>"
            f"Elevation: {row['elev']}m<br>Distance: {row['dist_km']:.1f} km</div>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=10 if row["is_centre"] else 7,
            color=color_hex, fill=True, fill_color=color_hex,
            fill_opacity=0.8 if row["is_centre"] else 0.65,
            weight=3 if row["is_centre"] else 1.5,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['label']} — {row['prob']*100:.1f}%",
        ).add_to(m)
    folium.Marker(
        location=[user_lat, user_lon], popup=f"📍 {user_name}",
        icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
    ).add_to(m)
    return m


# ══════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════

def render_gauge_svg(prob, color):
    # Single source of truth: one point, computed once, used for BOTH the
    # arc endpoint and the needle tip. The old version computed the needle
    # around the arc's CENTER (90,95) but the arc endpoint around its LEFT
    # EDGE (18,95) — same angle, wrong origin — so they diverged by a
    # constant 72px (the radius) at every probability value.
    cx, cy, r = 90, 95, 72
    angle_deg = -180 + prob * 180   # 0% -> 180° (left baseline) ... 100% -> 0° (right baseline)
    rad = math.radians(angle_deg)
    px = cx + r * math.cos(rad)
    py = cy + r * math.sin(rad)
    large = 1 if prob > 0.5 else 0

    svg_html = f"""<div style="display:flex;justify-content:center;">
<svg viewBox="0 0 180 110" width="200" height="115" overflow="visible">
<defs><linearGradient id="gFill" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="#22C55E"/><stop offset="35%" stop-color="#EAB308"/>
<stop offset="65%" stop-color="#F97316"/><stop offset="100%" stop-color="{color}"/>
</linearGradient></defs>
<path d="M18 95 A72 72 0 0 1 162 95" fill="none" stroke="#1E2A40" stroke-width="13" stroke-linecap="round"/>
<path d="M18 95 A72 72 0 {large} 1 {px:.1f} {py:.1f}" fill="none" stroke="url(#gFill)" stroke-width="13" stroke-linecap="round"/>
<line x1="90" y1="95" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
<circle cx="90" cy="95" r="5" fill="{color}"/>
<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}" opacity="0.9"/>
<text x="90" y="88" text-anchor="middle" font-family="JetBrains Mono" font-size="22" font-weight="700" fill="{color}">{prob*100:.0f}%</text>
</svg></div>"""
    return svg_html


def render_hero(location_name, coords, risk_label, risk_color, risk_icon, prob):
    html(f"""
    <div class="hero-section">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#64748B;
                      letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            FLOODGUARD · LIVE ANALYSIS</div>
          <div style="font-size:28px;font-weight:700;color:#fff;letter-spacing:-0.5px;">{location_name}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748B;margin-top:4px;">
            {coords[0]:.4f}° N · {coords[1]:.4f}° E</div>
        </div>
        <div style="text-align:right;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:42px;font-weight:700;
                      color:{risk_color};line-height:1;text-shadow:0 0 30px {risk_color}88;">
            {prob*100:.0f}%</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{risk_color};
                      letter-spacing:2px;margin-top:2px;">{risk_icon} {risk_label}</div>
        </div>
      </div>
    </div>""")


def render_alert_banner(risk_label, risk_color, location_name, pred, bucket_map):
    days_text = bucket_map.get(pred["days_bucket"], "an uncertain timeframe")
    msgs = {
        "SAFE": f"✅ No flood risk detected for {location_name}. Conditions are stable.",
        "WATCH": f"👁 Elevated conditions in {location_name}. Monitor closely.",
        "WARNING": f"⚠️ Flood risk developing near {location_name}. Estimated window: {days_text}.",
        "DANGER": f"🚨 FLOOD WARNING — {location_name}. High probability event, {days_text}.",
        "EXTREME": f"🆘 EXTREME FLOOD ALERT — {location_name}. Immediate action advised.",
    }
    rc  = risk_color.lstrip("#")
    bg  = f"rgba({int(rc[0:2],16)},{int(rc[2:4],16)},{int(rc[4:6],16)},0.12)"
    html(f"""<div style="background:{bg};border:1px solid {risk_color}44;border-radius:10px;
        padding:12px 18px;margin-bottom:16px;"><span style="color:{risk_color};font-weight:600;">
        {msgs.get(risk_label,'')}</span></div>""")


def render_ai_insight(pred, rain_info, risk_label, risk_color, location_name, bucket_map):
    prob = pred["flood_prob"]
    sev  = SEV_LABELS[pred["severity"]]
    r7   = rain_info["r7d"]
    soil = rain_info["soil_sat"]
    days_text = bucket_map.get(pred["days_bucket"], "an uncertain timeframe")
    disc_a = rain_info["disc_available"]

    if risk_label == "SAFE":
        insight = (f"Conditions in <b>{location_name}</b> are stable. 7-day rainfall is "
                  f"<b>{r7:.1f}mm</b> and soil saturation is at <b>{soil*100:.0f}%</b> — "
                  f"both within safe ranges.")
    elif risk_label == "WATCH":
        insight = (f"Conditions in <b>{location_name}</b> are elevated. Cumulative 7-day "
                  f"rainfall of <b>{r7:.1f}mm</b> and soil saturation of <b>{soil*100:.0f}%</b> "
                  f"warrant monitoring.")
    elif risk_label == "WARNING":
        insight = (f"<b>Flood risk developing</b> near <b>{location_name}</b>. 7-day rainfall "
                  f"of <b>{r7:.1f}mm</b> has raised soil saturation to <b>{soil*100:.0f}%</b>. "
                  f"Model estimates a <b>{days_text}</b> window.")
    elif risk_label == "DANGER":
        insight = (f"<b style='color:{risk_color}'>High flood probability ({prob*100:.0f}%)</b> "
                  f"for <b>{location_name}</b>. Soil saturation is critically high at "
                  f"<b>{soil*100:.0f}%</b> after <b>{r7:.1f}mm</b> of rain in 7 days. "
                  f"Estimated window: <b>{days_text}</b>. "
                  f"{'River discharge is rising. ' if disc_a else ''}Prepare for low-lying areas.")
    else:
        insight = (f"<b style='color:{risk_color}'>CRITICAL FLOOD RISK ({prob*100:.0f}%)</b> — "
                  f"<b>{location_name}</b>. Soil saturation: <b>{soil*100:.0f}%</b>. "
                  f"7-day rain: <b>{r7:.1f}mm</b>. Immediate evacuation of flood-prone "
                  f"zones is strongly advised.")

    chips = [f"Rain 7d: {r7:.0f}mm", f"Soil: {soil*100:.0f}%", f"Window: {days_text}",
            f"Severity: {sev}", "River data ✅" if disc_a else "No river gauge"]
    rc = risk_color.lstrip("#")
    rgba = f"{int(rc[0:2],16)},{int(rc[2:4],16)},{int(rc[4:6],16)}"
    chip_html = "".join([
        f"<span style='background:rgba({rgba},0.12);border:1px solid {risk_color}44;"
        f"color:{risk_color};font-family:JetBrains Mono,monospace;font-size:10px;"
        f"padding:3px 9px;border-radius:4px;margin-right:6px;margin-bottom:4px;"
        f"display:inline-block;'>{c}</span>" for c in chips])

    html(f"""<div class="ai-card">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#38BDF8;
                  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">
        🤖 FloodGuard AI Insight</div>
      <div style="font-size:13.5px;color:#CBD5E1;line-height:1.6;">{insight}</div>
      <div style="margin-top:10px;">{chip_html}</div>
    </div>""")


def render_env_indicators(rain_info, pred):
    tiles = [
        ("7-Day Rain", f"{rain_info['r7d']:.1f}", "mm", min(1,rain_info['r7d']/300),
         "#EF4444" if rain_info['r7d']>150 else "#F97316" if rain_info['r7d']>80 else "#38BDF8"),
        ("Soil Saturation", f"{rain_info['soil_sat']*100:.0f}", "%", rain_info['soil_sat'],
         "#EF4444" if rain_info['soil_sat']>0.85 else "#F97316" if rain_info['soil_sat']>0.65 else "#22C55E"),
        ("Consec Wet Days", f"{rain_info['consec_wet']}", "days", min(1,rain_info['consec_wet']/20),
         "#EF4444" if rain_info['consec_wet']>10 else "#EAB308" if rain_info['consec_wet']>5 else "#38BDF8"),
        ("Flood Probability", f"{pred['flood_prob']*100:.0f}", "%", pred['flood_prob'],
         "#EF4444" if pred['flood_prob']>0.70 else "#F97316" if pred['flood_prob']>0.40 else "#22C55E"),
        ("Max Temp", f"{rain_info['temp_max']:.1f}", "°C", min(1,rain_info['temp_max']/45), "#F97316"),
        ("Wind Max", f"{rain_info['wind_max']:.1f}", "km/h", min(1,rain_info['wind_max']/100), "#38BDF8"),
    ]
    for name, val, unit, bar_pct, bar_color in tiles:
        bar_w = int(bar_pct*100)
        html(f"""<div class="env-tile"><div class="env-name">{name}</div>
            <div class="env-value">{val}<span style="font-size:11px;color:#64748B;
                font-weight:400;margin-left:3px;">{unit}</span></div>
            <div style="height:3px;background:#1E2A40;border-radius:2px;margin-top:7px;">
                <div style="width:{bar_w}%;height:100%;background:{bar_color};
                    border-radius:2px;"></div></div></div>""")


def render_rainfall_chart(rain_info):
    dates = [d[-5:] for d in rain_info["dates"]]
    rain  = rain_info["rain"]
    colors = ["#EF4444" if r>80 else "#F97316" if r>40 else "#EAB308" if r>15 else "#38BDF8" for r in rain]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=rain, marker_color=colors,
                         marker_line=dict(color="#0F1623", width=1),
                         hovertemplate="%{x}: %{y:.1f}mm<extra></extra>"))
    fig.update_layout(
        title=dict(text="7-Day Rainfall (mm)", font=dict(color="#E2E8F0", size=12, family="JetBrains Mono")),
        paper_bgcolor="#0F1623", plot_bgcolor="#141B2D", font=dict(color="#64748B"),
        xaxis=dict(gridcolor="#1E2A40", tickfont=dict(color="#64748B")),
        yaxis=dict(gridcolor="#1E2A40", tickfont=dict(color="#64748B")),
        margin=dict(l=30,r=10,t=40,b=30), height=220)
    st.plotly_chart(fig, use_container_width=True)


def render_severity_chart(pred):
    labels = [SEV_LABELS[i] for i in range(5)]
    probs  = pred["sev_probs"]
    colors = [SEV_COLORS[i] for i in range(5)]
    fig = go.Figure(go.Bar(x=labels, y=[p*100 for p in probs], marker_color=colors,
                           marker_line=dict(color="#0F1623", width=1),
                           hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(
        title=dict(text="Severity Probabilities (%)", font=dict(color="#E2E8F0", size=12, family="JetBrains Mono")),
        paper_bgcolor="#0F1623", plot_bgcolor="#141B2D", font=dict(color="#64748B"),
        xaxis=dict(gridcolor="#1E2A40", tickfont=dict(color="#64748B")),
        yaxis=dict(gridcolor="#1E2A40", tickfont=dict(color="#64748B"), range=[0,105]),
        margin=dict(l=30,r=10,t=40,b=30), height=220)
    st.plotly_chart(fig, use_container_width=True)


def render_days_bucket_chart(pred, bucket_map):
    labels = [bucket_map.get(i, str(i)) for i in range(4)]
    probs  = pred["days_probs"]
    colors = ["#EF4444", "#F97316", "#EAB308", "#38BDF8"]
    fig = go.Figure(go.Bar(x=[p*100 for p in probs], y=labels, orientation="h",
                           marker_color=colors, marker_line=dict(color="#0F1623", width=1),
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(
        title=dict(text="Flood Timing Window (%)", font=dict(color="#E2E8F0", size=12, family="JetBrains Mono")),
        paper_bgcolor="#0F1623", plot_bgcolor="#141B2D", font=dict(color="#64748B"),
        xaxis=dict(gridcolor="#1E2A40", tickfont=dict(color="#64748B"), range=[0,105]),
        yaxis=dict(gridcolor="#1E2A40", tickfont=dict(color="#64748B")),
        margin=dict(l=10,r=10,t=40,b=30), height=220)
    st.plotly_chart(fig, use_container_width=True)


def render_neighbourhood_alerts(user_pt, prop_alerts, grid_results):
    if not prop_alerts:
        html("""<div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
            border-radius:8px;padding:12px 16px;color:#22C55E;font-family:'JetBrains Mono',monospace;
            font-size:12px;">✅ No high-risk zones detected in your neighbourhood.
            Your location appears safe from upstream flood propagation.</div>""")
        return

    for alert in prop_alerts:
        color = alert["color"]
        rc = color.lstrip("#")
        rgba = f"{int(rc[0:2],16)},{int(rc[2:4],16)},{int(rc[4:6],16)}"
        html(f"""<div style="background:rgba({rgba},0.1);border:1px solid {color}44;
            border-radius:8px;padding:12px 16px;margin-bottom:8px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{color};
                font-weight:700;margin-bottom:4px;">⚠️ {alert['label']} zone {alert['dist_km']}km to your {alert['direction'].upper()}</div>
            <div style="font-size:12px;color:#94A3B8;line-height:1.5;">
                Flood probability: <b style="color:{color}">{alert['prob']*100:.1f}%</b> ·
                Elevation: <b>+{alert['elev_diff']}m above you</b><br>
                Water flow toward your location could increase your flood risk
                by an estimated <b style="color:{color}">+{alert['factor']*100:.0f}%</b>.
            </div></div>""")

    high_risk = [r for r in grid_results if not r["is_centre"] and r["prob"] > 0.20]
    high_risk.sort(key=lambda x: -x["prob"])
    if high_risk:
        st.markdown("**Nearby locations by flood risk:**")
        table_rows = ""
        for r in high_risk[:8]:
            rc2 = r['color'].lstrip("#")
            rgba2 = f"{int(rc2[0:2],16)},{int(rc2[2:4],16)},{int(rc2[4:6],16)}"
            table_rows += (
                f"<tr><td style='padding:6px 10px;color:#94A3B8;'>{r['direction'].upper()}</td>"
                f"<td style='padding:6px 10px;color:#94A3B8;'>{r['dist_km']:.0f} km</td>"
                f"<td style='padding:6px 10px;font-family:monospace;color:{r['color']};font-weight:700;'>"
                f"{r['prob']*100:.1f}%</td>"
                f"<td style='padding:6px 10px;'><span style='background:rgba({rgba2},0.15);"
                f"color:{r['color']};font-family:monospace;font-size:10px;padding:2px 8px;"
                f"border-radius:3px;'>{r['label']}</span></td>"
                f"<td style='padding:6px 10px;color:#64748B;'>{r['elev']}m</td></tr>"
            )
        html(f"""<table style="width:100%;border-collapse:collapse;font-size:12px;">
            <thead><tr style="border-bottom:1px solid #1E2A40;">
              <th style="padding:6px 10px;color:#64748B;text-align:left;font-family:monospace;
                  font-size:10px;text-transform:uppercase;">Direction</th>
              <th style="padding:6px 10px;color:#64748B;text-align:left;font-family:monospace;
                  font-size:10px;text-transform:uppercase;">Distance</th>
              <th style="padding:6px 10px;color:#64748B;text-align:left;font-family:monospace;
                  font-size:10px;text-transform:uppercase;">Flood Prob</th>
              <th style="padding:6px 10px;color:#64748B;text-align:left;font-family:monospace;
                  font-size:10px;text-transform:uppercase;">Risk</th>
              <th style="padding:6px 10px;color:#64748B;text-align:left;font-family:monospace;
                  font-size:10px;text-transform:uppercase;">Elevation</th>
            </tr></thead><tbody>{table_rows}</tbody></table>""")


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        html("""
        <div style="text-align:center;padding:16px 0 24px;">
          <div style="font-size:32px;">🌊</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;
                      color:#fff;">Flood<span style="color:#38BDF8;">Guard</span></div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#64748B;
                      letter-spacing:1px;margin-top:4px;">FLOOD RISK INTELLIGENCE</div>
        </div>
        <hr style="border-color:#1E2A40;margin-bottom:20px;">
        """)

        location_input = st.text_input(
            "📍 Enter location", placeholder="e.g. Lagos, Nigeria",
            help="City, town, or region name — add state/country for small towns",
            key="location_input_field")
        button_clicked = st.button("🔍 Analyse Flood Risk")

        # A Streamlit button's return value is True ONLY on the exact run
        # where it was physically clicked — it resets to False on every
        # later rerun, even a programmatic one (e.g. st.rerun() after
        # saving a location). Without this flag, ANY rerun triggered while
        # viewing a searched location would incorrectly bounce back to
        # the landing page, since the check downstream relies on
        # "was the button just clicked" rather than "is there an active
        # location being viewed." This flag tracks the latter, and persists
        # across reruns the same button click can't.
        if button_clicked and location_input.strip():
            st.session_state["_active_location_query"] = location_input.strip()
        run_analysis = button_clicked or (
            "_active_location_query" in st.session_state
            and location_input.strip() == st.session_state["_active_location_query"]
        )

        st.markdown("<hr style='border-color:#1E2A40;margin:20px 0;'>", unsafe_allow_html=True)
        html("""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
            color:#64748B;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            Settings</div>""")

        neighbourhood = st.toggle("🗺️ Neighbourhood Analysis", value=True,
                                   help="Evaluate 25 grid points around your location")
        map_mode = st.radio("Map Style",
            options=["📊 Plotly (Dashboard)", "🗺️ Folium (Street Map)"], index=0,
            help="Switch between dashboard grid view and interactive street map")

        st.markdown("<hr style='border-color:#1E2A40;margin:20px 0;'>", unsafe_allow_html=True)
        html("""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
            color:#64748B;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            Risk Scale</div>""")
        for (lo, hi), (label, color, icon) in RISK_LEVELS.items():
            html(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <div style="width:12px;height:12px;border-radius:2px;background:{color};
                    flex-shrink:0;"></div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#94A3B8;">
                    {icon} {label} ({int(lo*100)}–{int(hi*100)}%)</div></div>""")

        st.markdown("<hr style='border-color:#1E2A40;margin:20px 0;'>", unsafe_allow_html=True)
        html("""<div style="text-align:center;">
            <span class="validation-badge">✅ Validated on 31 real events</span></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#475569;
                text-align:center;line-height:1.6;margin-top:10px;">
            Data: Open-Meteo APIs<br>Model: Multi-task DNN, v4<br>
            140 cities · 2000–2020 training<br>Operating threshold: 0.70</div>""")

    return location_input, run_analysis, neighbourhood, map_mode


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def _evaluate_location(lat, lon, model, scaler, feature_cols, threshold):
    """Runs the full fetch+predict pipeline for one saved location."""
    w_data  = fetch_weather_data(lat, lon)
    d_vals  = fetch_discharge_data(lat, lon)
    elev    = get_elevation(lat, lon)
    coastal = is_coastal(lat, lon)
    feat_vec, rain_info = build_features(w_data, d_vals, elev, coastal, feature_cols)
    if feat_vec is None:
        return None
    return predict(model, scaler, threshold, feat_vec)


def _render_location_card(loc, idx, new_pred, on_remove_key):
    """One saved-location card showing current risk + delta since last visit."""
    risk_label, risk_color, risk_icon = get_risk_level(new_pred["flood_prob"])
    old_prob = loc.get("last_prob")

    if old_prob is not None:
        delta = new_pred["flood_prob"] - old_prob
        if abs(delta) < 0.02:
            delta_text = "No real change since your last visit"
            delta_color = "#64748B"
        elif delta > 0:
            delta_text = f"⬆ Risk rose {delta*100:.0f} points since your last visit"
            delta_color = "#EF4444"
        else:
            delta_text = f"⬇ Risk fell {abs(delta)*100:.0f} points since your last visit"
            delta_color = "#22C55E"
    else:
        delta_text = "First check for this location"
        delta_color = "#64748B"

    rc = risk_color.lstrip("#")
    bg = f"rgba({int(rc[0:2],16)},{int(rc[2:4],16)},{int(rc[4:6],16)},0.08)"

    html(f"""<div style="background:{bg};border:1px solid {risk_color}33;border-radius:12px;
        padding:16px 18px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
          <div>
            <div style="font-size:16px;font-weight:700;color:#fff;">{loc['label']}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#64748B;
                margin-top:2px;">{loc['lat']:.3f}°, {loc['lon']:.3f}°</div>
          </div>
          <div style="text-align:right;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;
                color:{risk_color};">{new_pred['flood_prob']*100:.0f}%</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{risk_color};
                letter-spacing:1px;">{risk_icon} {risk_label}</div>
          </div>
        </div>
        <div style="margin-top:10px;font-size:12px;color:{delta_color};font-weight:600;">
            {delta_text}</div>
    </div>""")

    btn_col1, btn_col2 = st.columns([5,1])
    with btn_col2:
        if st.button("🗑️ Remove", key=on_remove_key):
            return True
    return False


def render_my_locations_landing(saved_locations, model, scaler, feature_cols, threshold, bucket_map):
    """Shown on the landing page (no active search) when saved locations exist."""
    html(f"""<div style="text-align:center;padding:24px 20px 8px;">
      <div style="font-size:40px;margin-bottom:8px;">📍</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
          color:#fff;">Welcome back</div>
      <div style="font-size:14px;color:#64748B;margin-top:4px;">
          Checking {len(saved_locations)} saved location(s) for you…</div>
    </div>""")

    updated_locations = list(saved_locations)
    indices_to_remove = []

    for idx, loc in enumerate(saved_locations):
        with st.spinner(f"Checking {loc['label']}…"):
            new_pred = _evaluate_location(loc["lat"], loc["lon"], model, scaler,
                                          feature_cols, threshold)
        if new_pred is None:
            st.warning(f"⚠️ Couldn't refresh {loc['label']} right now — showing last known data.")
            continue

        removed = _render_location_card(loc, idx, new_pred, on_remove_key=f"landing_remove_{idx}")
        if removed:
            indices_to_remove.append(idx)
        else:
            updated_locations[idx] = update_saved_location_result(
                updated_locations, idx, new_pred["flood_prob"], new_pred["severity"])[idx]

    if indices_to_remove:
        updated_locations = [l for i, l in enumerate(updated_locations)
                             if i not in indices_to_remove]
        save_locations_to_storage(updated_locations)
        st.rerun()
    else:
        save_locations_to_storage(updated_locations)

    st.markdown("---")
    st.caption("🔍 Search a new location above anytime — your saved spots will always be here when you return.")


def render_my_locations_tab(saved_locations, current_already_saved_idx, model, scaler,
                            feature_cols, threshold, bucket_map):
    """The 'My Locations' tab, shown while viewing a searched location."""
    html("""<div class="fg-card-label">Your Saved Locations</div>""")

    if not saved_locations:
        st.info("You haven't saved any locations yet. Use the **📍 Save Location** "
                "button above to start tracking a spot — FloodGuard will re-check it "
                "and show you how its risk has changed every time you open the app.")
        return

    st.caption(f"{len(saved_locations)} saved location(s) — re-checked live against "
              f"current weather data.")

    updated_locations = list(saved_locations)
    indices_to_remove = []

    for idx, loc in enumerate(saved_locations):
        with st.spinner(f"Checking {loc['label']}…"):
            new_pred = _evaluate_location(loc["lat"], loc["lon"], model, scaler,
                                          feature_cols, threshold)
        if new_pred is None:
            st.warning(f"⚠️ Couldn't refresh {loc['label']} right now.")
            continue

        removed = _render_location_card(loc, idx, new_pred, on_remove_key=f"tab_remove_{idx}")
        if removed:
            indices_to_remove.append(idx)
        else:
            updated_locations[idx] = update_saved_location_result(
                updated_locations, idx, new_pred["flood_prob"], new_pred["severity"])[idx]

    if indices_to_remove:
        updated_locations = [l for i, l in enumerate(updated_locations)
                             if i not in indices_to_remove]
        save_locations_to_storage(updated_locations)
        st.rerun()
    else:
        save_locations_to_storage(updated_locations)


def main():
    model, scaler, feature_cols, threshold, bucket_map = load_model()
    if model is None:
        st.error("⚠️ Model file not found. Please upload `models/floodguard_best_model_v4.pt`")
        st.stop()

    location_input, run_analysis, neighbourhood, map_mode = render_sidebar()

    # Saved locations load from the BROWSER on every run — see the
    # first-run "still loading" quirk noted in the storage section above.
    #
    # IMPORTANT: this retry is BOUNDED. An earlier version called
    # st.rerun() unconditionally whenever the value was still None,
    # which caused a genuine infinite loop (blank screen, script stuck
    # "running" forever) on at least one real deployment — if the
    # underlying component never resolves for any reason, there must be
    # a way out, or the app becomes permanently unusable rather than
    # just missing one feature.
    saved_locations = load_saved_locations()
    locations_still_loading = saved_locations is None

    if locations_still_loading:
        retry_count = st.session_state.get("_locations_load_retries", 0)
        if retry_count < 3:
            st.session_state["_locations_load_retries"] = retry_count + 1
            saved_locations = []
            if not run_analysis or not location_input.strip():
                st.info("⏳ Loading your saved locations…")
                st.rerun()
        else:
            # Gave up after 3 attempts — proceed with an empty list rather
            # than hang forever. My Locations simply won't show anything
            # saved this session; the rest of the app still works normally.
            saved_locations = []
    else:
        st.session_state["_locations_load_retries"] = 0

    if not run_analysis or not location_input.strip():
        if saved_locations:
            render_my_locations_landing(saved_locations, model, scaler,
                                        feature_cols, threshold, bucket_map)
        else:
            html("""
            <div style="text-align:center;padding:80px 20px;">
              <div style="font-size:56px;margin-bottom:16px;">🌊</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;
                          color:#fff;margin-bottom:8px;">Flood<span style="color:#38BDF8;">Guard</span></div>
              <div style="font-size:16px;color:#64748B;max-width:480px;margin:0 auto;line-height:1.6;">
                AI-powered flood risk intelligence. Enter any location to get real-time flood
                probability, neighbourhood risk analysis, and early warning insights.
              </div>
              <div style="margin-top:32px;display:flex;gap:16px;justify-content:center;flex-wrap:wrap;">
                <div style="background:#0F1623;border:1px solid #1E2A40;border-radius:10px;
                    padding:16px 20px;font-family:monospace;font-size:12px;color:#64748B;">
                    🎯 5 prediction targets</div>
                <div style="background:#0F1623;border:1px solid #1E2A40;border-radius:10px;
                    padding:16px 20px;font-family:monospace;font-size:12px;color:#64748B;">
                    🗺️ 25-point neighbourhood grid</div>
                <div style="background:#0F1623;border:1px solid #1E2A40;border-radius:10px;
                    padding:16px 20px;font-family:monospace;font-size:12px;color:#64748B;">
                    ✅ Validated on 31 real events</div>
              </div>
            </div>""")
        return

    location_clean = location_input.strip()

    with st.spinner(f"📍 Locating {location_clean}…"):
        geo = geocode(location_clean)

    if not geo["found"]:
        st.error(f"❌ Could not find '{location_clean}'.")
        st.markdown(
            "Try adding a state/region or country (e.g. **'Itakpe, Kogi State, "
            "Nigeria'** instead of just 'Itakpe') — small towns are sometimes "
            "missing from the map database on their own, but resolve "
            "successfully with more context. You can also try the nearest "
            "larger city or LGA capital."
        )
        return

    # DISAMBIGUATION: "Derby" genuinely refers to different real places
    # (Kansas, Connecticut, New York, England, ...) — geocode() now
    # surfaces every distinct match rather than silently picking one.
    # The chosen option is remembered per exact query string, so it
    # survives the rerun that applies the choice, but a different
    # ambiguous search later still asks again rather than reusing a
    # stale pick.
    disambig_key = f"_disambig_choice::{location_clean.lower()}"
    name = None

    if geo.get("ambiguous"):
        chosen = st.session_state.get(disambig_key)
        if chosen is None:
            st.warning(f"⚠️ '{location_clean}' matches more than one place. Which did you mean?")
            for i, opt in enumerate(geo["options"]):
                if st.button(f"📍 {opt['display_name']}", key=f"disambig_opt_{i}"):
                    st.session_state[disambig_key] = i
                    st.rerun()
            return
        else:
            chosen_option = geo["options"][chosen]
            geo = {**chosen_option, "found": True}
            name = chosen_option["display_name"]

    lat, lon = geo["lat"], geo["lon"]
    if name is None:
        name = location_clean.title()

    with st.spinner("📡 Fetching weather & soil data…"):
        w_data  = fetch_weather_data(lat, lon)
        d_vals  = fetch_discharge_data(lat, lon)
        elev    = get_elevation(lat, lon)
        coastal = is_coastal(lat, lon)

    feat_vec, rain_info = build_features(w_data, d_vals, elev, coastal, feature_cols)
    if feat_vec is None:
        st.error("❌ Failed to fetch weather data. Please try again.")
        return

    pred = predict(model, scaler, threshold, feat_vec)
    risk_label, risk_color, risk_icon = get_risk_level(pred["flood_prob"])

    render_hero(name, (lat, lon), risk_label, risk_color, risk_icon, pred["flood_prob"])
    render_alert_banner(risk_label, risk_color, name, pred, bucket_map)

    # Save / update this location's button — sits right under the alert
    # banner so it's visible without scrolling, regardless of which tab
    # the user is on.
    already_saved_idx = next((i for i, l in enumerate(saved_locations)
                              if abs(l["lat"]-lat)<0.01 and abs(l["lon"]-lon)<0.01), None)
    save_col1, save_col2 = st.columns([4, 1])
    with save_col2:
        if already_saved_idx is not None:
            if st.button("📍 Saved ✓", key="already_saved_btn", disabled=True):
                pass
        else:
            if st.button("📍 Save Location", key="save_loc_btn"):
                new_record = {
                    "label": name, "lat": lat, "lon": lon,
                    "last_prob": pred["flood_prob"], "last_severity": pred["severity"],
                    "last_checked_at": datetime.utcnow().isoformat(),
                }
                updated = list(saved_locations) + [new_record]
                save_locations_to_storage(updated)
                st.session_state["_just_saved_location"] = name
                st.rerun()

    if st.session_state.get("_just_saved_location"):
        st.success(f"Saved {st.session_state['_just_saved_location']} to My Locations")
        del st.session_state["_just_saved_location"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "🗺️ Neighbourhood", "📈 Details", "📍 My Locations", "ℹ️ About"])

    # ══════════ TAB 1: DASHBOARD ══════════
    with tab1:
        col_ai, col_gauge = st.columns([3, 1])
        with col_ai:
            render_ai_insight(pred, rain_info, risk_label, risk_color, name, bucket_map)
        with col_gauge:
            html(f"""<div class="fg-card" style="text-align:center;">
                <div class="fg-card-label">Risk Index</div>
                {render_gauge_svg(pred['flood_prob'], risk_color)}
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:{risk_color};letter-spacing:1px;text-transform:uppercase;
                    margin-top:4px;">{risk_icon} {risk_label}</div></div>""")

        col_env, col_charts = st.columns([1, 2])
        with col_env:
            html("""<div class="fg-card-label" style="margin-bottom:8px;">
                Environmental Indicators</div>""")
            render_env_indicators(rain_info, pred)
        with col_charts:
            render_rainfall_chart(rain_info)
            c1, c2 = st.columns(2)
            with c1: render_severity_chart(pred)
            with c2: render_days_bucket_chart(pred, bucket_map)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Flood Probability", f"{pred['flood_prob']*100:.1f}%", delta=f"{risk_label}")
        with m2:
            st.metric("Predicted Severity", SEV_LABELS[pred["severity"]],
                      delta=f"Confidence: {max(pred['sev_probs'])*100:.0f}%")
        with m3:
            st.metric("Timing Window", bucket_map.get(pred["days_bucket"], "—"))
        with m4:
            st.metric("Soil Saturation", f"{pred['soil_sat']*100:.0f}%",
                      delta="critical" if pred['soil_sat']>0.85 else "normal")

    # ══════════ TAB 2: NEIGHBOURHOOD ══════════
    with tab2:
        if not neighbourhood:
            st.info("Enable 'Neighbourhood Analysis' in the sidebar to use this feature.")
        else:
            html(f"""<div class="fg-card-label">Analysing 25 grid points in a
                84km radius around {name}</div>""")

            # Cache the grid in session_state, keyed on location. Without
            # this, switching map style (Plotly <-> Folium) triggers a full
            # Streamlit rerun — st_folium is a bidirectional component and
            # registering/changing its return value causes one — and the
            # entire 25-point live-API neighbourhood fetch would re-run from
            # scratch every time, which looks and feels like the whole app
            # restarting. Caching here means that fetch only happens once
            # per location, and toggling map style just re-renders it.
            grid_cache_key = f"grid_{round(lat,4)}_{round(lon,4)}"
            if st.session_state.get("grid_cache_key") != grid_cache_key:
                with st.spinner("🗺️ Running neighbourhood flood analysis…"):
                    st.session_state["grid_data"] = analyse_neighbourhood(
                        lat, lon, model, scaler, feature_cols, threshold
                    )
                st.session_state["grid_cache_key"] = grid_cache_key

            grid = st.session_state["grid_data"]
            user_pt, neighbours = grid[0], grid[1:]
            prop_alerts = propagation_risk(user_pt, neighbours)
            use_folium = "Folium" in map_mode

            if use_folium:
                html("""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#38BDF8;letter-spacing:1px;margin-bottom:8px;">
                    🗺️ INTERACTIVE STREET MAP — Click markers for details</div>""")
                st_folium(build_folium_map(grid, name, lat, lon), width=None, height=500,
                          key="floodguard_folium_map")
            else:
                html("""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#38BDF8;letter-spacing:1px;margin-bottom:8px;">
                    📊 DASHBOARD GRID — Hover points for details</div>""")
                st.plotly_chart(build_plotly_map(grid, name), use_container_width=True)

            st.markdown("---")
            html("""<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#64748B;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
                ⚡ Flood Propagation Risk Assessment</div>""")
            render_neighbourhood_alerts(user_pt, prop_alerts, grid)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Your Risk", f"{user_pt['prob']*100:.1f}%")
            with c2: st.metric("Highest Nearby", f"{max(r['prob'] for r in neighbours)*100:.1f}%")
            with c3: st.metric("Avg Nearby", f"{np.mean([r['prob'] for r in neighbours])*100:.1f}%")
            with c4: st.metric("High Risk Zones", f"{sum(1 for r in neighbours if r['prob']>0.70)}/24")

    # ══════════ TAB 3: DETAILS ══════════
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📍 Location Details**")
            st.json({"location": name, "latitude": lat, "longitude": lon,
                    "elevation_m": elev, "is_coastal": bool(coastal),
                    "month": MONTH_NAMES[rain_info["month"]-1]})

            st.markdown("**🌊 Discharge Data**")
            if rain_info["disc_available"]:
                st.success("✅ River gauge data available")
                if rain_info["discharge"]:
                    disc_df = pd.DataFrame({"Date": rain_info["dates"],
                        "Discharge": [f"{d:.1f} m³/s" for d in rain_info["discharge"]]})
                    st.dataframe(disc_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No river gauge in this area. Prediction uses rainfall + soil only.")

        with col_b:
            st.markdown("**🧠 Model Predictions**")
            pred_df = pd.DataFrame({
                "Task": ["Flood Probability", "Severity Level", "Timing Window", "Soil Saturation"],
                "Output": [f"{pred['flood_prob']*100:.2f}%",
                          f"{SEV_LABELS[pred['severity']]} ({pred['severity']})",
                          bucket_map.get(pred["days_bucket"], "—"),
                          f"{pred['soil_sat']*100:.1f}%"],
                "Confidence": [f"Threshold: {threshold:.2f}", f"{max(pred['sev_probs'])*100:.1f}%",
                              f"{max(pred['days_probs'])*100:.1f}%", "R²≈0.99 (persistence-beating)"],
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            st.markdown("**📊 Severity Probability Breakdown**")
            sev_df = pd.DataFrame({"Level": [SEV_LABELS[i] for i in range(5)],
                "Probability": [f"{p*100:.2f}%" for p in pred["sev_probs"]]})
            st.dataframe(sev_df, use_container_width=True, hide_index=True)

            st.markdown("**🌧️ 7-Day Rainfall Summary**")
            rain_df = pd.DataFrame({"Date": rain_info["dates"],
                "Rain (mm)": [f"{r:.1f}" for r in rain_info["rain"]]})
            st.dataframe(rain_df, use_container_width=True, hide_index=True)

    # ══════════ TAB 4: MY LOCATIONS ══════════
    with tab4:
        render_my_locations_tab(saved_locations, already_saved_idx, model, scaler,
                                feature_cols, threshold, bucket_map)

    # ══════════ TAB 5: ABOUT ══════════
    with tab5:
        html("""
        <div class="fg-card">
          <div class="fg-card-label">About FloodGuard</div>
          <p style="color:#94A3B8;line-height:1.7;font-size:14px;">
            FloodGuard is an AI-powered flood risk intelligence system trained on 20 years
            of weather, soil, and river discharge data (2000–2020) across 140 cities
            globally, sourced from the Open-Meteo API and labelled using the Dartmouth
            Flood Observatory archive.
          </p>
        </div>

        <div class="fg-card">
          <div class="fg-card-label">Model Architecture</div>
          <table style="width:100%;font-size:13px;color:#94A3B8;border-collapse:collapse;">
            <tr><td style="padding:6px 0;color:#64748B;">Architecture</td>
                <td>Multi-task Deep Neural Network — residual backbone + 5 heads</td></tr>
            <tr><td style="padding:6px 0;color:#64748B;">Training data</td>
                <td>~1,063,000 rows × 40 features, 140 cities, 2000–2020</td></tr>
            <tr><td style="padding:6px 0;color:#64748B;">Validation</td>
                <td>5-fold GroupKFold by city (no city appears in both train and val)</td></tr>
            <tr><td style="padding:6px 0;color:#64748B;">Operating threshold</td>
                <td>0.70 (revised from an initial F1-optimised 0.760 — see below)</td></tr>
          </table>
        </div>

        <div class="fg-card">
          <div class="fg-card-label">🧪 Real-World Validation — Read This Before Trusting a Forecast</div>
          <p style="color:#94A3B8;line-height:1.7;font-size:13.5px;">
            Beyond standard cross-validation, this model was tested against
            <b>31 real, independently documented flood events and dry-season control
            periods</b>, all occurring <b>after the training cutoff of October 2020</b> —
            meaning the model could not have memorised any of these dates. Events span
            14 cities across 6 continents, from Thailand's record-breaking November 2025
            rainfall to calm dry-season days in Sudan and Australia.
          </p>
          <table style="width:100%;font-size:13px;color:#94A3B8;border-collapse:collapse;margin-top:10px;">
            <tr style="border-bottom:1px solid #1E2A40;">
              <td style="padding:8px 0;color:#64748B;font-family:monospace;font-size:11px;
                  text-transform:uppercase;">Event Type</td>
              <td style="padding:8px 0;color:#64748B;font-family:monospace;font-size:11px;
                  text-transform:uppercase;text-align:right;">Accuracy</td>
            </tr>
            <tr style="border-bottom:1px solid #1E2A40;">
              <td style="padding:8px 0;">🆘 Extreme flood events</td>
              <td style="padding:8px 0;text-align:right;color:#22C55E;font-weight:700;">75%</td>
            </tr>
            <tr style="border-bottom:1px solid #1E2A40;">
              <td style="padding:8px 0;">⚠️ Moderate flood events</td>
              <td style="padding:8px 0;text-align:right;color:#F97316;font-weight:700;">~35–45%*</td>
            </tr>
            <tr>
              <td style="padding:8px 0;">✅ Dry-season / no-flood controls</td>
              <td style="padding:8px 0;text-align:right;color:#22C55E;font-weight:700;">100%</td>
            </tr>
          </table>
          <p style="color:#64748B;line-height:1.6;font-size:12px;margin-top:10px;">
            *Improved from 22% at the model's original threshold to this range after
            real-world threshold recalibration. <b>Zero false positives</b> were recorded
            across all 31 cases — the model has never incorrectly flagged a calm period
            as flooding in this test set.
          </p>
        </div>

        <div class="fg-card">
          <div class="fg-card-label">⚠️ Known Limitation — Moderate-Severity Events</div>
          <p style="color:#94A3B8;line-height:1.7;font-size:13.5px;">
            The model is <b>highly reliable for extreme flood events and calm/dry
            conditions</b>, but currently has <b>reduced sensitivity to moderate-severity
            floods</b> — events driven by accumulated saturation, upstream river overflow,
            or localised intense bursts rather than sustained heavy regional rainfall.
            In real-world testing, several genuine moderate floods (e.g. urban flash
            floods in Buenos Aires, slow-building river overflow in Bogotá) were scored
            as lower-probability than the event severity warranted.
          </p>
          <p style="color:#94A3B8;line-height:1.7;font-size:13.5px;">
            <b>What this means in practice:</b> treat a "Watch" or "Warning" rating as
            worth monitoring even though it falls below the "Danger" threshold — the
            model is more likely to under-call a moderate flood than to false-alarm on
            a calm day. A "Safe" rating with very low probability (under 10%) remains
            highly reliable in either direction.
          </p>
        </div>

        <div class="fg-card">
          <div class="fg-card-label">Data Sources</div>
          <ul style="color:#94A3B8;font-size:13px;line-height:1.8;">
            <li>Weather & Soil: <b>Open-Meteo Historical Archive API</b></li>
            <li>River Discharge: <b>Open-Meteo Flood API</b> (GloFAS v4) — screened to
                require ≥95% genuine data coverage per city before inclusion in training</li>
            <li>Flood Labels: <b>Dartmouth Flood Observatory (DFO)</b></li>
            <li>Geocoding: <b>Open-Meteo Geocoding API</b></li>
            <li>Elevation: <b>Open-Elevation API</b></li>
          </ul>
        </div>

        <div class="fg-card">
          <div class="fg-card-label">Other Limitations</div>
          <ul style="color:#94A3B8;font-size:13px;line-height:1.8;">
            <li>Model trained on 140 cities — accuracy may vary for locations very
                different from the training distribution (e.g. extremely high-latitude
                or polar regions were not included).</li>
            <li>Neighbourhood grid uses 0.25° spacing (~28km) — highly localised
                micro-flood events may not be captured.</li>
            <li>River discharge reflects GloFAS modelled data, not direct gauge
                readings, and may differ from official local hydrological services.</li>
            <li>Not a substitute for official emergency services or local
                meteorological warnings.</li>
          </ul>
        </div>
        """)


if __name__ == "__main__":
    main()
