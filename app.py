"""
Zwigato Delivery Delay Predictor + Natural-Language Manager Assistant
---------------------------------------------------------------------
The original ML prediction pipeline is preserved. The LLM is an additive layer.

UI:
1. Natural-language assistant at the TOP: users can describe an order in rough,
   non-technical language.
2. The same existing ML model runs after all required inputs are available.
3. Descriptive + prescriptive knowledge is generated from the model result.
4. The original manual-parameter prediction form remains BELOW the chat.

Model files expected alongside this script:
    linear_regression_model.joblib
    logistic_regression_model.joblib
"""

import datetime as dt
import html
import json
import time
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Zwigato Delivery Delay Predictor", page_icon="🛵", layout="centered", initial_sidebar_state="collapsed")

LATE_THRESHOLD = 30

# =============================================================================
# EXISTING MODEL CODE — PRESERVED
# =============================================================================
@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_regression_model.joblib")
    logistic_model = joblib.load("logistic_regression_model.joblib")
    return linear_model, logistic_model


linear_model, logistic_model = load_models()
FEATURE_ORDER = list(linear_model.feature_names_in_)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def build_feature_row(order: dict) -> pd.DataFrame:
    """Original feature-engineering logic, kept unchanged."""
    distance_km = haversine_distance(
        order["restaurant_lat"], order["restaurant_lon"],
        order["delivery_lat"], order["delivery_lon"],
    )

    order_dt = order["order_datetime"]
    day_of_week = order_dt.weekday()
    month = order_dt.month
    hour = order_dt.hour
    time_of_day = get_time_of_day(hour)

    row = {
        "Delivery_person_Age": order["age"],
        "Delivery_person_Ratings": order["ratings"],
        "Restaurant_latitude": order["restaurant_lat"],
        "Restaurant_longitude": order["restaurant_lon"],
        "Delivery_location_latitude": order["delivery_lat"],
        "Delivery_location_longitude": order["delivery_lon"],
        "Vehicle_condition": order["vehicle_condition"],
        "multiple_deliveries": order["multiple_deliveries"],
        "Order_Preparation_Time": order["prep_time"],
        "Distance_km": distance_km,
    }

    one_hot_blocks = {
        "Weatherconditions": (order["weather"], ["Fog", "NaN", "Sandstorms", "Stormy", "Sunny", "Windy"]),
        "Road_traffic_density": (order["traffic"], ["Jam", "Low", "Medium"]),
        "Type_of_order": (order["order_type"], ["Drinks ", "Meal ", "Snack "]),
        "Type_of_vehicle": (order["vehicle_type"], ["motorcycle ", "scooter "]),
        "Festival": (order["festival"], ["Yes"]),
        "City": (order["city"], ["Semi-Urban", "Urban"]),
        "Time_of_Day": (time_of_day, ["Evening", "Morning", "Night"]),
        "Order_DayOfWeek": (day_of_week, [1, 2, 3, 4, 5, 6]),
        "Order_Month": (month, [3, 4]),
        "Order_Hour": (hour, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
    }

    for prefix, (value, categories) in one_hot_blocks.items():
        for cat in categories:
            col = f"{prefix}_{cat}"
            row[col] = 1 if value == cat else 0

    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=FEATURE_ORDER, fill_value=0)
    return df_row


def run_existing_prediction(order: dict):
    """Same two model calls as the original app."""
    X = build_feature_row(order)
    predicted_minutes = float(linear_model.predict(X)[0])
    late_proba = float(logistic_model.predict_proba(X)[0][1])
    is_late = late_proba >= 0.5
    return X, predicted_minutes, late_proba, is_late


def make_model_context(order, X, predicted_minutes, late_proba, is_late):
    coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
    contributions = (coefs * X.iloc[0]).sort_values(key=np.abs, ascending=False)
    top = contributions[contributions != 0].head(6)

    return {
        "prediction": {
            "predicted_delivery_time_minutes": round(predicted_minutes, 2),
            "late_probability_percent": round(late_proba * 100, 2),
            "late_threshold_minutes": LATE_THRESHOLD,
            "classification": "LIKELY LATE" if is_late else "LIKELY ON TIME",
        },
        "order": {
            "order_date": str(order["order_datetime"].date()),
            "order_time": str(order["order_datetime"].strftime("%H:%M")),
            "kitchen_preparation_time_minutes": float(order["prep_time"]),
            "multiple_deliveries": int(order["multiple_deliveries"]),
            "restaurant_latitude": float(order["restaurant_lat"]),
            "restaurant_longitude": float(order["restaurant_lon"]),
            "delivery_latitude": float(order["delivery_lat"]),
            "delivery_longitude": float(order["delivery_lon"]),
            "rider_age": int(order["age"]),
            "rider_rating": float(order["ratings"]),
            "vehicle_condition": int(order["vehicle_condition"]),
            "weather": str(order["weather"]).strip(),
            "traffic": str(order["traffic"]).strip(),
            "vehicle_type": str(order["vehicle_type"]).strip(),
            "order_type": str(order["order_type"]).strip(),
            "festival": str(order["festival"]).strip(),
            "city": str(order["city"]).strip(),
        },
        "model_drivers": [
            {
                "factor": str(name),
                "effect": "Increases late risk" if value > 0 else "Decreases late risk",
                "contribution_score": round(float(value), 6),
            }
            for name, value in top.items()
        ],
        "note": "Model associations are not causal proof.",
    }


# =============================================================================
# LLM LAYER
# =============================================================================
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def clean_secret(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().strip("`")
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value or None


# The app supports both providers. If Gemini is configured, it is used first;
# otherwise OpenRouter is used. Gemini 2.5 Flash-Lite currently has a free tier.
OPENROUTER_API_KEY = clean_secret(get_secret("OPENROUTER_API_KEY"))

# Use a specific free model instead of openrouter/free for more predictable
# demo behaviour. The free router selects a model at random and can occasionally
# return an empty completion. If the user has the old value in Secrets, we
# transparently replace it with the stable free model below.
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free")
if OPENROUTER_MODEL.strip() == "openrouter/free":
    OPENROUTER_MODEL = "qwen/qwen3.8-27b:free"

# Free models on OpenRouter rotate often (slugs get renamed, removed or rate-limited),
# so instead of hard-coding a list we ask OpenRouter which models are free *right now*.
# The list is cached for an hour. If that lookup fails, a small static list is used.
_STATIC_FREE_FALLBACKS = [
    "qwen/qwen3.8-27b:free",
    "openrouter/free",   # OpenRouter's own router: picks any currently-free model
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_free_openrouter_models(limit: int = 6) -> List[str]:
    """Return IDs of text models that currently cost $0, largest context first."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        r.raise_for_status()
        found = []
        for m in r.json().get("data", []):
            mid = m.get("id", "")
            pricing = m.get("pricing") or {}
            arch = m.get("architecture") or {}
            is_free = str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
            text_in = "text" in (arch.get("input_modalities") or ["text"])
            text_out = (arch.get("output_modalities") or ["text"]) == ["text"]
            if mid.endswith(":free") and is_free and text_in and text_out:
                found.append((m.get("context_length") or 0, mid))
        found.sort(reverse=True)
        return [mid for _, mid in found[:limit]]
    except Exception:
        return []


def _extract_text_from_openrouter_response(data: dict) -> str:
    """Read normal Chat Completions text robustly."""
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    # Some OpenAI-compatible responses may expose content as a list of blocks.
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces).strip()

    return ""


def call_openrouter(
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter directly without changing the existing ML model."""
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY is not configured in Streamlit Secrets."

    api_key = clean_secret(OPENROUTER_API_KEY)
    if not api_key:
        return None, "OPENROUTER_API_KEY is empty."

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "https://streamlit.io/",
        "X-Title": "Zwigato Delivery Delay Predictor",
    }

    # Try the configured model first, then a couple of current free fallbacks.
    models_to_try = []
    for m in [OPENROUTER_MODEL] + get_free_openrouter_models() + _STATIC_FREE_FALLBACKS:
        if m and m not in models_to_try:
            models_to_try.append(m)

    errors = []

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Prevent thinking-only responses for this application. We need the
            # final answer text / JSON, not a reasoning-only completion.
            "reasoning": {"enabled": False},
        }

        response = None
        for attempt in range(1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
            except requests.RequestException as exc:
                errors.append(f"{model_name}: request failed: {exc}")
                response = None
                break
            break
        if response is None:
            continue

        if response.status_code == 401:
            return None, (
                "OpenRouter authentication failed (HTTP 401). Check Streamlit Secrets: "
                "OPENROUTER_API_KEY must contain only the key value, without 'Bearer '."
            )

        if not response.ok:
            try:
                body = response.json()
            except ValueError:
                body = response.text[:1000]
            if response.status_code == 402:
                reason = "no credits on the OpenRouter account"
            elif response.status_code == 429:
                reason = "rate-limited"
            elif response.status_code == 404:
                reason = "model not available under this name"
            else:
                reason = str(body)[:120]
            errors.append(f"{model_name}: HTTP {response.status_code} ({reason})")
            continue

        try:
            data = response.json()
        except ValueError as exc:
            errors.append(f"{model_name}: invalid JSON response: {exc}")
            continue

        text = _extract_text_from_openrouter_response(data)

        if text:
            return text, None

        # Empty completion: try the next free endpoint rather than surfacing
        # an opaque error to the user. OpenRouter documents that empty outputs
        # can occur on provider responses.
        errors.append(f"{model_name}: empty completion")

    return None, (
        "All free OpenRouter models are busy right now. Please wait a minute and ask again. "
        + " | ".join(errors[-3:])
    )


def call_llm(
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Tuple[Optional[str], Optional[str]]:
    """Single LLM provider: OpenRouter."""
    return call_openrouter(system, messages, temperature, max_tokens)


# =============================================================================
# NATURAL-LANGUAGE ORDER EXTRACTION
# =============================================================================
FIELD_HELP = {
    "order_datetime": "order date and exact time",
    "prep_time": "estimated kitchen preparation time in minutes",
    "multiple_deliveries": "number of deliveries on this trip (0-3)",
    "restaurant_lat": "restaurant latitude",
    "restaurant_lon": "restaurant longitude",
    "delivery_lat": "delivery-location latitude",
    "delivery_lon": "delivery-location longitude",
    "age": "rider age",
    "ratings": "rider rating",
    "vehicle_condition": "vehicle condition from 0 to 3",
    "weather": "weather",
    "traffic": "road traffic density",
    "vehicle_type": "vehicle type",
    "order_type": "order type",
    "festival": "whether it is a festival day",
    "city": "city type",
}

CHAT_SYSTEM = """
You are the natural-language input layer for the Zwigato delivery-delay prediction app.
The app already has a trained ML model. You NEVER make the prediction yourself.
Your job is only to understand the user's language and extract the input parameters.

Return ONLY valid JSON with this exact structure:
{
  "intent": "new_prediction" | "follow_up" | "general",
  "order": {
    "order_datetime": "YYYY-MM-DD HH:MM" | null,
    "prep_time": number | null,
    "multiple_deliveries": integer | null,
    "restaurant_lat": number | null,
    "restaurant_lon": number | null,
    "delivery_lat": number | null,
    "delivery_lon": number | null,
    "age": number | null,
    "ratings": number | null,
    "vehicle_condition": integer | null,
    "weather": "Sunny" | "Cloudy" | "Fog" | "Sandstorms" | "Stormy" | "Windy" | "NaN" | null,
    "traffic": "Low" | "Medium" | "High" | "Jam" | null,
    "vehicle_type": "motorcycle " | "scooter " | "electric_scooter " | "bicycle " | null,
    "order_type": "Snack " | "Meal " | "Drinks " | "Buffet " | null,
    "festival": "No" | "Yes" | null,
    "city": "Urban" | "Metropolitian" | "Semi-Urban" | null
  },
  "reply": "brief natural-language response"
}

Mapping rules:
- clear/clear sky -> Sunny; stormy/storm -> Stormy; fog/foggy -> Fog;
  windy -> Windy; cloudy/overcast -> Cloudy; sandstorm -> Sandstorms.
- very low/low traffic -> Low; moderate/medium -> Medium; high -> High;
  jammed/gridlock/traffic jam -> Jam.
- bike/motorbike/motorcycle -> motorcycle; e-scooter/electric scooter -> electric_scooter;
  scooter -> scooter; bicycle/cycle -> bicycle.
- snack/quick bite -> Snack; meal/lunch/dinner -> Meal; drink/beverage -> Drinks; buffet -> Buffet.
- poor vehicle -> 0; fair -> 1; good -> 2; excellent/best -> 3.
- Explicitly stated festival yes/no only; never assume.
- Do not invent any numeric value.
- Do not invent coordinates. The ML model uses the restaurant and delivery coordinates directly.
- Do not invent date or exact time.
""".strip()


def blank_order() -> Dict[str, Any]:
    return {key: None for key in FIELD_HELP.keys()}


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def merge_orders(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(old)
    for key in result:
        value = new.get(key) if isinstance(new, dict) else None
        if value is not None and value != "":
            result[key] = value
    return result


def missing_fields(order: Dict[str, Any]) -> List[str]:
    return [key for key, value in order.items() if value is None or value == ""]


def parse_user_order(message: str) -> Tuple[Optional[dict], Optional[str]]:
    draft = st.session_state.get("chat_draft", blank_order())
    prompt = (
        "Current draft from earlier chat messages:\n"
        + json.dumps(draft, indent=2)
        + "\n\nNew user message:\n"
        + message
        + "\n\nMerge the new information into the current draft and return ONLY JSON."
    )
    content, error = call_llm(
        CHAT_SYSTEM,
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
    )
    if error:
        return None, error
    parsed = extract_json(content)
    if not parsed:
        return None, "I could not understand the details clearly. Please describe the order again in simple language."
    return parsed, None


def convert_chat_order(order: Dict[str, Any]) -> dict:
    order_dt = dt.datetime.strptime(str(order["order_datetime"]), "%Y-%m-%d %H:%M")
    return {
        "order_datetime": order_dt,
        "prep_time": float(order["prep_time"]),
        "multiple_deliveries": int(order["multiple_deliveries"]),
        "restaurant_lat": float(order["restaurant_lat"]),
        "restaurant_lon": float(order["restaurant_lon"]),
        "delivery_lat": float(order["delivery_lat"]),
        "delivery_lon": float(order["delivery_lon"]),
        "age": int(float(order["age"])),
        "ratings": float(order["ratings"]),
        "vehicle_condition": int(order["vehicle_condition"]),
        "weather": str(order["weather"]),
        "traffic": str(order["traffic"]),
        "vehicle_type": str(order["vehicle_type"]),
        "order_type": str(order["order_type"]),
        "festival": str(order["festival"]),
        "city": str(order["city"]),
    }


def validate_chat_order(order: dict):
    if not 0 <= order["prep_time"] <= 60:
        raise ValueError("Kitchen preparation time must be between 0 and 60 minutes.")
    if not 0 <= order["multiple_deliveries"] <= 3:
        raise ValueError("Multiple deliveries must be between 0 and 3.")
    if not 15 <= order["age"] <= 50:
        raise ValueError("Rider age must be between 15 and 50.")
    if not 1 <= order["ratings"] <= 6:
        raise ValueError("Rider rating must be between 1 and 6.")
    if not 0 <= order["vehicle_condition"] <= 3:
        raise ValueError("Vehicle condition must be between 0 and 3.")


def generate_manager_insights(context: dict) -> Tuple[Optional[str], Optional[str]]:
    system = """
You are the Zwigato AI Manager Assistant.
The existing ML model has already made the numerical prediction. Treat the supplied
prediction and factor direction as authoritative.

Write two clear sections:

### Descriptive knowledge
Explain what the model predicts, the late-delivery probability, the 30-minute
business threshold, and the main factors that are associated with higher/lower risk.
Do not claim causation.

### Prescriptive knowledge
Provide practical decision-support actions an operations manager could consider.
Tie each action to the supplied prediction/factors. Mention actions such as reviewing
the delivery promise, rider assignment, multiple-delivery load, preparation-time
bottlenecks, routing/traffic exposure, or closer monitoring only where relevant.
Do not invent real-time information, staffing, capacity, costs, or policies.
Do not say an action guarantees prevention of delay.

End with exactly: "Decision support, not autopilot."
""".strip()
    return call_llm(
        system,
        [{"role": "user", "content": "Model context:\n" + json.dumps(context, indent=2)}],
        temperature=0.15,
        max_tokens=1000,
    )


def generate_followup(question: str, context: dict, history: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    system = (
        "You are the Zwigato AI Manager Assistant answering a question about the CURRENT ML prediction.\n"
        "Use the provided model context as the source of truth. Explain the result and provide practical "
        "managerial decision support. Do not change the ML prediction, invent numbers, or claim causation.\n\n"
        "CURRENT MODEL CONTEXT:\n" + json.dumps(context, indent=2)
    )
    messages = history[-8:] + [{"role": "user", "content": question}]
    return call_llm(system, messages, temperature=0.2, max_tokens=800)


# =============================================================================
# STYLING
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Manrope:wght@400;500;600;700&display=swap');
:root{
  --ink:#221B2E; --muted:#6B6580; --line:#E3E2F0; --canvas:#F4F5FB; --card:#FFFFFF;
  --red:#E23744; --orange:#FF7A3D; --grad:linear-gradient(120deg,#E23744 0%,#F25C3B 55%,#FF8A3D 100%);
  --green:#12805C; --green-soft:#E3F5EC; --amber:#C2410C; --amber-soft:#FFEBDD;
  --shadow:0 8px 30px rgba(34,27,46,.08);
}
.stApp{
  background:
    radial-gradient(760px 380px at 100% -8%, #FFE3DC 0%, transparent 62%),
    radial-gradient(680px 380px at -10% 18%, #E6E4FF 0%, transparent 60%),
    var(--canvas);
  font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--ink);
}
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ max-width:820px; padding-top:5rem !important; padding-bottom:7rem; }
#MainMenu, footer{ visibility:hidden; }

/* Hero banner */
.zw-hero{ position:relative; overflow:hidden; display:flex; gap:18px; align-items:center;
  padding:26px 30px; border-radius:36px; background:var(--grad); color:#fff; margin-bottom:1.6rem;
  box-shadow:0 14px 34px rgba(226,55,68,.28); }
.zw-hero::before{ content:""; position:absolute; right:-50px; top:-70px; width:230px; height:230px; border-radius:50%; background:rgba(255,255,255,.14); }
.zw-hero::after{ content:""; position:absolute; right:110px; bottom:-80px; width:150px; height:150px; border-radius:50%; background:rgba(255,255,255,.10); }
.zw-hero > *{ position:relative; z-index:1; }
.zw-logo{ width:60px; height:60px; border-radius:50%; background:#fff; display:flex; align-items:center; justify-content:center; font-size:30px; flex:none; box-shadow:0 6px 16px rgba(0,0,0,.15); }
.zw-title{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:1.7rem; line-height:1.2; color:#fff; letter-spacing:-0.02em; }
.zw-sub{ color:rgba(255,255,255,.9); font-size:0.97rem; margin-top:5px; }

/* Headings and notes */
.zw-h{ font-family:'Sora','Manrope',sans-serif; font-weight:600; font-size:1.05rem; color:var(--ink); margin:1.5rem 0 0.7rem 0; }
.zw-note{ color:var(--muted); font-size:0.92rem; margin:0.8rem 0 0.4rem 0; }
.zw-center{ text-align:center; }

/* Pill-shaped method switch */
.st-key-picker{ background:#fff; border-radius:999px; padding:6px; box-shadow:var(--shadow); }
.st-key-picker [data-testid="stHorizontalBlock"]{ gap:6px; }
.st-key-picker button{ height:3.1rem; border:none; font-weight:700; }
.st-key-picker button p{ color:inherit !important; font-size:1rem; }
.st-key-picker button[data-testid="stBaseButton-secondary"]{ background:transparent; color:var(--muted); box-shadow:none; }
.st-key-picker button[data-testid="stBaseButton-secondary"]:hover{ background:var(--canvas); color:var(--ink); }
.st-key-picker button[data-testid="stBaseButton-primary"]{ background:var(--grad); color:#fff; box-shadow:0 6px 16px rgba(226,55,68,.35); }

/* Example chips */
.st-key-examples button{ background:#fff; border:1.5px solid var(--line); color:var(--ink); font-weight:600; }
.st-key-examples button:hover{ border-color:var(--red); color:var(--red); }


/* Feature cards (landing) */
.st-key-feat_ai, .st-key-feat_manual{ position:relative; background:#fff; border-radius:32px; padding:26px 24px 22px 24px;
  box-shadow:var(--shadow); border:2px solid transparent; transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
.st-key-feat_ai:hover, .st-key-feat_manual:hover{ transform:translateY(-5px); box-shadow:0 18px 44px rgba(226,55,68,.20); border-color:#F6B5AE; }
.st-key-feat_ai:has(button:focus-visible), .st-key-feat_manual:has(button:focus-visible){ border-color:var(--red); }
.st-key-pickbig_ai, .st-key-pickbig_manual{ position:absolute !important; inset:0; z-index:5; margin:0 !important; }
.st-key-pickbig_ai button, .st-key-pickbig_manual button{ width:100%; height:100%; min-height:100%; opacity:0; cursor:pointer; border-radius:32px; }
.zw-feat-ico{ width:62px; height:62px; border-radius:50%; background:var(--grad); display:flex; align-items:center; justify-content:center; font-size:30px; box-shadow:0 8px 20px rgba(226,55,68,.30); margin-bottom:14px; }
.zw-feat-t{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:1.2rem; color:var(--ink); }
.zw-feat-d{ color:var(--muted); font-size:0.92rem; line-height:1.5; margin:6px 0 12px 0; min-height:4.2em; }
.zw-feat-tags{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
.zw-feat-tags span{ background:#F1EFFB; color:#5B5380; font-size:0.74rem; font-weight:700; padding:4px 12px; border-radius:999px; }
.zw-feat-go{ display:inline-block; background:var(--grad); color:#fff; font-weight:700; font-size:0.92rem; padding:9px 20px; border-radius:999px; }

/* Steps strip */
.zw-steps{ display:flex; justify-content:center; gap:10px 22px; flex-wrap:wrap; margin:22px 0 4px 0; color:var(--muted); font-size:0.88rem; }
.zw-steps b{ display:inline-flex; width:24px; height:24px; border-radius:50%; background:#fff; color:var(--red); align-items:center; justify-content:center; font-size:0.8rem; margin-right:8px; box-shadow:0 2px 8px rgba(34,27,46,.12); }

/* Preset and follow-up chips */
.st-key-presets button, .st-key-followups button{ background:#fff; border:1.5px solid var(--line); color:var(--ink); font-weight:600; }
.st-key-presets button:hover, .st-key-followups button:hover{ border-color:var(--red); color:var(--red); }
.st-key-followups{ margin-top:6px; }

/* Result card with ring */
.zw-result{ display:flex; align-items:center; gap:24px; flex-wrap:wrap; background:#fff; border-radius:28px; padding:18px 26px; box-shadow:var(--shadow); margin:2px 0 14px 0; }
.zw-ring{ position:relative; width:110px; height:110px; border-radius:50%; flex:none; }
.zw-ring-in{ position:absolute; inset:12px; border-radius:50%; background:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.zw-ring-n{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:1.45rem; line-height:1; color:var(--ink); }
.zw-ring-l{ font-size:0.68rem; color:var(--muted); margin-top:3px; }
.zw-k{ color:var(--muted); font-size:0.82rem; margin-bottom:2px; }
.zw-v{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:2.3rem; color:var(--ink); line-height:1.1; }
.zw-v span{ font-size:1rem; font-weight:600; color:var(--muted); margin-left:4px; }
.zw-pill{ display:inline-block; padding:5px 14px; border-radius:999px; font-weight:700; font-size:0.85rem; margin-top:8px; }
.zw-pill.zw-late{ background:var(--amber-soft); color:var(--amber); }
.zw-pill.zw-ok{ background:var(--green-soft); color:var(--green); }
.zw-limit{ color:var(--muted); font-size:0.8rem; margin-left:8px; }

/* Driver chips */
.zw-chips{ display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 6px 0; }
.zw-chip{ padding:6px 14px; border-radius:999px; font-size:0.86rem; font-weight:600; }
.zw-chip.up{ background:var(--amber-soft); color:var(--amber); }
.zw-chip.down{ background:var(--green-soft); color:var(--green); }


/* Feature cards (landing) */
.st-key-feat_ai, .st-key-feat_manual{ position:relative; background:#fff; border-radius:32px; padding:26px 24px 22px 24px;
  box-shadow:var(--shadow); border:2px solid transparent; transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
.st-key-feat_ai:hover, .st-key-feat_manual:hover{ transform:translateY(-5px); box-shadow:0 18px 44px rgba(226,55,68,.20); border-color:#F6B5AE; }
.st-key-feat_ai:has(button:focus-visible), .st-key-feat_manual:has(button:focus-visible){ border-color:var(--red); }
.st-key-pickbig_ai, .st-key-pickbig_manual{ position:absolute !important; inset:0; z-index:5; margin:0 !important; }
.st-key-pickbig_ai button, .st-key-pickbig_manual button{ width:100%; height:100%; min-height:100%; opacity:0; cursor:pointer; border-radius:32px; }
.zw-feat-ico{ width:62px; height:62px; border-radius:50%; background:var(--grad); display:flex; align-items:center; justify-content:center; font-size:30px; box-shadow:0 8px 20px rgba(226,55,68,.30); margin-bottom:14px; }
.zw-feat-t{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:1.2rem; color:var(--ink); }
.zw-feat-d{ color:var(--muted); font-size:0.92rem; line-height:1.5; margin:6px 0 12px 0; min-height:4.2em; }
.zw-feat-tags{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
.zw-feat-tags span{ background:#F1EFFB; color:#5B5380; font-size:0.74rem; font-weight:700; padding:4px 12px; border-radius:999px; }
.zw-feat-go{ display:inline-block; background:var(--grad); color:#fff; font-weight:700; font-size:0.92rem; padding:9px 20px; border-radius:999px; }

/* Steps strip */
.zw-steps{ display:flex; justify-content:center; gap:10px 22px; flex-wrap:wrap; margin:22px 0 4px 0; color:var(--muted); font-size:0.88rem; }
.zw-steps b{ display:inline-flex; width:24px; height:24px; border-radius:50%; background:#fff; color:var(--red); align-items:center; justify-content:center; font-size:0.8rem; margin-right:8px; box-shadow:0 2px 8px rgba(34,27,46,.12); }

/* Preset and follow-up chips */
.st-key-presets button, .st-key-followups button{ background:#fff; border:1.5px solid var(--line); color:var(--ink); font-weight:600; }
.st-key-presets button:hover, .st-key-followups button:hover{ border-color:var(--red); color:var(--red); }
.st-key-followups{ margin-top:6px; }

/* Result card with ring */
.zw-result{ display:flex; align-items:center; gap:24px; flex-wrap:wrap; background:#fff; border-radius:28px; padding:18px 26px; box-shadow:var(--shadow); margin:2px 0 14px 0; }
.zw-ring{ position:relative; width:110px; height:110px; border-radius:50%; flex:none; }
.zw-ring-in{ position:absolute; inset:12px; border-radius:50%; background:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.zw-ring-n{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:1.45rem; line-height:1; color:var(--ink); }
.zw-ring-l{ font-size:0.68rem; color:var(--muted); margin-top:3px; }
.zw-k{ color:var(--muted); font-size:0.82rem; margin-bottom:2px; }
.zw-v{ font-family:'Sora','Manrope',sans-serif; font-weight:700; font-size:2.3rem; color:var(--ink); line-height:1.1; }
.zw-v span{ font-size:1rem; font-weight:600; color:var(--muted); margin-left:4px; }
.zw-pill{ display:inline-block; padding:5px 14px; border-radius:999px; font-weight:700; font-size:0.85rem; margin-top:8px; }
.zw-pill.zw-late{ background:var(--amber-soft); color:var(--amber); }
.zw-pill.zw-ok{ background:var(--green-soft); color:var(--green); }
.zw-limit{ color:var(--muted); font-size:0.8rem; margin-left:8px; }

/* Driver chips */
.zw-chips{ display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 6px 0; }
.zw-chip{ padding:6px 14px; border-radius:999px; font-size:0.86rem; font-weight:600; }
.zw-chip.up{ background:var(--amber-soft); color:var(--amber); }
.zw-chip.down{ background:var(--green-soft); color:var(--green); }


/* Recommendations */
.zw-rec{ background:#fff; border-radius:28px; padding:20px 24px 10px 24px; box-shadow:var(--shadow); margin:0 0 14px 0; border-left:8px solid var(--line); }
.zw-rec-high{ border-left-color:var(--amber); } .zw-rec-watch{ border-left-color:#E9A23B; } .zw-rec-low{ border-left-color:var(--green); }
.zw-rec-h{ font-family:'Sora','Manrope',sans-serif; font-weight:600; font-size:1.02rem; color:var(--ink); }
.zw-rec-s{ color:var(--muted); font-size:0.9rem; margin:2px 0 12px 0; }
.zw-act{ display:flex; gap:14px; align-items:flex-start; padding:10px 0; border-top:1px solid #F0EFF7; }
.zw-act-ico{ width:40px; height:40px; border-radius:50%; background:var(--canvas); display:flex; align-items:center; justify-content:center; font-size:1.15rem; flex:none; }
.zw-act-t{ font-weight:700; font-size:0.96rem; color:var(--ink); }
.zw-act-d{ color:var(--muted); font-size:0.88rem; margin-top:2px; line-height:1.45; }
.zw-tag{ display:inline-block; margin-left:10px; padding:2px 10px; border-radius:999px; font-size:0.7rem; font-weight:700; vertical-align:middle; }
.zw-tag.now{ background:var(--amber-soft); color:var(--amber); }
.zw-tag.ok{ background:var(--green-soft); color:var(--green); }
.zw-tag.maybe{ background:#ECEAF8; color:#5B5380; }

/* Manual form groups */
[class*="st-key-grp_"]{ background:#fff; border:none !important; border-radius:26px !important; box-shadow:var(--shadow); padding:6px 8px; }
.zw-grp{ font-family:'Sora','Manrope',sans-serif; font-weight:600; font-size:0.97rem; margin-bottom:4px; color:var(--ink); }

/* Chat */
[data-testid="stChatMessage"]{ background:#fff; border-radius:26px; box-shadow:var(--shadow); padding:16px 20px; margin-bottom:12px; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{ border-radius:50% !important; }
[data-testid="stChatInput"]{ border-radius:999px; box-shadow:var(--shadow); }

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button{ font-weight:600; }
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"]{ background:var(--grad); border:none; color:#fff; }
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover{ filter:brightness(1.06); }
button:focus-visible, a:focus-visible{ outline:2px solid var(--red) !important; outline-offset:2px; }
textarea:focus-visible, input:focus-visible{ outline:none !important; }

.zw-foot{ text-align:center; color:var(--muted); font-size:0.78rem; margin-top:2.4rem; }
@media (max-width:640px){ .zw-title{ font-size:1.35rem; } .zw-hero{ padding:20px; border-radius:28px; } .zw-result{ gap:16px; } }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================
_defaults = {
    "mode": None,               # None | "ai" | "manual"
    "chat_messages": [],        # [{"role", "content", "result"}]
    "chat_draft": blank_order(),
    "last_context": None,
    "last_ai_insights": None,
    "last_ai_error": None,
    "manual_result": None,      # stored so the result survives reruns
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# =============================================================================
# SMALL UI HELPERS
# =============================================================================
def set_mode(mode: str) -> None:
    st.session_state.mode = mode


def clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.chat_draft = blank_order()


def use_example(text: str) -> None:
    st.session_state.pending_prompt = text


def build_recommendations(context: dict) -> dict:
    """Rule-based manager actions. Works without any AI service."""
    o = context["order"]
    p = context["prediction"]
    prob = p["late_probability_percent"]
    minutes = p["predicted_delivery_time_minutes"]

    dist = float(haversine_distance(
        o["restaurant_latitude"], o["restaurant_longitude"],
        o["delivery_latitude"], o["delivery_longitude"],
    ))
    hour = int(o["order_time"].split(":")[0])
    peak = 12 <= hour < 14 or 18 <= hour < 22

    if prob >= 60 or minutes > LATE_THRESHOLD:
        tier, headline = "high", "Act before dispatch: this order is likely to run late."
    elif prob >= 30 or minutes >= LATE_THRESHOLD - 6:
        tier, headline = "watch", "Keep an eye on this one: it is close to the limit."
    else:
        tier, headline = "low", "Looks fine. Standard handling is enough."

    acts = []  # (priority 1=do now / 2=consider, icon, title, detail)

    others = int(o["multiple_deliveries"])
    if others >= 2:
        acts.append((1, "📦", "Lighten the rider's load",
                     f"He is already carrying {others} other deliveries. Move one of them to a nearby rider if you can."))
    elif others == 1 and tier != "low":
        acts.append((2, "📦", "Check the batching",
                     "One other drop is on this trip. Make sure this customer is not the last stop."))

    if o["traffic"] in ("High", "Jam"):
        acts.append((1, "🚦", "Plan around the traffic",
                     "Send the rider on the clearest route and dispatch as soon as the food is nearly ready."))
    if o["weather"] in ("Stormy", "Fog", "Sandstorms", "Windy"):
        acts.append((1, "🌧️", "Allow extra time for the weather",
                     f"{o['weather']} conditions slow riders down. Add a buffer to the ETA and remind the rider to ride safely."))

    prep = float(o["kitchen_preparation_time_minutes"])
    if prep >= 20:
        acts.append((1 if tier == "high" else 2, "🍳", "Call the kitchen",
                     f"Preparation is about {prep:.0f} minutes. Ask the restaurant to prioritise this order, "
                     "and time the pickup so the rider is not left waiting."))

    if dist >= 8:
        acts.append((1 if tier == "high" else 2, "📍", "Long trip",
                     f"The drop is about {dist:.1f} km away. A rider closer to the restaurant, or an honest ETA, will help."))

    if float(o["rider_rating"]) < 4.0 or int(o["vehicle_condition"]) <= 1:
        acts.append((2, "🛵", "Consider a stronger rider or vehicle",
                     "This rider's rating or vehicle condition is on the low side for a risky delivery."))
    if o["vehicle_type"] == "bicycle" and dist > 4:
        acts.append((2, "🚲", "Bicycle on a longer route",
                     f"At {dist:.1f} km a motorbike or scooter would be quicker."))

    if o["festival"] == "Yes":
        acts.append((2, "🎉", "Festival demand",
                     "Restaurants and roads are busier on festival days. Expect pickup delays."))
    if peak and tier != "low":
        acts.append((2, "⏰", "Peak-hour pressure",
                     "Order volume is high at this time of day. Keep an eye on the rider's progress."))

    if tier == "high":
        acts.append((1, "💬", "Tell the customer early",
                     f"Send a revised ETA now rather than after the {LATE_THRESHOLD}-minute mark has passed."))
        acts.append((2, "🎁", "Plan a goodwill gesture",
                     "If the order does arrive late, a small credit protects the relationship."))
    elif tier == "watch":
        acts.append((2, "👀", "Re-check in a few minutes",
                     "If pickup slips or traffic worsens, step in early with the actions above."))
    else:
        extras = len(acts)
        acts.append((0, "✅", "No intervention needed",
                     "Dispatch as normal and let the order run."))
        if extras == 0:
            acts.append((2, "📈", "Keep monitoring",
                         "Conditions can change. Check again if the rider is held up at pickup."))
        else:
            headline = "Looks fine. Standard handling is enough, with a few things to keep in mind."

    acts.sort(key=lambda a: a[0])
    return {
        "tier": tier,
        "headline": headline,
        "actions": [{"priority": a[0], "icon": a[1], "title": a[2], "detail": a[3]} for a in acts[:6]],
    }


TAGS = {0: ("ok", "All clear"), 1: ("now", "Do now"), 2: ("maybe", "Consider")}


def render_recommendations(rec: dict) -> None:
    items = "\n".join(f"- **{a['title']}**: {a['detail']}" for a in rec["actions"])
    st.markdown(f"**Recommended actions**\n\n{items}")


def render_tiles(minutes: float, prob_pct: float, is_late: bool) -> None:
    color = "#C2410C" if is_late else "#12805C"
    cls = "zw-late" if is_late else "zw-ok"
    status = "Likely late" if is_late else "Likely on time"
    pct = max(0.0, min(100.0, prob_pct))
    st.markdown(
        '<div class="zw-result">'
        f'<div class="zw-ring" style="background:conic-gradient({color} {pct:.1f}%, #ECEAF5 0)">'
        f'<div class="zw-ring-in"><div class="zw-ring-n">{prob_pct:.0f}%</div>'
        '<div class="zw-ring-l">late risk</div></div></div>'
        '<div><div class="zw-k">Predicted delivery time</div>'
        f'<div class="zw-v">{minutes:.0f}<span>min</span></div>'
        f'<span class="zw-pill {cls}">{status}</span>'
        f'<span class="zw-limit">limit {LATE_THRESHOLD} min</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def history_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Strip UI-only keys and re-add the numbers so follow-up questions have context."""
    out = []
    for m in messages:
        text = m.get("content", "")
        r = m.get("result")
        if r:
            verdict = "likely late" if r["late"] else "likely on time"
            text = (
                f"Predicted delivery time: {r['minutes']:.0f} minutes; "
                f"late probability {r['prob']:.0f}%; {verdict}.\n\n" + text
            )
        out.append({"role": m["role"], "content": text})
    return out


def show_ai_problem(headline: str, detail: Optional[str]) -> None:
    st.warning(headline)
    if detail:
        with st.expander("Technical details"):
            st.code(detail)


# =============================================================================
# HEADER + METHOD PICKER
# =============================================================================
st.markdown(
    '<div class="zw-hero"><div class="zw-logo">🛵</div><div>'
    '<div class="zw-title">Zwigato Delivery Delay Predictor</div>'
    '<div class="zw-sub">Check whether an order is likely to arrive late, and what the manager can do about it.</div>'
    "</div></div>",
    unsafe_allow_html=True,
)

if st.session_state.mode is None:
    st.markdown('<div class="zw-h">How would you like to enter the order?</div>', unsafe_allow_html=True)
    _feats = [
        ("ai", "💬", "Describe it in words",
         "Type or paste the order like a message. The AI assistant picks out the details and asks for anything missing.",
         ["Fastest", "Uses AI"], "Start typing →"),
        ("manual", "🧮", "Fill in the form",
         "Pick areas, rider and conditions from simple dropdowns. Presets get you started in one tap.",
         ["Full control", "No AI needed"], "Open the form →"),
    ]
    for _col, (_key, _ico, _title, _desc, _tags, _go) in zip(st.columns(2), _feats):
        with _col:
            with st.container(key=f"feat_{_key}"):
                st.markdown(
                    f'<div class="zw-feat-ico">{_ico}</div><div class="zw-feat-t">{_title}</div>'
                    f'<div class="zw-feat-d">{_desc}</div>'
                    f'<div class="zw-feat-tags">{"".join(f"<span>{t}</span>" for t in _tags)}</div>'
                    f'<div class="zw-feat-go">{_go}</div>',
                    unsafe_allow_html=True,
                )
                st.button(f"Choose: {_title}", key=f"pickbig_{_key}", on_click=set_mode, args=(_key,))
    st.markdown(
        '<div class="zw-steps"><span><b>1</b>Enter the order</span><span><b>2</b>See the delay prediction</span>'
        '<span><b>3</b>Get recommended actions</span></div>'
        f'<div class="zw-note zw-center">An order counts as late when delivery takes more than {LATE_THRESHOLD} minutes.</div>',
        unsafe_allow_html=True,
    )
else:
    with st.container(key="picker"):
        _pc1, _pc2 = st.columns(2)
        for _col, (_key, _label) in zip(
            (_pc1, _pc2),
            [("ai", "💬  Describe it in words"), ("manual", "🧮  Fill in the form")],
        ):
            with _col:
                st.button(
                    _label,
                    key=f"pick_{_key}",
                    on_click=set_mode,
                    args=(_key,),
                    type="primary" if st.session_state.mode == _key else "secondary",
                    use_container_width=True,
                )


# =============================================================================
# MODE 1 — AI ASSISTANT
# =============================================================================
EXAMPLE_ORDERS = [
    "Rider is 30 with a 4.8 rating, bike in good shape, two other deliveries. Evening rush, heavy traffic and raining. "
    "Kitchen needs 20 minutes. Restaurant at 12.9716, 77.5946 and customer at 13.0500, 77.6500. "
    "Normal meal order in an urban area, no festival.",
    "Rider is 24, rated 4.2, scooter in average condition, no other deliveries. Clear weather, low traffic, "
    "snack order, prep time 10 minutes. Restaurant 12.9716, 77.5946, customer 12.9850, 77.6100. "
    "Metropolitan city, not a festival day.",
]


FOLLOWUPS = [
    ("Why is it risky?", "Why is this order risky? Explain the main reasons briefly."),
    ("What first?", "What is the single most important thing the manager should do first?"),
    ("Message customer", "Write a short, polite message to the customer about this delivery."),
]


def process_chat_message(chat_text: str, history: List[Dict[str, str]]) -> Tuple[str, Optional[dict]]:
    """Runs inside an assistant chat bubble. Returns (text to store, result dict or None)."""
    with st.spinner("Reading the order details..."):
        parsed, parse_error = parse_user_order(chat_text)

    if parse_error:
        show_ai_problem(
            "The AI service isn't responding right now. Try again in a minute, or use the form instead.",
            parse_error,
        )
        return "The AI service wasn't available for this message.", None

    intent = parsed.get("intent", "new_prediction")
    reply = parsed.get("reply", "")

    if intent == "follow_up" and st.session_state.last_context:
        answer, follow_error = generate_followup(chat_text, st.session_state.last_context, history)
        if follow_error:
            show_ai_problem("I couldn't write a reply just now. Please try again.", follow_error)
            return "I couldn't write a reply for that question.", None
        st.markdown(answer)
        return answer, None

    if intent == "general" and st.session_state.last_context is None:
        answer = reply or "Describe a delivery order and I'll assess its delay risk."
        st.markdown(answer)
        return answer, None

    st.session_state.chat_draft = merge_orders(st.session_state.chat_draft, parsed.get("order", {}) or {})
    draft = st.session_state.chat_draft
    missing = missing_fields(draft)

    if missing:
        friendly = reply or "I've captured some of the order details."
        names = [FIELD_HELP[f] for f in missing]
        ask_for = ", ".join(names) if len(names) <= 4 else ", ".join(names[:4]) + f", and {len(names) - 4} more"
        answer = f"{friendly}\n\nI still need **{ask_for}** before I can run the prediction."
        st.markdown(answer)
        return answer, None

    try:
        order = convert_chat_order(draft)
        validate_chat_order(order)
        X, predicted_minutes, late_proba, is_late = run_existing_prediction(order)
        context = make_model_context(order, X, predicted_minutes, late_proba, is_late)
        st.session_state.last_context = context
        st.session_state.chat_draft = blank_order()
    except Exception as exc:
        msg = f"I understood the message, but the model couldn't run: {exc}"
        st.error(msg)
        return msg, None

    rec = build_recommendations(context)
    result = {"minutes": predicted_minutes, "prob": late_proba * 100, "late": bool(is_late), "rec": rec}
    render_tiles(result["minutes"], result["prob"], result["late"])
    render_recommendations(rec)

    with st.spinner("Writing manager advice..."):
        insights, insight_error = generate_manager_insights(context)
    st.session_state.last_ai_insights = insights
    st.session_state.last_ai_error = insight_error

    if insight_error:
        show_ai_problem(
            "The prediction is ready, but the written advice couldn't be generated right now.",
            insight_error,
        )
        return "", result
    st.markdown(insights)
    return insights or "", result


def render_ai_mode() -> None:
    # chat_input clears itself after sending, so the old text never lingers in the box.
    prompt = st.chat_input("Describe the order: rider, traffic, weather, locations, kitchen time...")
    if not prompt:
        prompt = st.session_state.pop("pending_prompt", None)

    messages = st.session_state.chat_messages

    if not messages and not prompt:
        st.markdown(
            '<div class="zw-note zw-center">Mention the rider (age, rating), vehicle, traffic, weather, kitchen time '
            "and where the restaurant and customer are. Type below, or try an example:</div>",
            unsafe_allow_html=True,
        )
        with st.container(key="examples"):
            ex_cols = st.columns(2)
            for i, (col, label) in enumerate(zip(ex_cols, ["🌧️  Rainy evening rush", "☀️  Calm afternoon"])):
                with col:
                    st.button(label, key=f"ex_{i}", on_click=use_example, args=(EXAMPLE_ORDERS[i],),
                              use_container_width=True)
    else:
        top_l, top_r = st.columns([4, 1])
        with top_l:
            st.markdown('<div class="zw-h" style="margin-top:0.6rem">Conversation</div>', unsafe_allow_html=True)
        with top_r:
            st.button("Start over", key="clear_chat", on_click=clear_chat, use_container_width=True)

    for m in messages:
        with st.chat_message(m["role"], avatar="🛵" if m["role"] == "assistant" else None):
            if m.get("result"):
                r = m["result"]
                render_tiles(r["minutes"], r["prob"], r["late"])
                if r.get("rec"):
                    render_recommendations(r["rec"])
            if m.get("content"):
                st.markdown(m["content"])

    if prompt:
        history = history_for_llm(messages)
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant", avatar="🛵"):
            answer, result = process_chat_message(prompt, history)
        messages.append({"role": "assistant", "content": answer, "result": result})

    if st.session_state.last_context and st.session_state.chat_messages:
        with st.container(key="followups"):
            st.caption("Ask a follow-up:")
            fcols = st.columns(3)
            for i, (col, (label, text)) in enumerate(zip(fcols, FOLLOWUPS)):
                with col:
                    st.button(label, key=f"fu_{i}", on_click=use_example, args=(text,), use_container_width=True)


# =============================================================================
# MODE 2 — MANUAL FORM
# =============================================================================
AREAS = {
    "MG Road": (12.9756, 77.6068),
    "Indiranagar": (12.9784, 77.6408),
    "Koramangala": (12.9352, 77.6245),
    "HSR Layout": (12.9116, 77.6474),
    "Jayanagar": (12.9250, 77.5938),
    "Whitefield": (12.9698, 77.7500),
    "Electronic City": (12.8452, 77.6602),
    "Hebbal": (13.0358, 77.5970),
    "Malleshwaram": (13.0031, 77.5643),
    "Yelahanka": (13.1007, 77.5963),
}
CUSTOM_AREA = "Custom location"
AREA_OPTIONS = list(AREAS) + [CUSTOM_AREA]

FORM_DEFAULTS = {
    "f_date": None,  # filled with today's date at reset time
    "f_time": dt.time(19, 0),
    "f_prep": 15.0, "f_multi": 1,
    "f_rest_area": "MG Road", "f_cust_area": "Hebbal",
    "f_age": 30, "f_vehicle": "motorcycle ", "f_rating": 4.7, "f_vcond": 1,
    "f_weather": "Sunny", "f_traffic": "Medium", "f_city": "Metropolitian",
    "f_otype": "Snack ", "f_festival": "No",
}

PRESETS = {
    "🌧️ Rainy rush": {
        "f_time": dt.time(19, 30), "f_prep": 20.0, "f_multi": 2,
        "f_rest_area": "MG Road", "f_cust_area": "Whitefield",
        "f_age": 30, "f_vehicle": "motorcycle ", "f_rating": 4.8, "f_vcond": 2,
        "f_weather": "Stormy", "f_traffic": "High", "f_city": "Metropolitian",
        "f_otype": "Meal ", "f_festival": "No",
    },
    "☀️ Calm afternoon": {
        "f_time": dt.time(15, 0), "f_prep": 10.0, "f_multi": 0,
        "f_rest_area": "Indiranagar", "f_cust_area": "Koramangala",
        "f_age": 26, "f_vehicle": "scooter ", "f_rating": 4.6, "f_vcond": 3,
        "f_weather": "Sunny", "f_traffic": "Low", "f_city": "Metropolitian",
        "f_otype": "Snack ", "f_festival": "No",
    },
    "🎉 Festival night": {
        "f_time": dt.time(20, 30), "f_prep": 25.0, "f_multi": 3,
        "f_rest_area": "Jayanagar", "f_cust_area": "Hebbal",
        "f_age": 24, "f_vehicle": "motorcycle ", "f_rating": 4.3, "f_vcond": 1,
        "f_weather": "Cloudy", "f_traffic": "Jam", "f_city": "Urban",
        "f_otype": "Buffet ", "f_festival": "Yes",
    },
}


def _sync_coords(which: str) -> None:
    area = st.session_state[f"f_{which}_area"]
    if area in AREAS:
        st.session_state[f"f_{which}_lat"], st.session_state[f"f_{which}_lon"] = AREAS[area]


def reset_form() -> None:
    for k, v in FORM_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["f_date"] = dt.date.today()
    _sync_coords("rest")
    _sync_coords("cust")


def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
    _sync_coords("rest")
    _sync_coords("cust")


def on_area_change(which: str) -> None:
    _sync_coords(which)


def on_coord_change(which: str) -> None:
    st.session_state[f"f_{which}_area"] = CUSTOM_AREA


def build_summary_text(context: dict, rec: dict) -> str:
    p, o = context["prediction"], context["order"]
    lines = [
        "Zwigato delivery check",
        "=" * 24,
        f"Predicted delivery time: {p['predicted_delivery_time_minutes']:.0f} min (limit {LATE_THRESHOLD} min)",
        f"Chance of being late: {p['late_probability_percent']:.0f}%",
        f"Verdict: {p['classification'].title()}",
        "",
        rec["headline"],
        "",
        "Recommended actions:",
    ]
    for a in rec["actions"]:
        lines.append(f"- [{TAGS[a['priority']][1]}] {a['title']}: {a['detail']}")
    lines += [
        "",
        "Order details:",
        f"- Date and time: {o['order_date']} {o['order_time']}",
        f"- Kitchen prep: {o['kitchen_preparation_time_minutes']:.0f} min; other deliveries: {o['multiple_deliveries']}",
        f"- Rider: age {o['rider_age']}, rating {o['rider_rating']}, {o['vehicle_type']} (condition {o['vehicle_condition']}/3)",
        f"- Conditions: weather {o['weather']}, traffic {o['traffic']}, {o['order_type']} order, festival {o['festival']}",
    ]
    return "\n".join(lines)


def render_manual_mode() -> None:
    if "f_prep" not in st.session_state:
        reset_form()

    st.markdown('<div class="zw-h">Order details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="zw-note" style="margin-top:-0.3rem">Start from a typical scenario, then change anything you like.</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="presets"):
        pcols = st.columns(4)
        for col, name in zip(pcols, PRESETS):
            with col:
                st.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), use_container_width=True)
        with pcols[3]:
            st.button("↺ Reset", key="reset_form_btn", on_click=reset_form, use_container_width=True)

    with st.container(border=True, key="grp_order"):
        st.markdown('<div class="zw-grp">Order</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            order_date = st.date_input("Order date", key="f_date")
            order_time = st.time_input("Order time", key="f_time")
        with c2:
            prep_time = st.number_input(
                "Kitchen preparation time (minutes)", min_value=0.0, max_value=60.0, step=1.0, key="f_prep",
                help="Expected time between the order being placed and the rider picking it up.",
            )
            multiple_deliveries = st.selectbox("Other deliveries on this trip", [0, 1, 2, 3], key="f_multi")

    with st.container(border=True, key="grp_locations"):
        st.markdown('<div class="zw-grp">Locations</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.selectbox("Restaurant area", AREA_OPTIONS, key="f_rest_area", on_change=on_area_change, args=("rest",))
        with c4:
            st.selectbox("Customer area", AREA_OPTIONS, key="f_cust_area", on_change=on_area_change, args=("cust",))
        with st.expander("Exact coordinates (optional)"):
            e1, e2 = st.columns(2)
            with e1:
                restaurant_lat = st.number_input("Restaurant latitude", format="%.6f", key="f_rest_lat",
                                                 on_change=on_coord_change, args=("rest",))
                restaurant_lon = st.number_input("Restaurant longitude", format="%.6f", key="f_rest_lon",
                                                 on_change=on_coord_change, args=("rest",))
            with e2:
                delivery_lat = st.number_input("Customer latitude", format="%.6f", key="f_cust_lat",
                                               on_change=on_coord_change, args=("cust",))
                delivery_lon = st.number_input("Customer longitude", format="%.6f", key="f_cust_lon",
                                               on_change=on_coord_change, args=("cust",))
        _km = float(haversine_distance(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon))
        st.caption(f"Straight-line distance: about {_km:.1f} km. Areas are approximate Bengaluru centres; "
                   "use exact coordinates for anywhere else.")

    with st.container(border=True, key="grp_rider"):
        st.markdown('<div class="zw-grp">Rider and vehicle</div>', unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        with c5:
            age = st.number_input("Rider age", min_value=15, max_value=50, key="f_age")
            vehicle_type = st.selectbox(
                "Vehicle type", ["motorcycle ", "scooter ", "electric_scooter ", "bicycle "], key="f_vehicle",
                format_func=lambda v: v.strip().replace("_", " "),
            )
        with c6:
            ratings = st.number_input("Rider rating", min_value=1.0, max_value=6.0, step=0.1, key="f_rating")
        with c7:
            vehicle_condition = st.selectbox("Vehicle condition (0 = poor, 3 = best)", [0, 1, 2, 3], key="f_vcond")

    with st.container(border=True, key="grp_conditions"):
        st.markdown('<div class="zw-grp">Conditions</div>', unsafe_allow_html=True)
        c8, c9 = st.columns(2)
        with c8:
            weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "NaN"], key="f_weather")
            traffic = st.selectbox("Road traffic density", ["Low", "Medium", "High", "Jam"], key="f_traffic")
            city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"], key="f_city")
        with c9:
            order_type = st.selectbox("Type of order", ["Snack ", "Meal ", "Drinks ", "Buffet "], key="f_otype",
                                      format_func=lambda v: v.strip())
            festival = st.selectbox("Festival day", ["No", "Yes"], key="f_festival")

    submitted = st.button("Predict delivery outcome", key="predict_btn", type="primary", use_container_width=True)

    if submitted:
        order = {
            "order_datetime": dt.datetime.combine(order_date, order_time),
            "prep_time": prep_time,
            "multiple_deliveries": multiple_deliveries,
            "restaurant_lat": restaurant_lat,
            "restaurant_lon": restaurant_lon,
            "delivery_lat": delivery_lat,
            "delivery_lon": delivery_lon,
            "age": age,
            "ratings": ratings,
            "vehicle_condition": vehicle_condition,
            "weather": weather,
            "traffic": traffic,
            "vehicle_type": vehicle_type,
            "order_type": order_type,
            "festival": festival,
            "city": city,
        }
        X = build_feature_row(order)
        predicted_minutes = float(linear_model.predict(X)[0])
        late_proba = float(logistic_model.predict_proba(X)[0][1])
        is_late = late_proba >= 0.5

        context = make_model_context(order, X, predicted_minutes, late_proba, is_late)
        st.session_state.last_context = context

        coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
        contributions = (coefs * X.iloc[0]).sort_values(key=np.abs, ascending=False)
        top = contributions[contributions != 0].head(6)

        insights, ai_error = None, None
        if OPENROUTER_API_KEY:
            with st.spinner("Writing manager advice..."):
                insights, ai_error = generate_manager_insights(context)
            st.session_state.last_ai_insights = insights
            st.session_state.last_ai_error = ai_error

        st.session_state.manual_result = {
            "minutes": predicted_minutes,
            "prob": late_proba * 100,
            "late": bool(is_late),
            "drivers": [(name, "Increases risk" if v > 0 else "Decreases risk") for name, v in top.items()],
            "features": X.T.rename(columns={0: "value"}),
            "insights": insights,
            "ai_error": ai_error,
            "has_key": bool(OPENROUTER_API_KEY),
            "rec": build_recommendations(context),
            "summary": build_summary_text(context, build_recommendations(context)),
        }

    res = st.session_state.manual_result
    if res:
        st.markdown('<div class="zw-h">Result</div>', unsafe_allow_html=True)
        render_tiles(res["minutes"], res["prob"], res["late"])
        render_recommendations(res["rec"])
        st.download_button("⬇  Download summary", data=res["summary"], file_name="zwigato_order_summary.txt",
                           mime="text/plain", key="dl_summary")
        st.markdown('<div class="zw-h">What is driving this prediction</div>', unsafe_allow_html=True)
        if res["drivers"]:
            chips = "".join(
                f'<span class="zw-chip {"up" if eff == "Increases risk" else "down"}">'
                f'{"▲" if eff == "Increases risk" else "▼"} {html.escape(str(name).replace("_", " ").capitalize())}</span>'
                for name, eff in res["drivers"]
            )
            st.markdown(f'<div class="zw-chips">{chips}</div>', unsafe_allow_html=True)
            st.caption("▲ pushes the late-delivery risk up, ▼ pushes it down. Biggest effects first.")
        else:
            st.caption("No single input stands out strongly for this order.")

        with st.expander("See the exact values sent to the models"):
            st.dataframe(res["features"])

        st.markdown('<div class="zw-h">AI note for the manager</div>', unsafe_allow_html=True)
        if not res["has_key"]:
            st.info("Add an OPENROUTER_API_KEY in Streamlit Secrets to get written advice with each prediction.")
        elif res["ai_error"]:
            st.info("The prediction is ready, but the written advice couldn't be generated. Submit again in a minute.")
        elif res["insights"]:
            st.markdown(res["insights"])


# =============================================================================
# ROUTER
# =============================================================================
if st.session_state.mode == "ai":
    render_ai_mode()
elif st.session_state.mode == "manual":
    render_manual_mode()

st.markdown(
    '<div class="zw-foot">Linear regression estimates delivery minutes; logistic regression estimates the chance of '
    f"a late delivery, defined as taking more than {LATE_THRESHOLD} minutes (a business rule for this case study, not a fixed SLA). "
    "The ML models make the prediction; the AI layer only explains it.</div>",
    unsafe_allow_html=True,
)
