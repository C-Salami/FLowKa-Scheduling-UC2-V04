import os
import json
import re
from datetime import timedelta, datetime

import pytz
from dateutil import parser as dtp
import streamlit as st
import pandas as pd
import altair as alt
from streamlit_mic_recorder import mic_recorder
from nlp_extractor import ai_extract_intent


# ============================ PAGE & SECRETS ============================

TZ = pytz.timezone("Europe/Paris")
st.set_page_config(page_title="Scooter Wheels Scheduler", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY is not set. Voice/NLP will not work.")


# ============================ HELPERS ============================

def parse_datetime_safe(x, default_tz=TZ):
    """
    Robust datetime parser that:
    - works with both naive & timezone-aware strings
    - parses Excel-style "2025-01-01 08:00:00" and ISO dates
    - always returns timezone-aware datetime in default_tz
    """
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return default_tz.localize(x)
        return x.astimezone(default_tz)
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            d = dtp.parse(x)
            if d.tzinfo is None:
                d = default_tz.localize(d)
            else:
                d = d.astimezone(default_tz)
            return d
        except Exception:
            return None
    return None


def minutes_to_timedelta(mins):
    try:
        return timedelta(minutes=float(mins))
    except Exception:
        return timedelta(0)


def normalize_order_references(cmd_text: str) -> str:
    """
    Normalize user phrases like "order 1", "Order #2", "ORD-3" into a canonical
    form "order 1" that the extractor can rely on, but don't over-engineer.
    """
    if not cmd_text:
        return ""

    text = cmd_text

    # Replace "this order" / "that order" with a special token
    text = re.sub(r"\bthis order\b", "order THIS", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthat order\b", "order THIS", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis\b", "THIS", text, flags=re.IGNORECASE)

    # unify "order #12" → "order 12"
    text = re.sub(
        r"\border\s*#\s*(\d+)\b",
        r"order \1",
        text,
        flags=re.IGNORECASE,
    )

    # unify "ORD-12" → "order 12"
    text = re.sub(
        r"\bORD-(\d+)\b",
        r"order \1",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ============================ LOAD DATA & BUILD SCHEDULE ============================

@st.cache_data
def load_and_generate_data():
    """
    Load a scooter wheels dataset and build a simple multi-operation, multi-machine
    schedule DataFrame with:
      - order_id
      - operation (Mixing, Transfer, Filling, QC)
      - machine_name
      - start, end, duration_min
      - wheel_type, qty, due_date, priority
    """

    csv_path = "data/scooter_orders.csv"
    if not os.path.exists(csv_path):
        # Minimal fallback dataset
        raw_data = [
            {
                "order_id": f"ORD-{i+1}",
                "wheel_type": "Standard" if i % 2 == 0 else "Offroad",
                "qty": 500 + i * 100,
                "due_date": (datetime(2025, 1, 1, 8, 0) + timedelta(days=i)).strftime(
                    "%Y-%m-%d 08:00"
                ),
                "priority": 1 if i < 3 else 2,
            }
            for i in range(8)
        ]
        orders_df = pd.DataFrame(raw_data)
    else:
        orders_df = pd.read_csv(csv_path)

    # Normalize columns
    if "order_id" not in orders_df.columns:
        orders_df["order_id"] = [f"ORD-{i+1}" for i in range(len(orders_df))]

    if "wheel_type" not in orders_df.columns:
        orders_df["wheel_type"] = "Standard"

    if "qty" not in orders_df.columns:
        orders_df["qty"] = 1000

    if "priority" not in orders_df.columns:
        orders_df["priority"] = 2

    # Parse due dates
    orders_df["due_date_dt"] = orders_df.get("due_date", "").apply(
        lambda x: parse_datetime_safe(x, TZ)
    )

    # Basic operations + machine mapping
    operations = [
        ("Mix", "Mixing/Processing", 30),
        ("Transfer", "Transfer/Holding", 15),
        ("Fill", "Filling/Capping", 45),
        ("QC", "Finishing/QC", 20),
    ]

    schedule_rows = []
    start_cursor = datetime(2025, 1, 1, 8, 0, tzinfo=TZ)

    for _, row in orders_df.sort_values(by=["priority", "order_id"]).iterrows():
        order_id = row["order_id"]
        wheel_type = row["wheel_type"]
        qty = row["qty"]
        due_dt = row["due_date_dt"] or (start_cursor + timedelta(days=1))

        cur_start = start_cursor
        for op_name, machine_name, base_dur in operations:
            dur = base_dur + (qty / 1000.0) * 10
            op_start = cur_start
            op_end = op_start + timedelta(minutes=dur)

            schedule_rows.append(
                {
                    "order_id": order_id,
                    "wheel_type": wheel_type,
                    "qty": qty,
                    "operation": op_name,
                    "machine_name": machine_name,
                    "start": op_start,
                    "end": op_end,
                    "duration_min": dur,
                    "due_date": due_dt,
                    "priority": row["priority"],
                }
            )
            cur_start = op_end

        # Stagger next order a bit
        start_cursor = start_cursor + timedelta(minutes=30)
        if cur_start > start_cursor:
            start_cursor = cur_start

    schedule_df = pd.DataFrame(schedule_rows)
    return orders_df, schedule_df


orders, base_schedule = load_and_generate_data()

# ============================ SESSION STATE ============================

if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = base_schedule.copy()

if "filters_visible" not in st.session_state:
    st.session_state.filters_visible = True
if "filt_max_orders" not in st.session_state:
    st.session_state.filt_max_orders = 20
if "filt_products" not in st.session_state:
    st.session_state.filt_products = []
if "filt_machines" not in st.session_state:
    st.session_state.filt_machines = []
if "cmd_log" not in st.session_state:
    st.session_state.cmd_log = []
if "color_mode" not in st.session_state:
    st.session_state.color_mode = "Product"
if "last_audio_fp" not in st.session_state:
    st.session_state.last_audio_fp = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = None
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""
if "last_processed_cmd" not in st.session_state:
    st.session_state.last_processed_cmd = None

if "selected_order_id" not in st.session_state:
    st.session_state.selected_order_id = None


# ============================ SIDEBAR / HEADER ============================

sidebar_display = "block" if st.session_state.filters_visible else "none"
sidebar_css = f"""
<style>
[data-testid="stSidebar"] {{
    display: {sidebar_display};
}}
</style>
"""
st.markdown(sidebar_css, unsafe_allow_html=True)

top_left, top_right = st.columns([0.8, 0.2])
with top_left:
    st.markdown("#### 🛞 Scooter Wheels Production Schedule")

with top_right:
    if st.button(
        "Hide filters" if st.session_state.filters_visible else "Show filters",
        key="toggle_filters",
    ):
        st.session_state.filters_visible = not st.session_state.filters_visible
        st.rerun()


# ============================ FILTERS ============================

with st.sidebar:
    st.markdown("### Filters")

    # Max orders
    st.session_state.filt_max_orders = st.slider(
        "Max orders",
        min_value=5,
        max_value=50,
        value=st.session_state.filt_max_orders,
        step=1,
        key="max_orders_slider",
    )

    # Product filter
    all_products = sorted(orders["wheel_type"].unique().tolist())
    st.session_state.filt_products = st.multiselect(
        "Products",
        options=all_products,
        default=st.session_state.filt_products or all_products,
        key="product_ms",
    )

    # Machine filter
    all_machines = sorted(base_schedule["machine_name"].unique().tolist())
    st.session_state.filt_machines = st.multiselect(
        "Machines",
        options=all_machines,
        default=st.session_state.filt_machines,
        key="machine_ms",
    )

    color_options = ["Order", "Product", "Machine", "Operation"]
    st.session_state.color_mode = st.selectbox(
        "Color by",
        color_options,
        index=color_options.index(st.session_state.color_mode)
        if st.session_state.color_mode in color_options
        else 1,
        key="color_mode_sb",
    )

    if st.button("Reset", key="reset_filters"):
        st.session_state.filt_max_orders = 20
        st.session_state.filt_products = all_products
        st.session_state.filt_machines = []
        st.session_state.color_mode = "Product"
        st.session_state.selected_order_id = None


# ============================ FILTER SCHEDULE ============================

sched = st.session_state.schedule_df.copy()

# Filter by product
if st.session_state.filt_products:
    sched = sched[sched["wheel_type"].isin(st.session_state.filt_products)]

# Filter by machines
if st.session_state.filt_machines:
    sched = sched[sched["machine_name"].isin(st.session_state.filt_machines)]

# Limit number of orders
order_ids = (
    sched[["order_id"]]
    .drop_duplicates()
    .sort_values(by="order_id")
    .head(st.session_state.filt_max_orders)["order_id"]
    .tolist()
)
sched = sched[sched["order_id"].isin(order_ids)]

# Ensure TZ-aware datetime
sched["start"] = sched["start"].apply(lambda x: parse_datetime_safe(x, TZ))
sched["end"] = sched["end"].apply(lambda x: parse_datetime_safe(x, TZ))
sched["due_date"] = sched["due_date"].apply(lambda x: parse_datetime_safe(x, TZ))

# ============================ GANTT CHART ============================

machine_order = [
    "Mixing/Processing",
    "Transfer/Holding",
    "Filling/Capping",
    "Finishing/QC",
]

color_mode = st.session_state.color_mode

# Build a color palette for orders
unique_orders = sched["order_id"].unique().tolist()
color_palette = [
    "#3498db",
    "#e74c3c",
    "#2ecc71",
    "#9b59b6",
    "#f1c40f",
    "#e67e22",
    "#1abc9c",
    "#34495e",
]

order_color_map = {
    oid: color_palette[i % len(color_palette)]
    for i, oid in enumerate(unique_orders)
}
sched["order_color"] = sched["order_id"].map(order_color_map)

select_order = alt.selection_point(
    name="order_select",
    fields=["order_id"],
    on="click",
    clear="dblclick",
)

if color_mode == "Order":
    color_encoding = alt.condition(
        select_order,
        alt.Color("order_color:N", scale=None, legend=None),
        alt.value("#e0e0e0"),
    )
elif color_mode == "Product":
    product_domain = sorted(sched["wheel_type"].unique().tolist())
    product_palette = ["#8e44ad", "#e74c3c", "#3498db", "#27ae60", "#f39c12"]
    product_palette = product_palette[: len(product_domain)]
    color_encoding = alt.condition(
        select_order,
        alt.Color(
            "wheel_type:N",
            scale=alt.Scale(domain=product_domain, range=product_palette),
            legend=None,
        ),
        alt.value("#e0e0e0"),
    )
elif color_mode == "Machine":
    machine_domain = machine_order
    machine_palette = ["#16a085", "#2980b9", "#8e44ad", "#c0392b"]
    color_encoding = alt.condition(
        select_order,
        alt.Color(
            "machine_name:N",
            scale=alt.Scale(domain=machine_domain, range=machine_palette),
            legend=None,
        ),
        alt.value("#e0e0e0"),
    )
else:
    field_map = {
        "Product": "wheel_type",
        "Machine": "machine_name",
        "Operation": "operation",
    }
    actual_field = field_map.get(color_mode, "order_id")
    color_encoding = alt.condition(
        select_order,
        alt.Color(actual_field + ":N", legend=None),
        alt.value("#e0e0e0"),
    )

base_enc = {
    "y": alt.Y(
        "machine_name:N",
        sort=machine_order,
        title=None,
        axis=alt.Axis(labelLimit=200),
    ),
    "x": alt.X("start:T", title=None, axis=alt.Axis(format="%b %d %H:%M")),
    "x2": "end:T",
}

bars = (
    alt.Chart(sched)
    .mark_bar(cornerRadius=2)
    .encode(
        color=color_encoding,
        opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.3)),
        tooltip=[
            alt.Tooltip("order_id:N", title="Order"),
            alt.Tooltip("operation:N", title="Op"),
            alt.Tooltip("machine_name:N", title="Machine"),
            alt.Tooltip("wheel_type:N", title="Product"),
            alt.Tooltip("qty:Q", title="Qty"),
            alt.Tooltip("start:T", title="Start", format="%b %d %H:%M"),
            alt.Tooltip("end:T", title="End", format="%b %d %H:%M"),
            alt.Tooltip("due_date:T", title="Due", format="%b %d %H:%M"),
        ],
    )
)

labels = (
    alt.Chart(sched)
    .mark_text(align="left", dx=4, baseline="middle", fontSize=9, color="white")
    .encode(
        text="order_id:N",
        opacity=alt.condition(select_order, alt.value(1.0), alt.value(0.7)),
    )
)

gantt = (
    alt.layer(bars, labels, data=sched)
    .encode(**base_enc)
    .add_params(select_order)
    .properties(width="container", height=350)
    .configure_view(stroke=None)
)

event = st.altair_chart(
    gantt,
    use_container_width=True,
    on_select="rerun",
    key="gantt_chart",
)

if event and "selection" in event and "order_select" in event["selection"]:
    sel = event["selection"]["order_select"]
    selected_id = None

    # Best-effort decoding of the selection payload
    if isinstance(sel, dict):
        vals = sel.get("values") or []
        if vals:
            v = vals[-1]
            if isinstance(v, dict):
                selected_id = v.get("order_id")
            elif isinstance(v, str):
                selected_id = v

    if selected_id:
        st.session_state.selected_order_id = selected_id

if st.session_state.selected_order_id:
    st.caption(f"Selected order on chart: **{st.session_state.selected_order_id}**")


# ============================ DEEPGRAM TRANSCRIPTION =========================

def _deepgram_transcribe_bytes(wav_bytes: bytes,
                               api_key: str,
                               model: str = "nova-2-general"):
    """
    Minimal Deepgram transcription via REST. Expects a WAV/PCM16 short clip.
    """
    import requests

    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    params = {
        "model": model,
        "smart_format": "true",
        "punctuate": "true",
    }
    resp = requests.post(url, headers=headers, params=params, data=wav_bytes, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception:
        return ""


# ============================ NLP INTENT HANDLING ============================

def extract_intent(cmd_text: str) -> dict:
    """
    Thin wrapper calling external NLP (OpenAI / custom) to extract intent.
    Returns a Python dict with keys such as:
      - intent: "delay_order" | "swap_orders" | ...
      - order_id
      - other_order_id
      - delta_minutes (positive or negative)
    """
    if not cmd_text:
        return {"intent": None, "_source": "empty"}

    try:
        payload = ai_extract_intent(cmd_text)
        if not isinstance(payload, dict):
            return {"intent": None, "_source": "bad_nlp"}
        payload["_raw"] = cmd_text
        payload["_source"] = "nlp_extractor"
        return payload
    except Exception as e:
        return {
            "intent": None,
            "_source": "nlp_error",
            "error": str(e),
            "_raw": cmd_text,
        }


def validate_intent(payload: dict, orders_df: pd.DataFrame, schedule_df: pd.DataFrame):
    """
    Basic validation: check that referenced orders exist, etc.
    """
    intent = payload.get("intent")
    if not intent:
        return False, "No intent detected."

    if intent == "delay_order":
        oid = payload.get("order_id")
        delta = payload.get("delta_minutes")
        if not oid:
            return False, "Missing order_id."
        if oid not in schedule_df["order_id"].unique():
            return False, f"Unknown order_id '{oid}'."
        if delta is None:
            return False, "Missing delta_minutes."
        return True, "OK"

    if intent == "swap_orders":
        o1 = payload.get("order_id")
        o2 = payload.get("other_order_id")
        if not o1 or not o2:
            return False, "Need two order ids."
        all_ids = schedule_df["order_id"].unique().tolist()
        if o1 not in all_ids or o2 not in all_ids:
            return False, "At least one of the orders does not exist in schedule."
        return True, "OK"

    return False, f"Unsupported intent '{intent}'."


def apply_delay(schedule_df: pd.DataFrame, order_id: str, delta_minutes: float):
    """
    Shift all operations of a given order by delta_minutes.
    """
    delta = timedelta(minutes=delta_minutes)
    df = schedule_df.copy()
    mask = df["order_id"] == order_id
    df.loc[mask, "start"] = df.loc[mask, "start"] + delta
    df.loc[mask, "end"] = df.loc[mask, "end"] + delta
    return df


def apply_swap(schedule_df: pd.DataFrame, order_id_1: str, order_id_2: str):
    """
    Swap time windows of two orders (block-wise).
    We keep internal relative structure per order but exchange start/end windows.
    """
    df = schedule_df.copy()
    m1 = df["order_id"] == order_id_1
    m2 = df["order_id"] == order_id_2

    if not m1.any() or not m2.any():
        return df

    # Compute global min start / max end for each order
    s1_min = df.loc[m1, "start"].min()
    s1_max = df.loc[m1, "end"].max()
    s2_min = df.loc[m2, "start"].min()
    s2_max = df.loc[m2, "end"].max()

    # Offsets to move order1 into order2's window and vice versa
    offset_1 = s2_min - s1_min
    offset_2 = s1_min - s2_min

    df.loc[m1, "start"] = df.loc[m1, "start"] + offset_1
    df.loc[m1, "end"] = df.loc[m1, "end"] + offset_1

    df.loc[m2, "start"] = df.loc[m2, "start"] + offset_2
    df.loc[m2, "end"] = df.loc[m2, "end"] + offset_2

    return df


def _process_and_apply(cmd_text: str, *, source_hint: str = None):
    from copy import deepcopy
    try:
        normalized = normalize_order_references(cmd_text)
        payload = extract_intent(normalized)

        # If user didn't specify an order but clicked one on the chart,
        # implicitly target the selected order from the Gantt.
        if (
            payload.get("intent") == "delay_order"
            and not payload.get("order_id")
            and st.session_state.get("selected_order_id")
        ):
            payload["order_id"] = st.session_state.selected_order_id
            payload["_selected_from"] = "chart"

        ok, msg = validate_intent(payload, orders, st.session_state.schedule_df)

        log_payload = deepcopy(payload)
        st.session_state.cmd_log.append({
            "raw": cmd_text,
            "normalized": normalized,
            "payload": log_payload,
            "ok": bool(ok),
            "msg": msg,
            "source": source_hint or payload.get("_source", "?"),
            "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        })

        if not ok:
            st.error(f"❌ {msg}")
            return

        intent = payload.get("intent")
        if intent == "delay_order":
            oid = payload["order_id"]
            delta = payload["delta_minutes"]
            st.session_state.schedule_df = apply_delay(
                st.session_state.schedule_df, oid, delta
            )
            direction = "later" if delta > 0 else "earlier"
            st.success(
                f"✅ Moved order **{oid}** {abs(delta)} min {direction} "
                f"(source: {source_hint or payload.get('_source')})"
            )

        elif intent == "swap_orders":
            o1 = payload["order_id"]
            o2 = payload["other_order_id"]
            st.session_state.schedule_df = apply_swap(
                st.session_state.schedule_df, o1, o2
            )
            st.success(
                f"✅ Swapped order **{o1}** with **{o2}** "
                f"(source: {source_hint or payload.get('_source')})"
            )

        else:
            st.warning(f"Intent '{intent}' is not yet implemented.")

    except Exception as e:
        st.error(f"Error while processing command: {e}")


# ============================ COMMAND PANEL (TEXT + VOICE) ============================

st.markdown("---")
st.markdown("### 🎛️ Command Center")

prompt_container = st.container()

with prompt_container:
    c1, c2 = st.columns([0.82, 0.18])

    with c1:
        st.markdown("**🧠 Command**")
        user_cmd = st.text_input(
            "Type: delay / advance / swap orders…",
            key="prompt_text",
            label_visibility="collapsed",
        )

    with c2:
        st.markdown(
            "<div style='text-align:right; font-size:0.8rem; "
            "margin-bottom:0.25rem;'>🎤 Voice</div>",
            unsafe_allow_html=True,
        )
        audio = mic_recorder(
            start_prompt="Press and hold",
            stop_prompt="Release",
            key="mic",
        )

        if audio is not None:
            # We receive a dict with "bytes"
            audio_bytes = audio["bytes"]
            st.session_state.last_audio_fp = audio_bytes

            try:
                transcript = _deepgram_transcribe_bytes(
                    audio_bytes,
                    api_key=os.getenv("DEEPGRAM_API_KEY", st.secrets.get("DEEPGRAM_API_KEY", "")),
                )
                st.session_state.last_transcript = transcript
                if transcript:
                    _process_and_apply(transcript, source_hint="voice/deepgram")
                    st.rerun()  # rerun AFTER applying voice command
                else:
                    st.warning("No speech detected.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# Text: process once per new command string
if user_cmd and user_cmd != st.session_state.last_processed_cmd:
    st.session_state.last_processed_cmd = user_cmd   # mark as processed
    _process_and_apply(user_cmd, source_hint="text")
    st.session_state.prompt_text = ""                # clear input box
    st.rerun()                                       # rerun AFTER applying text command
