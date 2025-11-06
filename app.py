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

st.set_page_config(page_title="Bulk Production Scheduler", layout="wide")

st.markdown("""
<style>
.main .block-container {
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
    padding-left: 0.6rem;
    padding-right: 0.6rem;
    max-width: 100%;
}
header[data-testid="stHeader"] {
    display: none;
}
#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

try:
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

try:
    os.environ["DEEPGRAM_API_KEY"] = os.environ.get("DEEPGRAM_API_KEY") or st.secrets["DEEPGRAM_API_KEY"]
except Exception:
    pass


# ============================ DATA LOADING ============================

@st.cache_data
def load_and_generate_data():
    orders_df = pd.read_csv("data/orders.csv", parse_dates=["due_date"])
    lines_df = pd.read_csv("data/lines.csv")
    
    time_percentages = {
        'VRAC_SHAMPOO_BASE': {'MIX': 0.100, 'TRF': 0.090, 'FILL': 0.130, 'FIN': 0.080},
        'VRAC_CONDITIONER_BASE': {'MIX': 0.100, 'TRF': 0.050, 'FILL': 0.100, 'FIN': 0.050},
        'VRAC_HAIR_MASK': {'MIX': 0.130, 'TRF': 0.110, 'FILL': 0.100, 'FIN': 0.070}
    }
    
    # Map line_id -> nice name (Mixing/Processing, Transfer/Holding, etc.)
    machine_names = {row['line_id']: row['name'] for _, row in lines_df.iterrows()}
    
    # Logical operation -> physical line
    # NOTE: align with data/lines.csv (TRANS_1 not TRF_1)
    line_map = {
        'MIX': 'MIX_1',
        'TRF': 'TRANS_1',
        'FILL': 'FILL_1',
        'FIN': 'FIN_1'
    }
    
    schedule_rows = []

    # Start horizon
    base_start = datetime(2025, 11, 3, 6, 0)
    machine_timeline = {v: base_start for v in line_map.values()}

    # Sort orders by due date then ID
    orders_sorted = orders_df.sort_values(['due_date', 'order_id']).reset_index(drop=True)
    
    for idx, order in orders_sorted.iterrows():
        order_id = order['order_id']
        sku = order['sku_id']
        qty = order['qty_kg']
        due = order['due_date']
        
        # If unknown sku, fall back to shampoo profile
        percentages = time_percentages.get(sku, time_percentages['VRAC_SHAMPOO_BASE'])

        # Stretch factor so that 12 orders spread over several days
        # (qty / 300 instead of qty / 1000)
        base_time = qty / 300.0  # "pseudo-hours" per order

        operations = [
            ('MIX', percentages['MIX'], 1),
            ('TRF', percentages['TRF'], 2),
            ('FILL', percentages['FILL'], 3),
            ('FIN', percentages['FIN'], 4)
        ]
        
        order_start_time = None
        prev_end = None
        
        for op_type, time_pct, seq in operations:
            machine = line_map[op_type]
            duration_hours = base_time * time_pct
            
            if order_start_time is None:
                op_start = machine_timeline[machine]
                order_start_time = op_start
            else:
                earliest_start = max(prev_end, machine_timeline[machine])
                op_start = earliest_start
            
            op_end = op_start + timedelta(hours=duration_hours)
            
            schedule_rows.append({
                'order_id': order_id,
                'operation': op_type,
                'sequence': seq,
                'machine': machine,
                'machine_name': machine_names.get(machine, machine),
                'start': op_start,
                'end': op_end,
                'due_date': due,
                'wheel_type': sku
            })
            
            machine_timeline[machine] = op_end
            prev_end = op_end
    
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
if "do_rerun" not in st.session_state:
    st.session_state.do_rerun = False


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
    st.markdown("""
        <h4 style="margin-left: 3cm; margin-top: 0.3rem; color: #333;">
            🏭 Bulk Production Planning
        </h4>
    """, unsafe_allow_html=True)

if st.session_state.filters_visible:
    with st.sidebar:
        st.header("Filters ⚙️")
        
        st.session_state.filt_max_orders = st.number_input(
            "Orders",
            1,
            100,
            value=st.session_state.filt_max_orders,
            step=1,
            key="num_orders",
        )
        
        products_all = sorted(base_schedule["wheel_type"].unique().tolist())
        st.session_state.filt_products = st.multiselect(
            "Products",
            products_all,
            default=st.session_state.filt_products,
            key="product_ms",
        )
        
        machines_all = sorted(base_schedule["machine_name"].unique().tolist())
        st.session_state.filt_machines = st.multiselect(
            "Machines",
            machines_all,
            default=st.session_state.filt_machines,
            key="machine_ms",
        )

        color_options = ["Order", "Product", "Machine", "Operation"]
        st.session_state.color_mode = st.selectbox(
            "Color by",
            color_options,
            index=color_options.index(st.session_state.color_mode)
            if st.session_state.color_mode in color_options
            else 1,  # default: Product
            key="color_mode_sb",
        )
        
        if st.button("Reset", key="reset_filters"):
            st.session_state.filt_max_orders = 20
            st.session_state.filt_products = []
            st.session_state.filt_machines = []
            st.session_state.color_mode = "Product"
            st.session_state.cmd_log = []
            st.session_state.last_transcript = None
            st.session_state.last_audio_fp = None
            st.session_state.schedule_df = base_schedule.copy()
            st.rerun()
        
        # === Rich Debug / Trace ===
        with st.expander("🐛 Debug / Trace", expanded=False):
            if st.session_state.last_transcript:
                st.caption("**Last transcript (voice):**")
                st.code(st.session_state.last_transcript)
            else:
                st.caption("No transcript yet.")
            
            if st.session_state.cmd_log:
                last = st.session_state.cmd_log[-1]
                st.markdown("**Last command:**")
                st.markdown(
                    f"- ⏱️ `{last.get('ts', '-')}` "
                    f"from **{last.get('source', '?')}**"
                )
                st.markdown(f"- Raw: `{last.get('raw', '')}`")
                if last.get("normalized") and last["normalized"] != last.get("raw", ""):
                    st.markdown(f"- Normalized: `{last['normalized']}`")
                st.markdown(
                    f"- Status: {'✅ OK' if last.get('ok') else '❌ Error'} — {last.get('msg','')}"
                )
                st.caption("Payload:")
                st.json(last.get("payload", {}))

                # History table (last ~10 commands)
                hist = st.session_state.cmd_log[-10:]
                hist_rows = []
                for h in hist:
                    payload = h.get("payload", {}) or {}
                    hist_rows.append({
                        "time": h.get("ts", ""),
                        "source": h.get("source", ""),
                        "ok": "✅" if h.get("ok") else "❌",
                        "intent": payload.get("intent", ""),
                        "order_1": payload.get("order_id", ""),
                        "order_2": payload.get("order_id_2", ""),
                        "days": payload.get("days", ""),
                        "hours": payload.get("hours", ""),
                        "mins": payload.get("minutes", ""),
                        "msg": h.get("msg", ""),
                    })
                hist_df = pd.DataFrame(hist_rows)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No commands applied yet.")


max_orders = int(st.session_state.filt_max_orders)
product_choice = (
    st.session_state.filt_products
    or sorted(base_schedule["wheel_type"].unique().tolist())
)
machine_choice = (
    st.session_state.filt_machines
    or sorted(base_schedule["machine_name"].unique().tolist())
)
color_mode = st.session_state.color_mode


# ============================ NLP / INTELLIGENCE =========================

DEFAULT_TZ = "Africa/Casablanca"
TZ = pytz.timezone(DEFAULT_TZ)

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

# Pattern to normalize order references "order 1", "Order one", "ord-3", etc.
ORDER_REF_RE = re.compile(
    r"\b(ord(?:er)?)\s*(?:number\s*)?(?P<num>\d{1,3}|\w+)\b",
    flags=re.I,
)

def _num_token_to_float(tok: str):
    t = tok.strip().lower().replace("-", " ").replace(",", ".")
    try:
        return float(t)
    except Exception:
        pass
    parts = [p for p in t.split() if p]
    if len(parts) == 1 and parts[0] in NUM_WORDS:
        return float(NUM_WORDS[parts[0]])
    if len(parts) == 2 and parts[0] in NUM_WORDS and parts[1] in NUM_WORDS:
        return float(NUM_WORDS[parts[0]] + NUM_WORDS[parts[1]])
    return None


def normalize_order_references(text: str) -> str:
    """
    Convert variations like:
    - 'order 1', 'order one', 'ord 7', 'order number 15'
    into canonical IDs: 'ORD-001' .. 'ORD-100'
    """
    def repl(m: re.Match) -> str:
        raw = m.group("num")
        n = None
        if raw.isdigit():
            n = int(raw)
        else:
            key = raw.lower()
            if key in NUM_WORDS:
                n = NUM_WORDS[key]
        if not n or n < 1 or n > 100:
            return m.group(0)
        return f"ORD-{n:03d}"

    # main pattern: "order X"
    new_text = ORDER_REF_RE.sub(repl, text)

    # Optionally, support bare '#1' → ORD-001 when it's clearly an order ref
    def repl_hash(m: re.Match) -> str:
        num = m.group(1)
        if not num.isdigit():
            return m.group(0)
        n = int(num)
        if n < 1 or n > 100:
            return m.group(0)
        return f"ORD-{n:03d}"

    new_text = re.sub(r"\border\s*#(\d{1,3})\b", repl_hash, new_text, flags=re.I)
    return new_text


def _parse_duration_chunks(text: str):
    d = {"days": 0.0, "hours": 0.0, "minutes": 0.0}
    for num, unit in re.findall(
        r"([\d\.,]+|\b\w+\b)\s*(days?|d|hours?|h|minutes?|mins?|m)\b",
        text,
        flags=re.I,
    ):
        n = _num_token_to_float(num)
        if n is None:
            continue
        u = unit.lower()
        if u.startswith("d"):
            d["days"] += n
        elif u.startswith("h"):
            d["hours"] += n
        else:
            d["minutes"] += n
    return d


def _regex_fallback(user_text: str):
    """
    Lightweight rule-based extractor for:
      - swap ORD-001 with ORD-005
      - delay ORD-003 by 2 days / 5 hours
    Works on already-normalized order IDs (ORD-001 etc.).
    """
    t = user_text.strip()
    low = t.lower()
    
    # swap ORD-001 with ORD-005
    m = re.search(
        r"(?:^|\b)(swap|switch)\s+(ord-\d{3})\s*(?:with|and|&)?\s*(ord-\d{3})\b",
        low,
    )
    if m:
        return {
            "intent": "swap_orders",
            "order_id": m.group(2).upper(),
            "order_id_2": m.group(3).upper(),
            "_source": "regex",
        }
    
    # delay / advance (advance = negative delay)
    delay_sign = +1
    if re.search(r"\b(advance|bring\s+forward|pull\s+in)\b", low):
        delay_sign = -1
        low_norm = re.sub(r"\b(advance|bring\s+forward|pull\s+in)\b", "delay", low)
    else:
        low_norm = low
    
    m = re.search(r"(delay|push|postpone)\s+(ord-\d{3}).*?\bby\b\s+(.+)$", low_norm)
    if m:
        oid = m.group(2).upper()
        dur_text = m.group(3)
        dur = _parse_duration_chunks(dur_text)
        if any(v != 0 for v in dur.values()):
            return {
                "intent": "delay_order",
                "order_id": oid,
                "days": delay_sign * dur["days"],
                "hours": delay_sign * dur["hours"],
                "minutes": delay_sign * dur["minutes"],
                "_source": "regex",
            }
    
    # fallback if we just see "delay ORD-001 2 days"
    m = re.search(
        r"(delay|push|postpone)\s+(ord-\d{3}).*?(days?|d|hours?|h|minutes?|mins?|m)\b",
        low_norm,
    )
    if m:
        oid = m.group(2).upper()
        dur = _parse_duration_chunks(low_norm)
        if any(v != 0 for v in dur.values()):
            return {
                "intent": "delay_order",
                "order_id": oid,
                "days": delay_sign * dur["days"],
                "hours": delay_sign * dur["hours"],
                "minutes": delay_sign * dur["minutes"],
                "_source": "regex",
            }
    
    return {"intent": "unknown", "raw": user_text, "_source": "regex"}


def extract_intent(user_text: str) -> dict:
    user_text = normalize_order_references(user_text)
    payload = _regex_fallback(user_text)
    if payload.get("intent") == "unknown":
        ai_payload = ai_extract_intent(user_text)
        ai_payload["_source"] = "openai"
        return ai_payload
    return payload


def validate_intent(payload: dict, orders_df, sched_df):
    intent = payload.get("intent")
    
    def order_exists(oid):
        return oid and (orders_df["order_id"] == oid).any()
    
    if intent not in ("delay_order", "swap_orders"):
        return False, "Unsupported intent"
    
    if intent in ("delay_order", "swap_orders"):
        oid = payload.get("order_id")
        if not order_exists(oid):
            return False, f"Unknown order: {oid}"
    
    if intent == "swap_orders":
        oid2 = payload.get("order_id_2")
        if not order_exists(oid2):
            return False, f"Unknown order: {oid2}"
        if oid2 == payload.get("order_id"):
            return False, "Cannot swap same order"
        return True, "ok"
    
    if intent == "delay_order":
        for k in ("days", "hours", "minutes"):
            if k in payload and payload[k] is not None:
                try:
                    payload[k] = float(payload[k])
                except Exception:
                    return False, f"{k} must be numeric"
        if not any(payload.get(k) for k in ("days", "hours", "minutes")):
            return False, "Need duration (days/hours/minutes)"
        return True, "ok"
    
    return False, "Invalid payload"


# ============================ APPLY FUNCTIONS =========================

def _repack_touched_machines(s: pd.DataFrame, touched_orders):
    machines = s.loc[s["order_id"].isin(touched_orders), "machine"].unique().tolist()
    for m in machines:
        block_idx = s.index[s["machine"] == m]
        block = s.loc[block_idx].sort_values(["start", "end"]).copy()
        last_end = None
        for idx, row in block.iterrows():
            if last_end is not None and row["start"] < last_end:
                dur = row["end"] - row["start"]
                s.at[idx, "start"] = last_end
                s.at[idx, "end"] = last_end + dur
            last_end = s.at[idx, "end"]
    return s


def apply_delay(schedule_df: pd.DataFrame, order_id: str, days=0, hours=0, minutes=0):
    s = schedule_df.copy()
    delta = timedelta(
        days=float(days or 0),
        hours=float(hours or 0),
        minutes=float(minutes or 0),
    )
    
    order_ops = s[s["order_id"] == order_id].sort_values("sequence")
    
    for idx, op in order_ops.iterrows():
        dur = op["end"] - op["start"]
        s.at[idx, "start"] = op["start"] + delta
        s.at[idx, "end"] = s.at[idx, "start"] + dur
    
    return _repack_touched_machines(s, [order_id])


def apply_swap(schedule_df: pd.DataFrame, a: str, b: str):
    s = schedule_df.copy()
    a0 = s.loc[s["order_id"] == a, "start"].min()
    b0 = s.loc[s["order_id"] == b, "start"].min()
    da = b0 - a0
    db = a0 - b0
    s = apply_delay(
        s, a,
        days=da.days,
        hours=da.seconds // 3600,
        minutes=(da.seconds % 3600) // 60,
    )
    s = apply_delay(
        s, b,
        days=db.days,
        hours=db.seconds // 3600,
        minutes=(db.seconds % 3600) // 60,
    )
    return s


# ============================ GANTT =========================

sched = st.session_state.schedule_df.copy()
sched = sched[sched["wheel_type"].isin(product_choice)]
sched = sched[sched["machine_name"].isin(machine_choice)]

order_priority = (
    sched.groupby("order_id", as_index=False)["start"]
    .min()
    .sort_values("start")
)
keep_ids = order_priority["order_id"].head(max_orders).tolist()
sched = sched[sched["order_id"].isin(keep_ids)].copy()

if sched.empty:
    st.info("No operations match filters")
else:
    unique_orders = sorted(sched["order_id"].unique())
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]
    while len(color_palette) < len(unique_orders):
        color_palette.extend(color_palette[:10])
    
    order_color_map = {
        oid: color_palette[i % len(color_palette)]
        for i, oid in enumerate(unique_orders)
    }
    sched["order_color"] = sched["order_id"].map(order_color_map)

    select_order = alt.selection_point(
        fields=["order_id"], on="click", clear="dblclick"
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
    
    machine_order = [
        "Mixing/Processing",
        "Transfer/Holding",
        "Filling/Capping",
        "Finishing/QC"
    ]
    
    base_enc = {
        "y": alt.Y(
            "machine_name:N",
            sort=machine_order,
            title=None,
            axis=alt.Axis(labelLimit=200)
        ),
        "x": alt.X("start:T", title=None, axis=alt.Axis(format="%b %d %H:%M")),
        "x2": "end:T",
    }
    
    bars = (
        alt.Chart(sched)
        .mark_bar(cornerRadius=2)
        .encode(
            color=color_encoding,
            opacity=alt.condition(
                select_order, alt.value(1.0), alt.value(0.3)
            ),
            tooltip=[
                alt.Tooltip("order_id:N", title="Order"),
                alt.Tooltip("operation:N", title="Op"),
                alt.Tooltip("machine_name:N", title="Machine"),
                alt.Tooltip("start:T", title="Start", format="%b %d %H:%M"),
                alt.Tooltip("end:T", title="End", format="%b %d %H:%M"),
                alt.Tooltip("due_date:T", title="Due", format="%b %d"),
            ],
        )
    )
    
    labels = (
        alt.Chart(sched)
        .mark_text(align="left", dx=4, baseline="middle", fontSize=9, color="white")
        .encode(
            text="order_id:N",
            opacity=alt.condition(
                select_order, alt.value(1.0), alt.value(0.7)
            ),
        )
    )
    
    gantt = (
        alt.layer(bars, labels, data=sched)
        .encode(**base_enc)
        .add_params(select_order)
        .properties(width="container", height=350)
        .configure_view(stroke=None)
    )
    
    st.altair_chart(gantt, use_container_width=True)


# ============================ DEEPGRAM TRANSCRIPTION =========================

def _deepgram_transcribe_bytes(wav_bytes: bytes, mimetype: str = "audio/wav") -> str:
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")
    import requests
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en"
    headers = {"Authorization": f"Token {key}", "Content-Type": mimetype}
    r = requests.post(url, headers=headers, data=wav_bytes, timeout=45)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Deepgram error: {r.text}") from e
    j = r.json()
    try:
        return j["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except Exception:
        raise RuntimeError(f"Deepgram: no transcript in response: {j}")


# ============================ PROCESS COMMAND =========================

def _process_and_apply(cmd_text: str, *, source_hint: str = None):
    from copy import deepcopy
    try:
        # 1) Normalize order references for both text & voice
        normalized = normalize_order_references(cmd_text)

        # 2) Extract intent on normalized text (regex + OpenAI fallback)
        payload = extract_intent(normalized)

        # 3) Validate vs current data
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
        st.session_state.cmd_log = st.session_state.cmd_log[-50:]
        
        if not ok:
            st.error(f"❌ {msg}")
            return
        
        if payload["intent"] == "delay_order":
            st.session_state.schedule_df = apply_delay(
                st.session_state.schedule_df,
                payload["order_id"],
                days=payload.get("days", 0),
                hours=payload.get("hours", 0),
                minutes=payload.get("minutes", 0),
            )
            direction = "Advanced" if (payload.get("days", 0) < 0 or 
                       payload.get("hours", 0) < 0 or 
                       payload.get("minutes", 0) < 0) else "Delayed"
            st.success(f"✅ {direction} {payload['order_id']}")
        elif payload["intent"] == "swap_orders":
            st.session_state.schedule_df = apply_swap(
                st.session_state.schedule_df,
                payload["order_id"],
                payload["order_id_2"],
            )
            st.success(f"✅ Swapped {payload['order_id']} ↔ {payload['order_id_2']}")
        
        # ask for one rerun at end of script
        st.session_state.do_rerun = True

    except Exception as e:
        st.error(f"⚠️ Error: {e}")


# ============================ VOICE + TEXT PROMPT BAR =========================

st.markdown("---")
prompt_container = st.container()

with prompt_container:
    # prompt left, mic right
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
        rec = mic_recorder(
            start_prompt="●",   # minimal round icon style
            stop_prompt="■",
            key="voice_mic",
            just_once=False,
            format="wav",
            use_container_width=False,  # keep it compact
        )

# Handle voice
if rec and isinstance(rec, dict) and rec.get("bytes"):
    wav_bytes = rec["bytes"]
    fp = (len(wav_bytes), hash(wav_bytes[:1024]))
    if fp != st.session_state.last_audio_fp:
        st.session_state.last_audio_fp = fp
        try:
            with st.spinner("Transcribing…"):
                transcript = _deepgram_transcribe_bytes(wav_bytes, mimetype="audio/wav")
            st.session_state.last_transcript = transcript
            if transcript:
                _process_and_apply(transcript, source_hint="voice/deepgram")
            else:
                st.warning("No speech detected.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# Handle typed text
if user_cmd:
    _process_and_apply(user_cmd, source_hint="text")
    # Clear the box on next run
    st.session_state.prompt_text = ""

# Single rerun after successful command to refresh Gantt above
if st.session_state.get("do_rerun"):
    st.session_state.do_rerun = False
    st.rerun()
