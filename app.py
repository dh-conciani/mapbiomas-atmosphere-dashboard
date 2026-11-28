"""MapBiomas Atmosphere Explorer — analytical dashboard (v4 · redesigned UI).

Reads the original XLSX-derived, year-partitioned Parquet tables and genuine
Earth Engine GeoJSON geometries. No scientific data are modified or estimated.
Visualization palettes adopt documented MapBiomas brand colors, while numerical
maps use continuous scales appropriate to their variables.

v4: bilingual interface (Portuguese by default, English optional), redesigned
layout with tabbed analysis sections, modern controls and a unified visual
style. Statistical routines and data handling are unchanged from v3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import i18n

BASE = Path(__file__).resolve().parent / "data"
PERIODS = ["Annual", "Monthly", "Multiyear"]

# Verified MapBiomas institution-wide identity colors. These are NOT categorical
# map legends for atmospheric products, which are continuous measurements.
BRAND = {
    "blue": "#62A5BB", "sky": "#75C7E0", "green": "#77B893",
    "mint": "#84CAA0", "yellow": "#DCDB77", "olive": "#C5C56C",
    "orange": "#F7963F", "amber": "#E18837", "charcoal": "#333333",
}
OFFICIAL_LAND_CLASS = {"forest": "#1f8d49", "pasture": "#edde8e", "agriculture": "#E974ED"}
SERIES_COLORS = ["#28688B", "#C76441", "#27917A", "#A14D79", "#C18A26", "#715B9A", "#3596B1", "#9E6A37", "#447642", "#C54B55", "#668DB2", "#AA8A61"]
DIVERGING = [[0, "#285D8C"], [.22, "#75C7E0"], [.5, "#F5F5F1"], [.78, "#F7963F"], [1, "#B94236"]]
SEQUENTIAL = {
    "heat": ["#FFF6DB", "#F8D58B", "#F7963F", "#D65A35", "#902D31"],
    "rain": ["#F2F9FC", "#C2E4EE", "#75C7E0", "#367CA8", "#174C76"],
    "forest": ["#F2F8E6", "#C9E7BE", "#84CAA0", "#428F65", "#165137"],
    "agriculture": ["#FFF4F6", "#F3C3D3", "#E29AB8", "#CE668E", "#8F335F"],
    "pasture": ["#FFFCEB", "#EEEBA5", "#DCDB77", "#B6B24C", "#74772B"],
    "burn": ["#FFF3DA", "#F9C276", "#F7963F", "#D66C2E", "#8B371E"],
    "air": ["#FAF6FC", "#DBCCE9", "#B99BCC", "#8064A8", "#48316A"],
    "other": ["#F4F7F8", "#C1D9D9", "#62A5BB", "#438494", "#235360"],
}
FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"
INK = "#163548"

# ------------------------------- Language ----------------------------------
# Portuguese is the default; ?lang=en in the URL (or the header toggle) switches.
if "lang" not in st.session_state:
    requested = str(st.query_params.get("lang", i18n.DEFAULT_LANG)).lower()
    st.session_state["lang"] = requested if requested in i18n.LANGS else i18n.DEFAULT_LANG
LANG = st.session_state["lang"]

# -------------------------------- Theme ------------------------------------
# Light is the default. The header toggle (or ?theme=dark) switches Streamlit's
# own light/dark theme, so native widgets, tables and charts all follow it.
THEME = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
PALETTE = {
    "light": {"ink": "#163548", "text": "#213D50", "muted": "#6B8290", "grid": "rgba(100,140,158,0.14)",
              "paper": "#FFFFFF", "map_style": "carto-positron", "outline": "rgba(34,60,75,.58)",
              "fit": "#1B364A", "ref": "#7A8790",
              "kpi_bg": ["#E6F2F8", "#E7F4EE", "#F3EEF8", "#FDF1E4"]},
    "dark": {"ink": "#E8F1F5", "text": "#C9D6DD", "muted": "#8FA3AE", "grid": "rgba(160,190,205,0.13)",
             "paper": "#0F1418", "map_style": "carto-darkmatter", "outline": "rgba(210,228,236,.45)",
             "fit": "#F2F6F8", "ref": "#8FA3AE",
             "kpi_bg": ["#12303F", "#123326", "#2A2140", "#3A2A16"]},
}[THEME]


def _lighten(hex_color, amount=.35):
    r, g, b = (int(hex_color[k:k + 2], 16) for k in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(*(round(c + (255 - c) * amount) for c in (r, g, b)))


# Brighter categorical colors on the black background keep markers legible.
SERIES = SERIES_COLORS if THEME == "light" else [_lighten(c) for c in SERIES_COLORS]


def t(key, **kwargs):
    text = i18n.T[LANG].get(key, i18n.T["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text


MONTH_NAMES = i18n.MONTHS[LANG]


def fmt_num(value, spec=",.2f"):
    """Locale-aware number formatting (pt: 1.234,56 · en: 1,234.56)."""
    try:
        text = format(value, spec)
    except (TypeError, ValueError):
        return str(value)
    if LANG == "pt":
        text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return text


def vname(column):
    names = i18n.VARIABLES.get(column)
    if names:
        return names[0] if LANG == "pt" else names[1]
    return column.replace("_", " ").capitalize()


def unit_text(unit):
    return i18n.UNIT_LABELS[LANG].get(unit, unit)


def level_text(key, levels):
    return i18n.LEVEL_LABELS[LANG].get(key, levels.get(key, {}).get("label", key))


def period_text(period):
    return i18n.PERIOD_LABELS[LANG].get(period, period)


st.set_page_config(page_title=t("page_title"), page_icon="🌎", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --mb-teal: #27917A; --mb-sky: #75C7E0; --mb-green: #77B893; --mb-radius: 14px;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif !important; }
.stApp { background: var(--mb-bg); }
.block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1640px; }
h1, h2, h3, h4 { letter-spacing: -.02em; color: var(--mb-ink); font-weight: 700; }

/* ---------- Header ---------- */
.st-key-mb_hero {
  background: radial-gradient(1200px 300px at 100% 0%, rgba(117,199,224,.35), transparent 60%),
              linear-gradient(120deg, #123347 0%, #1D5F84 55%, #23806F 100%);
  border-radius: 20px; padding: 1.35rem 1.6rem 1.25rem 1.6rem; box-shadow: var(--mb-shadow);
  margin-bottom: .35rem; }
.st-key-mb_hero .mb-eyebrow, .st-key-mb_hero .mb-title, .st-key-mb_hero .mb-subtitle { margin: 0; }
.mb-eyebrow { color: #BFE6F1; font-size: .78rem; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; margin: 0 0 .25rem 0; display: flex; align-items: center; gap: .5rem; }
.mb-eyebrow .dot { width: 9px; height: 9px; border-radius: 50%;
  background: linear-gradient(135deg, var(--mb-sky), var(--mb-green)); display: inline-block; }
.st-key-mb_hero .mb-title { color: #FFFFFF; font-size: 2.1rem; font-weight: 800; line-height: 1.12; margin: 0; letter-spacing: -.03em; }
.st-key-mb_hero .mb-subtitle { color: #D6E9F0; font-size: 1rem; margin-top: .45rem; font-weight: 400; }
.mb-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .9rem; }
.mb-chip { background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.22); color: #FFFFFF;
  padding: .22rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 500; white-space: nowrap; }
.mb-chip b { font-weight: 700; }
.st-key-mb_lang [data-testid="stWidgetLabel"] p, .st-key-mb_lang [data-testid="stWidgetLabel"] label { color: #BFE6F1 !important; font-size: .72rem;
  font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.st-key-mb_lang button { background: rgba(255,255,255,.12) !important; color: #FFFFFF !important;
  border-color: rgba(255,255,255,.3) !important; font-weight: 600; min-width: 2.9rem; }
.st-key-mb_lang { gap: .9rem !important; }
.st-key-mb_theme_js { display: none; }
.st-key-mb_lang button[aria-checked="true"], .st-key-mb_lang button[kind*="Active"] {
  background: #FFFFFF !important; color: #163548 !important; }

/* ---------- KPI cards ---------- */
.mb-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .9rem 0 .4rem 0; }
.mb-kpi { background: var(--mb-card); border: 1px solid var(--mb-line); border-radius: var(--mb-radius);
  padding: .85rem 1rem; box-shadow: var(--mb-shadow); display: flex; gap: .8rem; align-items: center; }
.mb-kpi .ic { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
  font-size: 1.15rem; flex: none; }
.mb-kpi .lb { color: var(--mb-muted); font-size: .76rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.mb-kpi .vl { color: var(--mb-ink); font-size: 1.38rem; font-weight: 750; line-height: 1.2; letter-spacing: -.02em; }
@media (max-width: 900px) { .mb-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .st-key-mb_hero .mb-title { font-size: 1.55rem; } }

/* ---------- Section headers ---------- */
.mb-section { margin: .6rem 0 .5rem 0; }
.mb-section h3 { margin: 0; font-size: 1.22rem; }
.mb-section p { margin: .2rem 0 0 0; color: var(--mb-muted); font-size: .9rem; }
.mb-label { font-weight: 700; color: var(--mb-ink); font-size: .95rem; margin: .2rem 0 .3rem 0; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--mb-line); }
.stTabs [data-baseweb="tab"] { height: 2.7rem; padding: 0 1rem; border-radius: 10px 10px 0 0;
  font-weight: 600; color: var(--mb-ink-2); }
.stTabs [data-baseweb="tab"]:hover { background: var(--mb-soft); color: var(--mb-blue); }
.stTabs [aria-selected="true"] { color: var(--mb-blue) !important; background: var(--mb-tab-sel); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--mb-blue); height: 3px; border-radius: 3px; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 999px; font-weight: 600; font-size: .86rem; padding: .38rem .95rem;
  border: 1px solid var(--mb-btn-border); color: var(--mb-ink); background: var(--mb-card);
  transition: all .15s ease; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--mb-blue); color: var(--mb-blue); background: var(--mb-hover); transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(36,119,165,.15); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(120deg, #1D6A93, #23806F); color: #FFFFFF; border: none; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover { color: #FFFFFF; filter: brightness(1.06); }
.stButton > button[kind="tertiary"] { box-shadow: none; border: none; background: transparent; color: var(--mb-muted); }

/* ---------- Cards / containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: var(--mb-radius); }
div[data-testid="stMetric"] { background: var(--mb-card); border: 1px solid var(--mb-line); border-radius: 12px;
  padding: .6rem .85rem; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
[data-testid="stMetricLabel"] p { color: var(--mb-muted); font-size: .78rem; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.3rem; color: var(--mb-ink); font-weight: 700; }
[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--mb-line); background: var(--mb-card); }
[data-testid="stExpander"] summary p { font-weight: 600; color: var(--mb-ink); }
[data-testid="stPlotlyChart"] { background: var(--mb-card); border: 1px solid var(--mb-line); border-radius: var(--mb-radius);
  padding: .35rem; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stCaptionContainer"] { color: var(--mb-muted); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { min-width: 360px !important; width: 360px !important; }
[data-testid="stSidebar"] > div:first-child { width: 360px !important; }
@media (max-width: 750px) { [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  min-width: min(360px, 95vw) !important; width: min(360px, 95vw) !important; } }
[data-testid="stSidebar"] { background: var(--mb-side); border-right: 1px solid var(--mb-line); }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .6rem; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div { max-height: 168px; overflow-y: auto; }
[data-testid="stSidebar"] .stButton > button { font-size: .8rem; padding: .28rem .6rem; min-height: 2.1rem; gap: .25rem; }
.st-key-mb_quick { gap: .35rem !important; }
[data-testid="stSidebarHeader"] { height: 2.2rem; min-height: 2.2rem; padding-bottom: 0; }
.mb-side-brand { display: flex; align-items: center; gap: .6rem; margin: -.4rem 0 .2rem 0; }
.mb-side-brand .logo { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  background: linear-gradient(135deg, #1D6A93, #23806F); color: #FFF; font-weight: 800; font-size: .95rem; }
.mb-side-brand .nm { font-weight: 800; color: var(--mb-ink); line-height: 1.1; font-size: 1rem; }
.mb-side-brand .sb { color: var(--mb-muted); font-size: .76rem; }
.mb-step { display: flex; align-items: center; gap: .5rem; margin: .9rem 0 .15rem 0; color: var(--mb-ink);
  font-weight: 700; font-size: .82rem; letter-spacing: .08em; text-transform: uppercase; }
.mb-step .n { width: 22px; height: 22px; border-radius: 50%; background: var(--mb-blue); color: #FFF;
  display: grid; place-items: center; font-size: .72rem; letter-spacing: 0; }
.mb-count { background: var(--mb-card); border: 1px dashed var(--mb-btn-border); border-radius: 10px; padding: .45rem .7rem;
  font-size: .8rem; color: var(--mb-ink-2); }

.mb-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--mb-line);
  color: var(--mb-muted); font-size: .8rem; text-align: center; }
</style>
""", unsafe_allow_html=True)
THEME_VARS = {
    "light": """--mb-ink: #163548; --mb-ink-2: #3E5A6C; --mb-muted: #6B8290; --mb-blue: #2477A5;
  --mb-line: #E1EAEE; --mb-soft: #F4F8FA; --mb-bg: #FAFCFD; --mb-card: #FFFFFF; --mb-hover: #F1F8FB;
  --mb-tab-sel: #EEF6FA; --mb-btn-border: #CFDDE4;
  --mb-side: linear-gradient(180deg, #F4F8FA 0%, #EEF4F6 100%);
  --mb-shadow: 0 1px 2px rgba(22,53,72,.05), 0 6px 18px rgba(22,53,72,.06);""",
    "dark": """--mb-ink: #E8F1F5; --mb-ink-2: #C3D2DA; --mb-muted: #8FA3AE; --mb-blue: #5BB8E0;
  --mb-line: #1F2A31; --mb-soft: #12191E; --mb-bg: #050708; --mb-card: #0F1418; --mb-hover: #14212A;
  --mb-tab-sel: #10222D; --mb-btn-border: #2A3942;
  --mb-side: linear-gradient(180deg, #0B1014 0%, #080B0E 100%);
  --mb-shadow: 0 1px 2px rgba(0,0,0,.5), 0 6px 18px rgba(0,0,0,.45);""",
}[THEME]
st.markdown(f"<style>:root {{ {THEME_VARS} }}</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def catalog():
    return json.loads((BASE / "catalog.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_territories(level):
    return pd.read_parquet(BASE / "territories" / f"{level}.parquet").fillna("")


@st.cache_data(show_spinner=False)
def get_dictionary(level):
    return json.loads((BASE / "dictionaries" / f"{level}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_geojson(level):
    path = BASE / "geometries" / f"{level}.geojson"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False, max_entries=16)
def load_metrics(level, period, years, columns, ids):
    directory = BASE / "metrics" / level / period.lower()
    paths = [directory / f"year={year}" / "data.parquet" for year in years]
    paths = [p for p in paths if p.exists()]
    keys = ["territory_id", "territory_name"]
    keys += (["year", "month"] if period == "Monthly" else ["year"] if period == "Annual" else ["period"])
    desired = list(dict.fromkeys(keys + list(columns)))
    if not paths:
        return pd.DataFrame(columns=desired)
    frames = [pd.read_parquet(p, columns=desired, filters=[("territory_id", "in", list(ids))]) for p in paths]
    return pd.concat(frames, ignore_index=True)


def metric_label(m, metadata):
    unit = metadata.get(m, {}).get("unit", "")
    return f"{vname(m)} [{unit_text(unit)}]" if unit else vname(m)


def clean_value(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def _polygon_parts(geometry):
    if not isinstance(geometry, dict):
        return
    kind = geometry.get("type")
    if kind == "Polygon":
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, list) and coordinates:
            yield coordinates
    elif kind == "MultiPolygon":
        for polygon in geometry.get("coordinates") or []:
            if isinstance(polygon, list) and polygon:
                yield polygon
    elif kind == "GeometryCollection":
        for child in geometry.get("geometries") or []:
            yield from _polygon_parts(child)


def polygonal_geometry(geometry):
    parts = list(_polygon_parts(geometry))
    if not parts:
        return None
    return {"type": "Polygon", "coordinates": parts[0]} if len(parts) == 1 else {"type": "MultiPolygon", "coordinates": parts}


def geo_bounds(features):
    xs, ys = [], []
    def walk(item):
        if not isinstance(item, (list, tuple)) or not item:
            return
        if len(item) >= 2 and isinstance(item[0], (int, float)) and isinstance(item[1], (int, float)):
            if np.isfinite(item[0]) and np.isfinite(item[1]):
                xs.append(float(item[0])); ys.append(float(item[1]))
            return
        for child in item:
            walk(child)
    for feature in features:
        polygon = polygonal_geometry(feature.get("geometry"))
        if polygon:
            walk(polygon["coordinates"])
    if not xs:
        return {"lat": -12, "lon": -54}, 3.5
    center = {"lat": (min(ys) + max(ys)) / 2, "lon": (min(xs) + max(xs)) / 2}
    span = max(max(xs) - min(xs), 1.65 * (max(ys) - min(ys)), 1.0)
    return center, float(np.clip(np.log2(360 / span) - 1.55, 2.0, 10.5))


def palette_family(metric, metadata):
    m = metric.lower()
    grp = metadata.get(metric, {}).get("group", "").lower()
    if any(s in m for s in ("anomaly", "anomalia", "change", "difference")):
        return "diverging"
    if any(s in m for s in ("temperature", "heatwave", "temp_")):
        return "heat"
    if any(s in m for s in ("dry_days", "drought")):
        return "burn"  # More dry days: warm, not rainfall blue.
    if any(s in m for s in ("precip", "rain", "wet_days", "water", "humidity")):
        return "rain"
    if any(s in m for s in ("pm10", "pm25", "pm2", "particulate", "aerosol", "vapor", "vpd")):
        return "air"
    if "burn" in m or "fire" in m:
        return "burn"
    if any(s in m for s in ("forest", "wood", "silviculture")):
        return "forest"
    if "pasture" in m or "grassland" in m:
        return "pasture"
    if any(s in m for s in ("agricult", "farming", "cropland")):
        return "agriculture"
    if "land cover" in grp:
        return "forest"
    return "other"


def variable_color_map(variables, metadata):
    """Consistent semantic colors; land-cover anchors are official class colors.

    Continuous atmospheric variables have no official class-code palette.
    """
    anchors = {"heat": "#D65A35", "rain": "#367CA8", "air": "#8064A8",
               "forest": OFFICIAL_LAND_CLASS["forest"],
               "pasture": OFFICIAL_LAND_CLASS["pasture"],
               "agriculture": OFFICIAL_LAND_CLASS["agriculture"],
               "burn": "#E18837", "other": "#438494", "diverging": "#B94236"}
    counts = {}
    result = {}
    for i, name in enumerate(variables):
        family = palette_family(name, metadata)
        k = counts.get(family, 0)
        counts[family] = k + 1
        # Retain the family anchor for the first member. Different shades allow
        # comparison of several metrics from the same thematic family.
        if k == 0:
            result[name] = anchors[family]
        else:
            family_palette = SEQUENTIAL.get(family, SEQUENTIAL["other"])
            result[name] = family_palette[[3, 2, 4, 1][(k - 1) % 4]]
    return result


def map_scale(metric, metadata):
    family = palette_family(metric, metadata)
    return DIVERGING if family == "diverging" else SEQUENTIAL[family]


def limits(*arrays, symmetric=False):
    numeric = np.concatenate([np.asarray(x, dtype=float).ravel() for x in arrays if len(x)]) if any(len(x) for x in arrays) else np.array([])
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        bound = float(np.max(np.abs(finite))) or 1.0
        return -bound, bound
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        return lo - max(abs(lo) * .05, .5), hi + max(abs(hi) * .05, .5)
    return lo, hi


def map_figure(layer, metric, features, center, zoom, scale, vmin, vmax, label, height=420):
    geo = {"type": "FeatureCollection", "features": features}
    frame = layer[["territory_id", "territory_label", metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    fig = px.choropleth_map(
        frame, geojson=geo, locations="territory_id", featureidkey="properties.territory_id",
        color=metric, hover_name="territory_label",
        hover_data={"territory_id": True, metric: ":.3f"},
        color_continuous_scale=scale, range_color=(vmin, vmax),
        map_style=PALETTE["map_style"], center=center, zoom=zoom, opacity=.83,
        labels={metric: label, "territory_id": "ID"},
    )
    fig.update_traces(marker_line_width=.45, marker_line_color=PALETTE["outline"])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title=dict(text="", side="right"), thickness=13, len=.73, x=1.01),
        font=dict(family=FONT_FAMILY, color=PALETTE["text"]), separators=",." if LANG == "pt" else ".,",
        paper_bgcolor=PALETTE["paper"])
    return fig


def observations_at(frame, year, month, period, metric):
    if period != "Multiyear":
        frame = frame[frame["year"] == year]
    if period == "Monthly":
        frame = frame[frame["month"] == month]
    subset = frame[["territory_id", "territory_label", metric]].drop_duplicates("territory_id").copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    return subset


def format_p(p):
    """Format p-values without displaying very small probabilities as zero."""
    if p is None or not np.isfinite(p):
        return t("n_a")
    if p <= 0:
        return "<1e-300"
    return f"{p:.2e}" if p < .001 else f"{p:.4f}"


def linear_regression(frame, x, y):
    """Simple OLS + two-sided nominal p-value for the slope/Pearson association.

    IMPORTANT: Inferential statistics assume independent observations. When the
    same territory contributes repeated years/months, p-values are exploratory,
    not temporally/spatially robust.
    """
    data = (frame[[x, y]].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).dropna())
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return None
    xa, ya = data[x].to_numpy(dtype=float), data[y].to_numpy(dtype=float)
    try:
        fitted = stats.linregress(xa, ya, alternative="two-sided")
    except (ValueError, FloatingPointError):
        return None
    if not all(np.isfinite(v) for v in
               (fitted.slope, fitted.intercept, fitted.rvalue, fitted.pvalue, fitted.stderr)):
        return None
    tcrit = stats.t.ppf(.975, df=len(data) - 2)
    delta = tcrit * fitted.stderr
    return {
        "slope": float(fitted.slope),
        "intercept": float(fitted.intercept),
        "r": float(fitted.rvalue),
        "r2": float(fitted.rvalue ** 2),
        "p_value": float(fitted.pvalue),
        "ci95_low": float(fitted.slope - delta),
        "ci95_high": float(fitted.slope + delta),
        "n": len(data),
        "xmin": float(xa.min()),
        "xmax": float(xa.max()),
    }


def correlation_tests(frame, variables):
    """Pairwise-complete Pearson r, nominal two-sided p and Benjamini–Hochberg q.

    BH adjustment is performed across the unique OFF-DIAGONAL pairs in the
    currently selected correlation matrix. Rows are assumed independent for the
    p-values; this is not generally satisfied by repeated territorial records.
    """
    data = (frame[list(variables)].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan))
    k = len(variables)
    r = np.full((k, k), np.nan, dtype=float)
    p = np.full((k, k), np.nan, dtype=float)
    q = np.full((k, k), np.nan, dtype=float)
    n = np.zeros((k, k), dtype=int)
    tested = []
    for i, a in enumerate(variables):
        valid = data[a].dropna()
        n[i, i] = len(valid)
        if len(valid) >= 2 and valid.nunique() >= 2:
            r[i, i] = 1.0
        for j in range(i+1, k):
            b = variables[j]
            pair = data[[a, b]].dropna()
            n[i, j] = n[j, i] = len(pair)
            if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                continue
            try:
                coefficient, pvalue = stats.pearsonr(pair[a].to_numpy(dtype=float),
                                                    pair[b].to_numpy(dtype=float),
                                                    alternative="two-sided")
            except ValueError:
                continue
            if np.isfinite(coefficient) and np.isfinite(pvalue):
                r[i, j] = r[j, i] = float(coefficient)
                p[i, j] = p[j, i] = float(pvalue)
                tested.append((i, j, a, b, len(pair), float(coefficient), float(pvalue)))

    # False discovery rate (Benjamini-Hochberg) across the tested metric pairs.
    if tested:
        values = np.array([record[6] for record in tested], dtype=float)
        order = np.argsort(values)
        m = len(values)
        ordered_q = np.minimum.accumulate((values[order] * m / np.arange(1, m+1))[::-1])[::-1]
        qvalues = np.empty_like(values)
        qvalues[order] = np.clip(ordered_q, 0, 1)
    else:
        qvalues = np.array([], dtype=float)

    table_rows = []
    for record, qvalue in zip(tested, qvalues):
        i, j, a, b, valid_n, corr_value, pvalue = record
        q[i, j] = q[j, i] = float(qvalue)
        table_rows.append({
            "Variable X": a,
            "Variable Y": b,
            "Pearson r": corr_value,
            "n (paired)": valid_n,
            "p (nominal, two-sided)": pvalue,
            "q (FDR-BH)": float(qvalue),
            "q < 0.05": bool(qvalue < .05),
        })
    table = pd.DataFrame(table_rows)
    if not table.empty:
        table = table.sort_values(["q (FDR-BH)", "p (nominal, two-sided)"]).reset_index(drop=True)
    return r, p, q, n, table


def prepare_long(filtered, chosen, period, metadata):
    id_cols = ["territory_id", "territory_label"] + (["year", "month"] if period == "Monthly" else ["year"] if period == "Annual" else ["period"])
    data = filtered.melt(id_vars=id_cols, value_vars=chosen, var_name="variable", value_name="value")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    data["unit"] = data["variable"].map(lambda m: unit_text(metadata.get(m, {}).get("unit", "") or "unverified"))
    data["variable_label"] = data["variable"].map(vname)
    if period == "Monthly":
        data["time"] = pd.to_datetime(dict(year=data.year.astype(int), month=data.month.astype(int), day=1))
    elif period == "Annual":
        data["time"] = data["year"].astype(int)
    return data


def zscore_column(long):
    means = long.groupby(["territory_id", "variable"], observed=True)["value"].transform("mean")
    stds = long.groupby(["territory_id", "variable"], observed=True)["value"].transform(lambda s: s.std(ddof=0))
    return (long["value"] - means) / stds.replace(0, np.nan)


def prettify(fig, height=470):
    fig.update_layout(height=height, paper_bgcolor=PALETTE["paper"], plot_bgcolor=PALETTE["paper"],
        font=dict(family=FONT_FAMILY, size=12, color=PALETTE["text"]),
        separators=",." if LANG == "pt" else ".,",
        margin=dict(l=16, r=10, t=56, b=16), hovermode="closest",
        legend=dict(orientation="h", y=-.18, x=0, title=""))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def safe_multiselect_state(key, allowed):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in allowed]



def style_fig(fig):
    """Shared typography and locale separators for figures not using prettify()."""
    fig.update_layout(font=dict(family=FONT_FAMILY, color=PALETTE["text"]),
                      separators=",." if LANG == "pt" else ".,", paper_bgcolor=PALETTE["paper"],
                      plot_bgcolor=PALETTE["paper"])
    return fig


def section(title, lead=None):
    st.markdown(f'<div class="mb-section"><h3>{title}</h3>' + (f"<p>{lead}</p>" if lead else "") + "</div>",
                unsafe_allow_html=True)


def sidebar_step(n, text):
    st.markdown(f'<div class="mb-step"><span class="n">{n}</span>{text}</div>', unsafe_allow_html=True)


def change_language():
    choice = st.session_state.get("mb_lang_choice")
    if choice in i18n.LANGS:
        st.session_state["lang"] = choice
        st.query_params["lang"] = choice


def change_theme():
    choice = st.session_state.get("mb_theme_choice")
    if choice in ("light", "dark") and choice != THEME:
        st.session_state["mb_theme_pending"] = choice
        st.query_params["theme"] = choice
        st.query_params["lang"] = LANG


def apply_theme_js(target, reload_page):
    """Store Streamlit's active-theme choice in the browser and reload once.

    Streamlit reads its light/dark choice from localStorage at page load
    (the same setting as Settings → Theme), so it persists across visits.
    """
    value = "Dark" if target == "dark" else "Light"
    st.sidebar.container(key="mb_theme_js").html(f"""<script>
    (function() {{
      try {{
        var key = "stActiveTheme-" + window.location.pathname + "-v2";
        var stored = window.localStorage.getItem(key);
        var unset = stored === null || stored === '"System"';
        if ({'true' if reload_page else 'false'} || unset) {{
          window.localStorage.setItem(key, JSON.stringify("{value}"));
          if ({'true' if reload_page else 'false'} || "{THEME}" !== "{target}") window.location.reload();
        }}
      }} catch (e) {{ console.warn("theme switch failed", e); }}
    }})();
    </script>""", unsafe_allow_javascript=True)


def reset_filters():
    for key in list(st.session_state.keys()):
        if key not in ("lang", "mb_lang_choice", "mb_theme_choice", "mb_theme_init"):
            del st.session_state[key]


# ------------------------------- Header ------------------------------------
hero = st.container(key="mb_hero")
with hero:
    hero_text, hero_lang = st.columns([4, 1.5], vertical_alignment="top")
    hero_slot = hero_text.empty()
    with hero_lang:
        with st.container(key="mb_lang", horizontal=True, horizontal_alignment="right"):
            st.segmented_control(t("language"), list(i18n.LANGS), default=LANG, required=True,
                                 format_func=lambda k: i18n.LANGS[k], key="mb_lang_choice",
                                 on_change=change_language)
            if st.session_state.get("mb_theme_choice") not in ("light", "dark"):
                st.session_state["mb_theme_choice"] = THEME
            st.segmented_control(t("theme"), ["light", "dark"], required=True,
                                 format_func=lambda k: ":material/light_mode:" if k == "light" else ":material/dark_mode:",
                                 key="mb_theme_choice", on_change=change_theme)

# Pending switch from the toggle: store it in the browser and reload once.
if st.session_state.get("mb_theme_pending"):
    apply_theme_js(st.session_state.pop("mb_theme_pending"), reload_page=True)
elif "mb_theme_init" not in st.session_state:
    # First run of a browser session: honor ?theme=..., otherwise default to light
    # (only when the visitor never chose a theme; system dark mode is not imposed).
    st.session_state["mb_theme_init"] = True
    wanted = str(st.query_params.get("theme", "")).lower()
    if wanted in ("light", "dark") and wanted != THEME:
        apply_theme_js(wanted, reload_page=True)
    else:
        apply_theme_js("light", reload_page=False)


def render_hero(chips=()):
    chip_html = "".join(f'<span class="mb-chip">{c}</span>' for c in chips)
    hero_slot.markdown(
        f'<div class="mb-eyebrow"><span class="dot"></span>{t("brand")}</div>'
        f'<div class="mb-title">{t("app_title")}</div>'
        f'<div class="mb-subtitle">{t("app_subtitle")}</div>'
        + (f'<div class="mb-chips">{chip_html}</div>' if chips else ""),
        unsafe_allow_html=True)


render_hero()
if not (BASE / "catalog.json").exists():
    st.error(t("no_data_bundle"), icon=":material/error:")
    st.stop()
manifest = catalog()
levels = manifest.get("levels", {})
if not levels:
    st.error(t("no_levels"), icon=":material/error:"); st.stop()

# ----------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.markdown(f'<div class="mb-side-brand"><div class="logo">MB</div><div><div class="nm">{t("brand")}</div>'
                f'<div class="sb">{t("app_title")}</div></div></div>', unsafe_allow_html=True)

    # 1 · Where
    sidebar_step(1, t("sb_where"))
    level = st.selectbox(t("level"), list(levels), format_func=lambda k: level_text(k, levels), key="mb_level")
    available = [p for p in PERIODS if levels[level]["periods"].get(p, {}).get("rows", 0) > 0]
    if not available:
        st.error(t("no_level_data")); st.stop()
    areas = get_territories(level).copy()
    areas["territory_id"] = areas["territory_id"].astype(str)
    areas["territory_name"] = areas["territory_name"].astype(str)
    areas = areas.drop_duplicates(subset=["territory_id"])
    name_counts = areas.territory_name.value_counts().to_dict()
    label_of = {r.territory_id: r.territory_name + (f" · {r.territory_id}" if name_counts[r.territory_name] > 1 else "")
                for r in areas[["territory_id", "territory_name"]].itertuples(index=False)}
    options = areas.territory_id.tolist()
    selection_key = f"mb_territories_{level}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = options if len(options) <= 30 else options[:10]
    safe_multiselect_state(selection_key, options)

    def change_territories(items):
        st.session_state[selection_key] = list(items)
    pick = st.multiselect(t("units"), options, key=selection_key, placeholder=t("units_placeholder"),
        format_func=lambda i: label_of.get(i, i), help=t("units_help"))
    s1, s2 = st.columns(2)
    s1.button(t("select_all"), on_click=change_territories, args=(options,), width="stretch",
              icon=":material/done_all:", key=f"all_units_{level}")
    s2.button(t("clear"), on_click=change_territories, args=([],), width="stretch",
              icon=":material/backspace:", key=f"clear_units_{level}")

    # 2 · When
    sidebar_step(2, t("sb_when"))
    period_key = f"mb_period_{level}"
    if st.session_state.get(period_key) not in available:
        st.session_state[period_key] = available[0]
    period = st.segmented_control(t("period"), available, required=True, format_func=period_text,
                                  key=period_key, width="stretch")
    info = levels[level]["periods"][period]
    years_available = sorted(int(s) for s in info["years"] if s != "all")
    year_range = None
    if period != "Multiyear":
        if len(years_available) < 2:
            st.info(t("one_year"))
            year_range = (years_available[0], years_available[0])
        else:
            year_range = st.slider(t("year_range"), min(years_available), max(years_available),
                (min(years_available), max(years_available)), key=f"range_{level}_{period}")
        years = [y for y in years_available if year_range[0] <= y <= year_range[1]]
        if period == "Monthly":
            month_key = f"months_{level}"
            if month_key not in st.session_state:
                st.session_state[month_key] = list(range(1, 13))
            months = st.pills(t("months"), list(range(1, 13)), selection_mode="multi",
                format_func=lambda m: MONTH_NAMES[m - 1], key=month_key)
            months = sorted(months or [])

            def all_months():
                st.session_state[month_key] = list(range(1, 13))
            st.button(t("all_months"), on_click=all_months, type="tertiary", icon=":material/calendar_month:",
                      key=f"all_months_{level}")
        else:
            months = []
    else:
        years, months = ["all"], []

    # 3 · What
    sidebar_step(3, t("sb_what"))
    dictionary = get_dictionary(level)
    meta = {r.get("column_name"): r for r in dictionary if r.get("period") == period}
    metrics = list(info.get("columns", []))
    variable_key = f"mb_metrics_{level}_{period}"
    if variable_key not in st.session_state:
        st.session_state[variable_key] = list(metrics)  # ALL selected by default.
    safe_multiselect_state(variable_key, metrics)

    def change_metrics(items):
        st.session_state[variable_key] = list(items)
    climate_vars = [m for m in metrics if meta.get(m, {}).get("group") == "Atmosphere C2"]
    land_vars = [m for m in metrics if meta.get(m, {}).get("group") != "Atmosphere C2"]
    quick = st.container(horizontal=True, gap="small", key="mb_quick")
    quick.button(t("quick_climate"), key=f"clim_{level}_{period}", on_click=change_metrics, args=(climate_vars,), help=t("quick_help"))
    quick.button(t("quick_land"), key=f"land_{level}_{period}", on_click=change_metrics, args=(land_vars,), help=t("quick_help"), disabled=not land_vars)
    quick.button(t("quick_all"), key=f"every_{level}_{period}", on_click=change_metrics, args=(metrics,), help=t("quick_help"))
    selected = st.multiselect(t("variables"), metrics, key=variable_key, placeholder=t("variables_placeholder"),
        format_func=lambda m: metric_label(m, meta), help=t("variables_help"))
    st.markdown('<div class="mb-count">' + t("sb_count", units=fmt_num(len(options), ",d"), avail=len(metrics),
                sel=len(selected)) + "</div>", unsafe_allow_html=True)

    st.button(t("reset"), on_click=reset_filters, icon=":material/restart_alt:", type="tertiary", key="mb_reset")
    with st.expander(t("about_colors"), expanded=False, icon=":material/palette:"):
        st.caption(t("about_colors_text"))
        if st.checkbox(t("show_dict"), key=f"source_{level}_{period}"):
            st.dataframe(pd.DataFrame([meta.get(m, {}) for m in selected]), hide_index=True, width="stretch")

dates_text = t("kpi_multiyear") if period == "Multiyear" else f"{year_range[0]}–{year_range[1]}"
render_hero([f"<b>{level_text(level, levels)}</b>", period_text(period), dates_text,
             f"{t('kpi_units')}: {len(pick)}", f"{t('kpi_vars')}: {len(selected)}/{len(metrics)}"])

if not pick or not selected:
    st.info(t("need_selection"), icon=":material/tune:"); st.stop()
if period == "Monthly" and not months:
    st.info(t("need_month"), icon=":material/calendar_month:"); st.stop()

with st.spinner(t("reading")):
    frames = load_metrics(level, period, tuple(years), tuple(selected), tuple(pick))
if frames.empty:
    st.warning(t("no_obs")); st.stop()
frames["territory_id"] = frames["territory_id"].astype(str)
filtered = frames[frames.territory_id.isin(set(pick))].copy()
if period == "Monthly":
    filtered = filtered[filtered.month.isin(months)].copy()
if filtered.empty:
    st.warning(t("no_obs_filtered")); st.stop()
filtered["territory_label"] = filtered["territory_id"].map(label_of).fillna(filtered["territory_name"])

kpis = [
    ("🗺️", PALETTE["kpi_bg"][0], t("kpi_units"), fmt_num(filtered.territory_id.nunique(), ",d")),
    ("📊", PALETTE["kpi_bg"][1], t("kpi_obs"), fmt_num(len(filtered), ",d")),
    ("🧪", PALETTE["kpi_bg"][2], t("kpi_vars"), f"{len(selected)} / {len(metrics)}"),
    ("🗓️", PALETTE["kpi_bg"][3], t("kpi_dates"), dates_text),
]
st.markdown('<div class="mb-kpis">' + "".join(
    f'<div class="mb-kpi"><div class="ic" style="background:{bg}">{ic}</div><div><div class="lb">{lb}</div>'
    f'<div class="vl">{vl}</div></div></div>' for ic, bg, lb, vl in kpis) + "</div>", unsafe_allow_html=True)

time_cols = ["year", "month"] if period == "Monthly" else ["year"] if period == "Annual" else ["period"]
axis_names = {"year": t("year"), "month": t("month"), "period": t("period"),
              "territory_label": t("territory"), "territory_id": "ID"}

tab_maps, tab_series, tab_reg, tab_corr, tab_data = st.tabs([
    f":material/map: {t('tab_maps')}", f":material/show_chart: {t('tab_series')}",
    f":material/scatter_plot: {t('tab_regression')}", f":material/grid_on: {t('tab_corr')}",
    f":material/table_view: {t('tab_data')}"])

# ---------------------------- Two-period maps ------------------------------
with tab_maps:
    section(t("maps_title"), t("maps_lead"))
    map_controls = st.columns([2.4, 1.0, 1.0, 1.2], vertical_alignment="bottom")
    with map_controls[0]:
        map_var = st.selectbox(t("map_var"), selected, format_func=lambda m: metric_label(m, meta), key=f"map_metric_{level}_{period}")
    if period != "Multiyear":
        with map_controls[1]:
            year_a = st.selectbox(t("year_a"), years, index=0, key=f"year_a_{level}_{period}")
        with map_controls[2]:
            year_b = st.selectbox(t("year_b"), years, index=len(years) - 1, key=f"year_b_{level}_{period}")
        with map_controls[3]:
            if period == "Monthly":
                map_month = st.selectbox(t("map_month"), months, index=len(months) - 1,
                    format_func=lambda m: MONTH_NAMES[m - 1], key=f"map_month_{level}")
            else:
                st.caption(t("same_scale"))
                map_month = None
    else:
        year_a = year_b = None
        map_month = None
        with map_controls[1]:
            st.caption(t("multiyear_note"))

    layer_a = observations_at(filtered, year_a, map_month, period, map_var)
    layer_b = observations_at(filtered, year_b, map_month, period, map_var)
    geo = get_geojson(level)
    if geo is None:
        st.warning(t("no_geojson"))
    else:
        ids_in_view = set(layer_a.territory_id) | set(layer_b.territory_id)
        selected_features = []
        skipped = 0
        for feature in geo.get("features", []):
            fid = str(feature.get("properties", {}).get("territory_id", ""))
            if fid not in ids_in_view:
                continue
            geometry = polygonal_geometry(feature.get("geometry"))
            if geometry is None:
                skipped += 1
                continue
            props = dict(feature.get("properties", {}))
            props["territory_id"] = fid
            selected_features.append({"type": "Feature", "properties": props, "geometry": geometry})
        if skipped:
            st.warning(t("skipped_features", n=skipped))
        if len(selected_features) > 450:
            st.info(t("map_limit"))
            selected_features = selected_features[:450]
        if not selected_features:
            st.warning(t("no_geometries"))
        else:
            displayed_ids = {f["properties"]["territory_id"] for f in selected_features}
            layer_a = layer_a[layer_a.territory_id.isin(displayed_ids)].copy()
            layer_b = layer_b[layer_b.territory_id.isin(displayed_ids)].copy()
            center, zoom = geo_bounds(selected_features)
            v1 = clean_value(layer_a, map_var).dropna().to_numpy(dtype=float)
            v2 = clean_value(layer_b, map_var).dropna().to_numpy(dtype=float)
            family = palette_family(map_var, meta)
            vmin, vmax = limits(v1, v2, symmetric=family == "diverging")
            palette = map_scale(map_var, meta)
            unit_label = metric_label(map_var, meta)
            month_suffix = f" · {MONTH_NAMES[map_month - 1]}" if map_month else ""
            if period == "Multiyear":
                fig = map_figure(layer_a, map_var, selected_features, center, zoom, palette, vmin, vmax, unit_label, height=520)
                st.plotly_chart(fig, width="stretch", key="single_multiyear_map")
            else:
                view = st.segmented_control(t("viz"), ["side", "change"], required=True, default="side",
                    format_func=lambda v: f":material/compare: {t('side_by_side')}" if v == "side"
                    else f":material/trending_up: {t('change_tab')}",
                    key="mb_map_view", label_visibility="collapsed")
                if view == "side":
                    left_map, right_map = st.columns(2, gap="small")
                    with left_map:
                        st.markdown(f'<div class="mb-label">{year_a}{month_suffix}</div>', unsafe_allow_html=True)
                        fig = map_figure(layer_a, map_var, selected_features, center, zoom, palette, vmin, vmax, unit_label)
                        st.plotly_chart(fig, width="stretch", key="map_a")
                    with right_map:
                        st.markdown(f'<div class="mb-label">{year_b}{month_suffix}</div>', unsafe_allow_html=True)
                        fig = map_figure(layer_b, map_var, selected_features, center, zoom, palette, vmin, vmax, unit_label)
                        st.plotly_chart(fig, width="stretch", key="map_b")
                    st.caption(t("same_limits"))
                else:
                    paired = layer_a[["territory_id", "territory_label", map_var]].merge(
                        layer_b[["territory_id", map_var]], on="territory_id", suffixes=("_a", "_b"))
                    paired["change"] = clean_value(paired, f"{map_var}_b") - clean_value(paired, f"{map_var}_a")
                    paired["change_pct"] = (100 * paired["change"] / clean_value(paired, f"{map_var}_a").abs().replace(0, np.nan))
                    delta_map = paired[["territory_id", "territory_label", "change"]].rename(columns={"change": "Δ B − A"})
                    diff_bounds = limits(delta_map["Δ B − A"].dropna().to_numpy(), symmetric=True)
                    if not paired.empty:
                        q1, q2, q3 = st.columns(3)
                        q1.metric(t("paired_terr"), str(paired.change.notna().sum()))
                        q2.metric(t("mean_delta"), fmt_num(paired.change.mean()) if paired.change.notna().any() else t("n_a"))
                        q3.metric(t("median_delta"), fmt_num(paired.change.median()) if paired.change.notna().any() else t("n_a"))
                    dm, rk = st.columns([1.65, 1], gap="medium")
                    with dm:
                        st.markdown(f'<div class="mb-label">{t("abs_change", a=year_a, b=year_b)}{month_suffix}</div>', unsafe_allow_html=True)
                        fig = map_figure(delta_map, "Δ B − A", selected_features, center, zoom, DIVERGING, *diff_bounds,
                                         t("change_unit", unit=unit_text(meta.get(map_var, {}).get("unit", ""))))
                        st.plotly_chart(fig, width="stretch", key="map_delta")
                    with rk:
                        st.markdown(f'<div class="mb-label">{t("terr_changes")}</div>', unsafe_allow_html=True)
                        st.caption(t("change_note"))
                        display = paired[["territory_label", f"{map_var}_a", f"{map_var}_b", "change", "change_pct"]].copy()
                        display.columns = [t("col_territory"), str(year_a), str(year_b), t("col_abs"), t("col_pct")]
                        st.dataframe(display.sort_values(t("col_abs"), ascending=False), width="stretch",
                            hide_index=True, height=360,
                            column_config={t("col_abs"): st.column_config.NumberColumn(format="%.3f"),
                                           t("col_pct"): st.column_config.NumberColumn(format="%.1f%%")})
                    st.caption(t("pct_note"))

# ------------------------- Time series explorer ---------------------------
with tab_series:
    section(t("series_title"), t("series_lead"))
    chart_a, chart_b, chart_c = st.columns([1.65, 1, 1])
    with chart_a:
        chart_key = f"focus_{level}_{period}"
        if chart_key not in st.session_state:
            st.session_state[chart_key] = list(selected[:min(6, len(selected))])
        safe_multiselect_state(chart_key, selected)
        focus = st.multiselect(t("focus_vars"), selected, key=chart_key,
            format_func=lambda m: metric_label(m, meta), help=t("focus_vars_help"))
    with chart_b:
        if period != "Multiyear":
            chart_mode = st.selectbox(t("viz"), i18n.CHART_MODES, format_func=lambda k: t(f"mode_{k}"),
                                      key=f"chart_mode_{level}_{period}_v4")
        else:
            chart_mode = "multiyear"
            st.caption(t("multiyear_cmp"))
    with chart_c:
        if period != "Multiyear":
            area_key = f"focus_area_{level}_{period}"
            if area_key not in st.session_state:
                st.session_state[area_key] = pick[:min(4, len(pick))]
            safe_multiselect_state(area_key, pick)
            chart_territories = st.multiselect(t("focus_terr"), pick, format_func=lambda i: label_of.get(i, i),
                key=area_key)
        else:
            chart_territories = pick[:min(10, len(pick))]
    if not focus:
        st.info(t("need_focus_vars"), icon=":material/tune:")
    elif not chart_territories:
        st.info(t("need_focus_terr"), icon=":material/tune:")
    else:
        plot_long = prepare_long(filtered[filtered.territory_id.isin(chart_territories)], focus, period, meta)
        colors = {vname(k): v for k, v in variable_color_map(focus, meta).items()}
        series_labels = {"value": t("value"), "territory_label": t("territory"), "variable_label": t("variable"),
                         "time": t("date"), "unit": "", "year": t("year"), "month": t("month")}
        if plot_long.empty:
            st.info(t("no_valid"))
        elif period == "Multiyear":
            fig = px.bar(plot_long, x="territory_label", y="value", color="variable_label", barmode="group",
                color_discrete_map=colors, facet_row="unit", height=max(410, 240 * plot_long.unit.nunique()),
                labels=series_labels)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(prettify(fig, max(410, 240 * plot_long.unit.nunique())), width="stretch")
        elif chart_mode == "zheat":
            plot_long["z"] = zscore_column(plot_long)
            # A separate heatmap per territory prevents averaging z-scores across unrelated regions.
            panels = st.columns(min(2, len(chart_territories)))
            for idx, territory in enumerate(chart_territories):
                part = plot_long[plot_long.territory_id == territory].copy()
                if part.empty:
                    continue
                if period == "Monthly":
                    part["time_label"] = part["time"].dt.strftime("%Y-%m")
                else:
                    part["time_label"] = part["year"].astype(str)
                matrix = part.pivot_table(index="variable_label", columns="time_label", values="z", aggfunc="mean")
                if matrix.empty:
                    continue
                fig = px.imshow(matrix, color_continuous_scale=DIVERGING, zmin=-3, zmax=3, aspect="auto",
                    labels={"color": "Z-score", "x": t("date"), "y": t("variable")})
                fig.update_xaxes(nticks=12)
                fig.update_layout(title=label_of.get(territory, str(territory)), height=max(270, 80 + 42 * len(matrix)),
                    margin=dict(l=5, r=5, t=55, b=25), coloraxis_colorbar=dict(title="z"))
                with panels[idx % len(panels)]:
                    st.plotly_chart(style_fig(fig), width="stretch", key=f"zheat_{idx}_{level}_{period}")
            st.caption(t("zheat_note"))
        else:
            if chart_mode == "ztraj":
                plot_long["plot_value"] = zscore_column(plot_long)
                y_label = t("y_z")
            elif chart_mode == "index":
                baseline = plot_long.sort_values("time").groupby(["territory_id", "variable"], observed=True)["value"].transform("first")
                plot_long["plot_value"] = 100 * plot_long["value"] / baseline.replace(0, np.nan)
                y_label = t("y_index")
            elif chart_mode == "pct":
                baseline = plot_long.sort_values("time").groupby(["territory_id", "variable"], observed=True)["value"].transform("first")
                plot_long["plot_value"] = 100 * (plot_long["value"] - baseline) / baseline.abs().replace(0, np.nan)
                y_label = t("y_pct")
            else:
                plot_long["plot_value"] = plot_long["value"]
                y_label = t("y_raw")
            labels = dict(series_labels, plot_value=y_label)
            if chart_mode == "raw":
                nu = plot_long.unit.nunique()
                fig = px.line(plot_long, x="time", y="plot_value", color="variable_label", line_dash="territory_label",
                    facet_row="unit", facet_row_spacing=min(.065, .24 / max(1, nu - 1)),
                    color_discrete_map=colors, hover_data=["territory_label", "value"], labels=labels)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                h = max(420, 235 * nu)
            else:
                fig = px.line(plot_long, x="time", y="plot_value", color="variable_label", line_dash="territory_label",
                    color_discrete_map=colors, hover_data=["territory_label", "value", "unit"], labels=labels)
                fig.add_hline(y=100 if chart_mode == "index" else 0, line_width=1,
                    line_dash="dot", line_color=PALETTE["ref"], opacity=.85)
                if chart_mode == "ztraj":
                    # Diverging marker colors encode the signed anomaly at each date;
                    # line colors still identify the metric across all plots.
                    for (_, _), points in plot_long.groupby(["territory_id", "variable"], observed=True):
                        fig.add_trace(go.Scatter(x=points["time"], y=points["plot_value"],
                            mode="markers", marker=dict(size=5, opacity=.8,
                                color=points["plot_value"], coloraxis="coloraxis"),
                            hoverinfo="skip", showlegend=False))
                    fig.update_layout(coloraxis=dict(colorscale=DIVERGING, cmin=-2.5, cmax=2.5,
                        colorbar=dict(title=t("signed_z"), thickness=12, len=.65)))
                h = 500
            st.plotly_chart(prettify(fig, h), width="stretch")
            st.caption(t("series_note"))

# ----------------------- Multivariate relationships -----------------------
with tab_reg:
    section(t("reg_title"), t("reg_lead"))
    if len(selected) >= 2:
        rcols = st.columns([1.2, 1.2, 1.0, 1.0])
        xvar = rcols[0].selectbox(t("x_var"), selected, index=0, format_func=lambda m: metric_label(m, meta), key=f"x_{level}_{period}")
        yvar = rcols[1].selectbox(t("y_var"), selected, index=min(1, len(selected) - 1), format_func=lambda m: metric_label(m, meta), key=f"y_{level}_{period}")
        choices = ["None"] + [m for m in selected if m not in (xvar, yvar)]
        hue = rcols[2].selectbox(t("hue_var"), choices, format_func=lambda m: t("hue_none") if m == "None" else metric_label(m, meta), key=f"hue_{level}_{period}")
        sz = rcols[3].selectbox(t("size_var"), choices, format_func=lambda m: t("size_none") if m == "None" else metric_label(m, meta), key=f"size_{level}_{period}")
        if xvar == yvar:
            st.info(t("need_diff_xy"))
        else:
            basic_cols = ["territory_label", "territory_id", xvar, yvar]
            plot_cols = list(dict.fromkeys(basic_cols + time_cols + ([hue] if hue != "None" else []) + ([sz] if sz != "None" else [])))
            scatter = filtered[plot_cols].copy()
            for m in [xvar, yvar] + ([hue] if hue != "None" else []) + ([sz] if sz != "None" else []):
                scatter[m] = pd.to_numeric(scatter[m], errors="coerce")
            scatter = scatter.replace([np.inf, -np.inf], np.nan).dropna(subset=[xvar, yvar])
            if sz != "None":
                scatter["marker_size"] = scatter[sz].clip(lower=0).fillna(0.0)
                if not scatter["marker_size"].gt(0).any():
                    scatter["marker_size"] = 1.0
                    st.info(t("no_positive_size"))
            scatter_labels = dict(axis_names, **{xvar: metric_label(xvar, meta), yvar: metric_label(yvar, meta)})
            if hue != "None":
                scatter_labels[hue] = metric_label(hue, meta)
            if sz != "None":
                scatter_labels[sz] = metric_label(sz, meta)
                scatter_labels["marker_size"] = metric_label(sz, meta)
            leg, statscol = st.columns([2.25, 1], gap="large")
            with leg:
                if hue == "None":
                    fig = px.scatter(scatter, x=xvar, y=yvar, color="territory_label",
                        color_discrete_sequence=SERIES,
                        size="marker_size" if sz != "None" else None, hover_data=time_cols,
                        labels=scatter_labels, opacity=.65)
                else:
                    fig = px.scatter(scatter.dropna(subset=[hue]), x=xvar, y=yvar, color=hue,
                        color_continuous_scale=DIVERGING if palette_family(hue, meta) == "diverging" else map_scale(hue, meta),
                        size="marker_size" if sz != "None" else None,
                        hover_data=["territory_label"] + time_cols,
                        labels=scatter_labels, opacity=.68)
                fit = linear_regression(scatter, xvar, yvar)
                if fit:
                    xx = np.linspace(fit["xmin"], fit["xmax"], 100)
                    yy = fit["slope"] * xx + fit["intercept"]
                    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines",
                        name=t("ols_line", r2=fmt_num(fit["r2"], ".3f"), p=format_p(fit["p_value"])),
                        line=dict(color=PALETTE["fit"], width=3, dash="dash"), hovertemplate=t("ols_hover") + "<extra></extra>"))
                st.plotly_chart(prettify(fig, 500), width="stretch")
            with statscol:
                st.markdown(f'<div class="mb-label">{t("pooled_ols")}</div>', unsafe_allow_html=True)
                if fit:
                    m1, m2 = st.columns(2)
                    m1.metric(t("r2"), fmt_num(fit["r2"], ".3f"))
                    m2.metric(t("pearson"), fmt_num(fit["r"], "+.3f"))
                    st.metric(t("pval"), format_p(fit["p_value"]))
                    m3, m4 = st.columns(2)
                    m3.metric(t("slope"), fmt_num(fit["slope"], ",.4g"))
                    m4.metric(t("n_rows"), fmt_num(fit["n"], ",d"))
                    st.caption(f"ŷ = {fmt_num(fit['intercept'], '.4g')} + ({fmt_num(fit['slope'], '.4g')}) × X")
                    st.caption(t("ci95", lo=fmt_num(fit["ci95_low"], ".4g"), hi=fmt_num(fit["ci95_high"], ".4g")))
                    st.caption(t("h0"))
                    if fit["p_value"] < .05:
                        st.caption(t("p_sig"))
                else:
                    st.info(t("stats_unavailable"))
                st.warning(t("inference_warning"), icon=":material/warning:")
            if scatter.territory_id.nunique() > 1:
                with st.expander(t("reg_by_terr"), expanded=False, icon=":material/table_chart:"):
                    rows = []
                    for tid, part in scatter.groupby("territory_id"):
                        fitted = linear_regression(part, xvar, yvar)
                        if fitted:
                            rows.append({t("col_territory"): label_of.get(tid, tid), t("col_n"): fitted["n"],
                                         t("col_slope"): fitted["slope"], t("col_intercept"): fitted["intercept"],
                                         t("pearson"): fitted["r"], "R²": fitted["r2"],
                                         t("col_p"): fitted["p_value"],
                                         t("col_ci_lo"): fitted["ci95_low"],
                                         t("col_ci_hi"): fitted["ci95_high"]})
                    if rows:
                        st.dataframe(pd.DataFrame(rows).sort_values("R²", ascending=False), hide_index=True, width="stretch")
                    else:
                        st.info(t("not_enough_terr"))
            if len(selected) >= 3:
                with st.expander(t("explore_3d"), expanded=False, icon=":material/view_in_ar:"):
                    zopts = [m for m in selected if m not in (xvar, yvar)]
                    zvar = st.selectbox(t("z_var"), zopts, format_func=lambda m: metric_label(m, meta), key=f"z_{level}_{period}")
                    subset = filtered[["territory_label", xvar, yvar, zvar] + time_cols].copy()
                    for m in (xvar, yvar, zvar):
                        subset[m] = pd.to_numeric(subset[m], errors="coerce")
                    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[xvar, yvar, zvar])
                    if not subset.empty:
                        fig = px.scatter_3d(subset, x=xvar, y=yvar, z=zvar, color="territory_label",
                            color_discrete_sequence=SERIES, opacity=.75,
                            hover_data=time_cols, height=570,
                            labels=dict(scatter_labels, **{zvar: metric_label(zvar, meta)}))
                        fig.update_layout(margin=dict(l=0, r=0, t=25, b=0), legend_title_text="")
                        st.plotly_chart(style_fig(fig), width="stretch")
    else:
        st.info(t("need_two_vars"), icon=":material/tune:")

# --------------------------- Correlations ---------------------------------
with tab_corr:
    section(t("corr_title"), t("corr_lead"))
    corr_cols = st.columns([2, 1], gap="large")
    with corr_cols[0]:
        correlation_choices = selected[:min(len(selected), 24)]
        corr_key = f"corr_{level}_{period}"
        if corr_key not in st.session_state:
            st.session_state[corr_key] = correlation_choices[:min(10, len(correlation_choices))]
        safe_multiselect_state(corr_key, selected)
        corr_sel = st.multiselect(t("corr_metrics"), selected,
            format_func=lambda m: metric_label(m, meta), key=corr_key)
        if len(corr_sel) >= 2:
            rmat, pmat, qmat, nmat, pair_table = correlation_tests(filtered, corr_sel)
            if not pair_table.empty:
                k = len(corr_sel)
                hdata = np.empty((k, k, 3), dtype=object)
                labels = np.empty((k, k), dtype=object)
                for i in range(k):
                    for j in range(k):
                        hdata[i, j] = [format_p(pmat[i, j]), format_p(qmat[i, j]), str(nmat[i, j])]
                        if np.isfinite(rmat[i, j]) and k <= 12:
                            significant = i != j and np.isfinite(qmat[i, j]) and qmat[i, j] < .05
                            labels[i, j] = fmt_num(rmat[i, j], ".2f") + ("*" if significant else "")
                        else:
                            labels[i, j] = ""
                axis = [vname(m) for m in corr_sel]
                heatmap = go.Heatmap(
                    z=rmat, x=axis, y=axis, zmin=-1, zmax=1, zmid=0,
                    colorscale=DIVERGING, colorbar=dict(title=t("pearson")),
                    customdata=hdata, text=labels, hovertemplate=t("corr_hover"),
                )
                heatmap.texttemplate = "%{text}" if k <= 12 else None
                fig = go.Figure(data=[heatmap])
                fig.update_layout(height=max(460, k * 38 + 200), margin=dict(l=10, r=10, t=35, b=10))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig), width="stretch")
                st.caption(t("corr_note"))
                with st.expander(t("pair_table"), expanded=False, icon=":material/table_chart:"):
                    shown = pair_table.copy()
                    shown["Variable X"] = shown["Variable X"].map(vname)
                    shown["Variable Y"] = shown["Variable Y"].map(vname)
                    shown = shown.rename(columns=t("pair_cols"))
                    st.dataframe(shown, width="stretch", hide_index=True)
                    st.download_button(t("download_corr"), shown.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"mapbiomas_correlations_{level}_{period.lower()}.csv", mime="text/csv",
                        icon=":material/download:")
            else:
                st.info(t("no_pairs"))
        else:
            st.info(t("need_two_corr"), icon=":material/tune:")
    with corr_cols[1]:
        with st.container(border=True):
            st.markdown(f'<div class="mb-label">{t("cross_analysis")}</div>', unsafe_allow_html=True)
            st.caption(t("corr_c1"))
            st.caption(t("corr_c2"))
            st.caption(t("corr_c3"))
        if period == "Monthly" and len(selected):
            st.markdown(f'<div class="mb-label">{t("seasonal")}</div>', unsafe_allow_html=True)
            seas_var = st.selectbox(t("seasonal_var"), selected, format_func=lambda m: metric_label(m, meta), key=f"season_{level}")
            seas = filtered[["territory_label", "month", seas_var]].copy()
            seas[seas_var] = pd.to_numeric(seas[seas_var], errors="coerce")
            seas = seas.groupby(["territory_label", "month"], as_index=False)[seas_var].mean()
            fig = px.line(seas, x="month", y=seas_var, color="territory_label",
                color_discrete_sequence=SERIES, markers=True,
                labels=dict(axis_names, **{seas_var: metric_label(seas_var, meta)}))
            fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTH_NAMES)
            st.plotly_chart(prettify(fig, 360), width="stretch")
            st.caption(t("seasonal_note"))

# --------------------------- Data / provenance ----------------------------
with tab_data:
    section(t("data_title"), t("data_lead"))
    cols = ["territory_id", "territory_name"] + time_cols + list(selected)
    displayed = filtered[cols].copy()
    st.dataframe(displayed, width="stretch", hide_index=True, height=380,
                 column_config={m: st.column_config.NumberColumn(metric_label(m, meta)) for m in selected})
    st.download_button(t("download_data"), displayed.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"mapbiomas_atmosphere_{level}_{period.lower()}_filtered.csv", mime="text/csv",
        type="primary", icon=":material/download:")
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    with st.expander(t("methods_title"), expanded=False, icon=":material/menu_book:"):
        descriptions = pd.DataFrame([r for r in dictionary if r.get("period") == period and r.get("column_name") in selected])
        st.dataframe(descriptions, hide_index=True, width="stretch")
        st.markdown(t("methods_md"))

st.markdown(f'<div class="mb-footer">{t("footer")}</div>', unsafe_allow_html=True)
