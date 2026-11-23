"""MapBiomas Atmosphere Explorer — analytical dashboard (v3).

Reads the original XLSX-derived, year-partitioned Parquet tables and genuine
Earth Engine GeoJSON geometries. No scientific data are modified or estimated.
Visualization palettes adopt documented MapBiomas brand colors, while numerical
maps use continuous scales appropriate to their variables.
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

BASE = Path(__file__).resolve().parent / "data"
PERIODS = ["Annual", "Monthly", "Multiyear"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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

st.set_page_config(page_title="MapBiomas Atmosphere Explorer", page_icon="🌎", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {--mb-ink: #163548; --mb-accent: #62A5BB;}
.block-container {padding-top: 1.15rem; padding-bottom: 2.1rem; max-width: 1700px;}
[data-testid="stSidebar"] {min-width: 370px !important; width: 370px !important;
  background: #F2F7F8;}
[data-testid="stSidebar"] > div:first-child {width: 370px !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .7rem;}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {min-height: 45px;}
[data-testid="stSidebar"] .stButton button {font-size: .86rem;}
h1,h2,h3 {letter-spacing: -.025em; color: #163548;}
[data-testid="stMetricValue"] {font-size: 1.4rem;}
@media(max-width: 750px) {
  [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
    min-width: min(370px, 95vw) !important; width: min(370px, 95vw) !important; }
}
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(show_spinner="Reading selected data...", max_entries=16)
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
    return f"{m.replace('_', ' ')} [{unit}]" if unit else m.replace("_", " ")


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
        map_style="carto-positron", center=center, zoom=zoom, opacity=.83,
        labels={metric: label},
    )
    fig.update_traces(marker_line_width=.45, marker_line_color="rgba(34,60,75,.58)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title=dict(text="", side="right"), thickness=13, len=.73, x=1.01),
        font=dict(family="Arial", color="#234153"))
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
        return "n/a"
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
    data["unit"] = data["variable"].map(lambda m: metadata.get(m, {}).get("unit", "") or "unverified")
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
    fig.update_layout(height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="#213D50"),
        margin=dict(l=16, r=10, t=56, b=16), hovermode="closest",
        legend=dict(orientation="h", y=-.18, x=0, title=""))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(100,140,158,0.14)", zeroline=False)
    return fig


def safe_multiselect_state(key, allowed):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in allowed]


# ----------------------------- Sidebar -------------------------------------
st.title("🌎 MapBiomas Atmosphere Explorer")
st.caption("Brazil · Atmospheric indicators, land cover and fire/water themes · XLSX-derived data · up to 2025")
if not (BASE / "catalog.json").exists():
    st.error("Dataset not bundled. Keep the original data/ folder when replacing app.py.")
    st.stop()
manifest = catalog()
levels = manifest.get("levels", {})
if not levels:
    st.error("The catalog does not contain territorial levels."); st.stop()

with st.sidebar:
    st.header("Explore the data")
    level = st.selectbox("Territorial level", list(levels), format_func=lambda k: levels[k]["label"])
    available = [p for p in PERIODS if levels[level]["periods"].get(p, {}).get("rows", 0) > 0]
    if not available:
        st.error("No data for this level."); st.stop()
    period = st.radio("Temporal resolution", available, horizontal=True)
    info = levels[level]["periods"][period]
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
    s1, s2 = st.columns(2)
    s1.button("Select all units", on_click=change_territories, args=(options,), use_container_width=True, key=f"all_units_{level}")
    s2.button("Clear units", on_click=change_territories, args=([],), use_container_width=True, key=f"clear_units_{level}")
    pick = st.multiselect("Territorial units", options, key=selection_key,
        format_func=lambda i: label_of.get(i, i), help="Territory IDs correspond to the genuine GEE geometries.")

    years_available = sorted(int(s) for s in info["years"] if s != "all")
    if period != "Multiyear":
        if len(years_available) < 2:
            st.info("This period has only one year of data.")
            year_range = (years_available[0], years_available[0])
        else:
            year_range = st.slider("Year interval", min(years_available), max(years_available),
                (min(years_available), max(years_available)), key=f"range_{level}_{period}")
        years = [y for y in years_available if year_range[0] <= y <= year_range[1]]
        if period == "Monthly":
            months = st.multiselect("Months to include", list(range(1, 13)), default=list(range(1, 13)),
                format_func=lambda m: MONTH_NAMES[m - 1], key=f"months_{level}")
        else:
            months = []
    else:
        years, months = ["all"], []

    dictionary = get_dictionary(level)
    meta = {r.get("column_name"): r for r in dictionary if r.get("period") == period}
    metrics = list(info.get("columns", []))
    variable_key = f"mb_metrics_{level}_{period}"
    if variable_key not in st.session_state:
        st.session_state[variable_key] = list(metrics)  # ALL selected by default.
    safe_multiselect_state(variable_key, metrics)
    def change_metrics(items):
        st.session_state[variable_key] = list(items)
    a, b, c = st.columns(3)
    a.button("Climate", key=f"clim_{level}_{period}", on_click=change_metrics,
        args=([m for m in metrics if meta.get(m, {}).get("group") == "Atmosphere C2"],), use_container_width=True)
    b.button("Land/themes", key=f"land_{level}_{period}", on_click=change_metrics,
        args=([m for m in metrics if meta.get(m, {}).get("group") != "Atmosphere C2"],), use_container_width=True)
    c.button("All variables", key=f"every_{level}_{period}", on_click=change_metrics, args=(metrics,), use_container_width=True)
    selected = st.multiselect("Variables (all preselected)", metrics, key=variable_key,
        format_func=lambda m: metric_label(m, meta), help="All variables are selected by default; choose a smaller chart subset in the main panel for legibility.")
    st.caption(f"{len(options):,} territorial units · {len(metrics)} variables available · {len(selected)} selected")
    with st.expander("About colors and sources", expanded=False):
        st.caption("Brand colors follow the MapBiomas identity manual. Land-cover categories and continuous atmospheric measurements use different legend conventions. Numerical map scales are consistent across compared years.")
        if st.checkbox("Show selected metric dictionary", key=f"source_{level}_{period}"):
            st.dataframe(pd.DataFrame([meta.get(m, {}) for m in selected]), hide_index=True, use_container_width=True)

if not pick or not selected:
    st.info("Select at least one territory and one variable in the left sidebar."); st.stop()
if period == "Monthly" and not months:
    st.info("Select at least one month in the sidebar."); st.stop()

frames = load_metrics(level, period, tuple(years), tuple(selected), tuple(pick))
if frames.empty:
    st.warning("No observations match this selection."); st.stop()
frames["territory_id"] = frames["territory_id"].astype(str)
filtered = frames[frames.territory_id.isin(set(pick))].copy()
if period == "Monthly":
    filtered = filtered[filtered.month.isin(months)].copy()
if filtered.empty:
    st.warning("No observations remain after filtering."); st.stop()
filtered["territory_label"] = filtered["territory_id"].map(label_of).fillna(filtered["territory_name"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Territorial units", f"{filtered.territory_id.nunique():,}")
c2.metric("Observations", f"{len(filtered):,}")
c3.metric("Selected variables", f"{len(selected)} / {len(metrics)}")
c4.metric("Date coverage", "Multiyear" if period == "Multiyear" else f"{year_range[0]}–{year_range[1]}")

# ---------------------------- Two-period maps ------------------------------
st.divider()
st.subheader("Territorial comparison · two dates")
map_controls = st.columns([2.4, 1.0, 1.0, 1.2])
with map_controls[0]:
    map_var = st.selectbox("Map variable", selected, format_func=lambda m: metric_label(m, meta), key=f"map_metric_{level}_{period}")
if period != "Multiyear":
    with map_controls[1]:
        year_a = st.selectbox("Year A", years, index=0, key=f"year_a_{level}_{period}")
    with map_controls[2]:
        year_b = st.selectbox("Year B", years, index=len(years) - 1, key=f"year_b_{level}_{period}")
    with map_controls[3]:
        if period == "Monthly":
            map_month = st.selectbox("Month (both maps)", months, index=len(months) - 1,
                format_func=lambda m: MONTH_NAMES[m - 1], key=f"map_month_{level}")
        else:
            st.caption("Same variable · shared color scale")
            map_month = None
else:
    year_a = year_b = None
    map_month = None
    with map_controls[1]:
        st.caption("Multiyear values have no individual year.")

layer_a = observations_at(filtered, year_a, map_month, period, map_var)
layer_b = observations_at(filtered, year_b, map_month, period, map_var)
geo = get_geojson(level)
if geo is None:
    st.warning("This territorial level has no GEE GeoJSON export. Other charts remain available.")
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
        st.warning(f"{skipped} features have no polygonal coordinates and were omitted from the map.")
    if len(selected_features) > 450:
        st.info("The map displays up to 450 territories for browser performance. Tables and graphs retain all selected data.")
        selected_features = selected_features[:450]
    if not selected_features:
        st.warning("No matching polygon geometries. Check the XLSX territory IDs and GEE GeoJSON IDs.")
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
        if period == "Multiyear":
            fig = map_figure(layer_a, map_var, selected_features, center, zoom, palette, vmin, vmax, unit_label, height=520)
            st.plotly_chart(fig, use_container_width=True, key="single_multiyear_map")
        else:
            tab1, tab2 = st.tabs(["Side-by-side maps", "Change map and rankings"])
            with tab1:
                left_map, right_map = st.columns(2, gap="small")
                with left_map:
                    st.markdown(f"**{year_a}**" + (f" · {MONTH_NAMES[map_month - 1]}" if map_month else ""))
                    fig = map_figure(layer_a, map_var, selected_features, center, zoom, palette, vmin, vmax, unit_label)
                    st.plotly_chart(fig, use_container_width=True, key="map_a")
                with right_map:
                    st.markdown(f"**{year_b}**" + (f" · {MONTH_NAMES[map_month - 1]}" if map_month else ""))
                    fig = map_figure(layer_b, map_var, selected_features, center, zoom, palette, vmin, vmax, unit_label)
                    st.plotly_chart(fig, use_container_width=True, key="map_b")
                st.caption("Both years use exactly the same numerical color limits. Map shading represents territorial summary statistics, not 30 m land-cover pixels.")
            with tab2:
                paired = layer_a[["territory_id", "territory_label", map_var]].merge(
                    layer_b[["territory_id", map_var]], on="territory_id", suffixes=("_a", "_b"))
                paired["change"] = clean_value(paired, f"{map_var}_b") - clean_value(paired, f"{map_var}_a")
                paired["change_pct"] = (100 * paired["change"] / clean_value(paired, f"{map_var}_a").abs().replace(0, np.nan))
                delta_map = paired[["territory_id", "territory_label", "change"]].rename(columns={"change": "Δ B − A"})
                diff_bounds = limits(delta_map["Δ B − A"].dropna().to_numpy(), symmetric=True)
                dm, rk = st.columns([1.65, 1], gap="medium")
                with dm:
                    st.markdown(f"**Absolute change · {year_b} − {year_a}**")
                    fig = map_figure(delta_map, "Δ B − A", selected_features, center, zoom, DIVERGING, *diff_bounds, f"Change [{meta.get(map_var, {}).get('unit', '')}]")
                    st.plotly_chart(fig, use_container_width=True, key="map_delta")
                with rk:
                    st.markdown("**Territorial changes**")
                    st.caption("Positive changes are warm; negative changes are blue. Direction does not imply benefit or harm.")
                    display = paired[["territory_label", f"{map_var}_a", f"{map_var}_b", "change", "change_pct"]].copy()
                    display.columns = ["Territory", str(year_a), str(year_b), "Δ absolute", "Δ %"]
                    st.dataframe(display.sort_values("Δ absolute", ascending=False), use_container_width=True,
                        hide_index=True, height=335,
                        column_config={"Δ absolute": st.column_config.NumberColumn(format="%.3f"),
                                       "Δ %": st.column_config.NumberColumn(format="%.1f%%")})
                if not paired.empty:
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Paired territories", str(paired.change.notna().sum()))
                    q2.metric("Mean absolute Δ", f"{paired.change.mean():,.2f}" if paired.change.notna().any() else "n/a")
                    q3.metric("Median absolute Δ", f"{paired.change.median():,.2f}" if paired.change.notna().any() else "n/a")
                st.caption("Percentage change = (B − A)/|A| × 100; undefined where A = 0. Summary means are unweighted across selected territories.")

# ------------------------- Time series explorer ---------------------------
st.divider()
st.subheader("Time series · multivariate exploration")
chart_a, chart_b, chart_c = st.columns([1.65, 1, 1])
with chart_a:
    chart_key = f"focus_{level}_{period}"
    if chart_key not in st.session_state:
        st.session_state[chart_key] = list(selected[:min(6, len(selected))])
    safe_multiselect_state(chart_key, selected)
    focus = st.multiselect("Variables in main chart", selected, key=chart_key,
        format_func=lambda m: metric_label(m, meta), help="The sidebar selects all available metrics for analysis. Here select those to draw together without overplotting.")
with chart_b:
    if period != "Multiyear":
        chart_modes = ["Diverging z-score heatmap", "Z-score trajectories", "Raw values by unit", "Index (first = 100)", "Percent change from first"]
        chart_mode = st.selectbox("Visualization", chart_modes, key=f"chart_mode_{level}_{period}")
    else:
        chart_mode = "Multiyear comparison"
        st.caption("Multiyear comparison")
with chart_c:
    if period != "Multiyear":
        safe_multiselect_state(f"focus_area_{level}_{period}", pick)
        chart_territories = st.multiselect("Territories in main chart", pick,
            default=pick[:min(4, len(pick))], format_func=lambda i: label_of.get(i, i),
            key=f"focus_area_{level}_{period}")
    else:
        chart_territories = pick[:min(10, len(pick))]
if not focus:
    st.info("Choose variables in 'Variables in main chart' to visualize trajectories.")
elif not chart_territories:
    st.info("Choose a territory in 'Territories in main chart'.")
else:
    plot_long = prepare_long(filtered[filtered.territory_id.isin(chart_territories)], focus, period, meta)
    if plot_long.empty:
        st.info("No valid values for this chart selection.")
    elif period == "Multiyear":
        fig = px.bar(plot_long, x="territory_label", y="value", color="variable", barmode="group",
            color_discrete_map=variable_color_map(focus, meta), facet_row="unit", height=max(410, 240 * plot_long.unit.nunique()),
            labels={"value": "Value", "territory_label": "Territory"})
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("unit=", "")))
        st.plotly_chart(prettify(fig, max(410, 240 * plot_long.unit.nunique())), use_container_width=True)
    elif chart_mode == "Diverging z-score heatmap":
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
            matrix = part.pivot_table(index="variable", columns="time_label", values="z", aggfunc="mean")
            if matrix.empty:
                continue
            fig = px.imshow(matrix, color_continuous_scale=DIVERGING, zmin=-3, zmax=3, aspect="auto",
                labels={"color": "Z-score", "x": "Date", "y": "Variable"})
            fig.update_xaxes(nticks=12)
            fig.update_layout(title=label_of.get(territory, str(territory)), height=max(270, 80 + 42 * len(matrix)),
                margin=dict(l=5, r=5, t=55, b=25), coloraxis_colorbar=dict(title="z"))
            with panels[idx % len(panels)]:
                st.plotly_chart(fig, use_container_width=True, key=f"zheat_{idx}_{level}_{period}")
        st.caption("Blue = below each variable's own mean; warm = above. Each series is standardized within its territory and selected time interval. Color saturates beyond ±3 SD; data are not clipped.")
    else:
        if chart_mode == "Z-score trajectories":
            plot_long["plot_value"] = zscore_column(plot_long)
            y_label = "Z-score (standard deviations)"
        elif chart_mode == "Index (first = 100)":
            baseline = plot_long.sort_values("time").groupby(["territory_id", "variable"], observed=True)["value"].transform("first")
            plot_long["plot_value"] = 100 * plot_long["value"] / baseline.replace(0, np.nan)
            y_label = "Index (first observed = 100)"
        elif chart_mode == "Percent change from first":
            baseline = plot_long.sort_values("time").groupby(["territory_id", "variable"], observed=True)["value"].transform("first")
            plot_long["plot_value"] = 100 * (plot_long["value"] - baseline) / baseline.abs().replace(0, np.nan)
            y_label = "Change from first observed (%)"
        else:
            plot_long["plot_value"] = plot_long["value"]
            y_label = "Observed value"
        if chart_mode == "Raw values by unit":
            nu = plot_long.unit.nunique()
            fig = px.line(plot_long, x="time", y="plot_value", color="variable", line_dash="territory_label",
                facet_row="unit", facet_row_spacing=min(.065, .24 / max(1, nu - 1)),
                color_discrete_map=variable_color_map(focus, meta), hover_data=["territory_label", "value"],
                labels={"plot_value": y_label, "time": "Date"})
            fig.for_each_annotation(lambda a: a.update(text=a.text.replace("unit=", "")))
            h = max(420, 235 * nu)
        else:
            fig = px.line(plot_long, x="time", y="plot_value", color="variable", line_dash="territory_label",
                color_discrete_map=variable_color_map(focus, meta), hover_data=["territory_label", "value", "unit"],
                labels={"plot_value": y_label, "time": "Date"})
            fig.add_hline(y=100 if chart_mode == "Index (first = 100)" else 0, line_width=1,
                line_dash="dot", line_color="#7A8790", opacity=.85)
            if chart_mode == "Z-score trajectories":
                # Diverging marker colors encode the signed anomaly at each date;
                # line colors still identify the metric across all plots.
                for (_, _), points in plot_long.groupby(["territory_id", "variable"], observed=True):
                    fig.add_trace(go.Scatter(x=points["time"], y=points["plot_value"],
                        mode="markers", marker=dict(size=5, opacity=.8,
                            color=points["plot_value"], coloraxis="coloraxis"),
                        hoverinfo="skip", showlegend=False))
                fig.update_layout(coloraxis=dict(colorscale=DIVERGING, cmin=-2.5, cmax=2.5,
                    colorbar=dict(title="Signed z", thickness=12, len=.65)))
            h = 500
        st.plotly_chart(prettify(fig, h), use_container_width=True)
        st.caption("Color identifies the variable; line style identifies the territory. Z-scores and baseline indices are visualization transformations, not replacements for original measurements. Zero baselines yield undefined relative changes.")

# ----------------------- Multivariate relationships -----------------------
st.divider()
st.subheader("Cross-variable relationships · linear regression")
if len(selected) >= 2:
    rcols = st.columns([1.2, 1.2, 1.0, 1.0])
    xvar = rcols[0].selectbox("X variable", selected, index=0, format_func=lambda m: metric_label(m, meta), key=f"x_{level}_{period}")
    yvar = rcols[1].selectbox("Y variable", selected, index=min(1, len(selected) - 1), format_func=lambda m: metric_label(m, meta), key=f"y_{level}_{period}")
    choices = ["None"] + [m for m in selected if m not in (xvar, yvar)]
    hue = rcols[2].selectbox("Color by third variable", choices, format_func=lambda m: "Territory" if m == "None" else metric_label(m, meta), key=f"hue_{level}_{period}")
    sz = rcols[3].selectbox("Size by fourth variable", choices, format_func=lambda m: "Equal size" if m == "None" else metric_label(m, meta), key=f"size_{level}_{period}")
    if xvar == yvar:
        st.info("Choose different X and Y variables for regression.")
    else:
        basic_cols = ["territory_label", "territory_id", xvar, yvar]
        time_cols = ["year", "month"] if period == "Monthly" else ["year"] if period == "Annual" else ["period"]
        plot_cols = list(dict.fromkeys(basic_cols + time_cols + ([hue] if hue != "None" else []) + ([sz] if sz != "None" else [])))
        scatter = filtered[plot_cols].copy()
        for m in [xvar, yvar] + ([hue] if hue != "None" else []) + ([sz] if sz != "None" else []):
            scatter[m] = pd.to_numeric(scatter[m], errors="coerce")
        scatter = scatter.replace([np.inf, -np.inf], np.nan).dropna(subset=[xvar, yvar])
        if sz != "None":
            scatter["marker_size"] = scatter[sz].clip(lower=0).fillna(0.0)
            if not scatter["marker_size"].gt(0).any():
                scatter["marker_size"] = 1.0
                st.info("Selected size variable has no positive values; markers use equal size.")
        leg, statscol = st.columns([2.25, 1], gap="large")
        with leg:
            if hue == "None":
                fig = px.scatter(scatter, x=xvar, y=yvar, color="territory_label",
                    color_discrete_sequence=SERIES_COLORS,
                    size="marker_size" if sz != "None" else None, hover_data=time_cols,
                    labels={xvar: metric_label(xvar, meta), yvar: metric_label(yvar, meta)}, opacity=.65)
            else:
                fig = px.scatter(scatter.dropna(subset=[hue]), x=xvar, y=yvar, color=hue,
                    color_continuous_scale=DIVERGING if palette_family(hue, meta) == "diverging" else map_scale(hue, meta),
                    size="marker_size" if sz != "None" else None,
                    hover_data=["territory_label"] + time_cols,
                    labels={xvar: metric_label(xvar, meta), yvar: metric_label(yvar, meta)}, opacity=.68)
            fit = linear_regression(scatter, xvar, yvar)
            if fit:
                xx = np.linspace(fit["xmin"], fit["xmax"], 100)
                yy = fit["slope"] * xx + fit["intercept"]
                fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=f"OLS · R²={fit['r2']:.3f} · p={format_p(fit['p_value'])}",
                    line=dict(color="#1B364A", width=3, dash="dash"), hovertemplate="OLS fitted line<extra></extra>"))
            st.plotly_chart(prettify(fig, 470), use_container_width=True)
        with statscol:
            st.markdown("**Pooled ordinary least squares**")
            if fit:
                st.metric("Linear R²", f"{fit['r2']:.3f}")
                st.metric("Pearson r", f"{fit['r']:+.3f}")
                st.metric("p-value (two-sided; nominal)", format_p(fit["p_value"]))
                st.metric("Slope (β₁)", f"{fit['slope']:,.4g}")
                st.metric("Sample size (rows)", f"{fit['n']:,}")
                st.caption(f"ŷ = {fit['intercept']:.4g} + ({fit['slope']:.4g}) × X")
                st.caption(f"95% CI for slope: [{fit['ci95_low']:.4g}, {fit['ci95_high']:.4g}]")
                st.caption("H₀: β₁ = 0 (equivalently Pearson r = 0 in simple OLS with an intercept).")
                if fit["p_value"] < .05:
                    st.caption("Nominal p < 0.05 under an independence assumption; not causal evidence.")
            else:
                st.info("Statistics unavailable: at least three complete observations and variation in both X and Y are required.")
            st.warning("Inference caution: nominal p-values and confidence intervals assume independent errors. Repeated years/months within territories, temporal trends and spatial dependence violate this assumption; these results are exploratory, not robust tests of causality or significance.", icon="⚠️")
        if scatter.territory_id.nunique() > 1:
            with st.expander("Linear regressions by territory (R², p-value, CI)", expanded=False):
                rows = []
                for tid, part in scatter.groupby("territory_id"):
                    fitted = linear_regression(part, xvar, yvar)
                    if fitted:
                        rows.append({"Territory": label_of.get(tid, tid), "n": fitted["n"],
                                     "Slope": fitted["slope"], "Intercept": fitted["intercept"],
                                     "Pearson r": fitted["r"], "R²": fitted["r2"],
                                     "p (two-sided, nominal)": fitted["p_value"],
                                     "Slope CI95 lower": fitted["ci95_low"],
                                     "Slope CI95 upper": fitted["ci95_high"]})
                if rows:
                    st.dataframe(pd.DataFrame(rows).sort_values("R²", ascending=False), hide_index=True, use_container_width=True)
                else:
                    st.info("Not enough variation per territory for separate regressions.")
        if len(selected) >= 3:
            with st.expander("Explore three variables in 3D", expanded=False):
                zopts = [m for m in selected if m not in (xvar, yvar)]
                zvar = st.selectbox("Z variable", zopts, format_func=lambda m: metric_label(m, meta), key=f"z_{level}_{period}")
                subset = filtered[["territory_label", xvar, yvar, zvar] + time_cols].copy()
                for m in (xvar, yvar, zvar):
                    subset[m] = pd.to_numeric(subset[m], errors="coerce")
                subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[xvar, yvar, zvar])
                if not subset.empty:
                    fig = px.scatter_3d(subset, x=xvar, y=yvar, z=zvar, color="territory_label",
                        color_discrete_sequence=SERIES_COLORS, opacity=.75,
                        hover_data=time_cols, height=570)
                    fig.update_layout(margin=dict(l=0, r=0, t=25, b=0))
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least two variables in the sidebar for regression and multivariate relationships.")

# --------------------------- Correlations ---------------------------------
st.divider()
st.subheader("Correlation and multivariate patterns")
corr_cols = st.columns([2, 1])
with corr_cols[0]:
    correlation_choices = selected[:min(len(selected), 24)]
    safe_multiselect_state(f"corr_{level}_{period}", selected)
    corr_sel = st.multiselect("Metrics in correlation matrix", selected,
        default=correlation_choices[:min(10, len(correlation_choices))],
        format_func=lambda m: metric_label(m, meta), key=f"corr_{level}_{period}")
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
                        labels[i, j] = f"{rmat[i, j]:.2f}" + ("*" if significant else "")
                    else:
                        labels[i, j] = ""
            heatmap = go.Heatmap(
                z=rmat, x=corr_sel, y=corr_sel, zmin=-1, zmax=1, zmid=0,
                colorscale=DIVERGING, colorbar=dict(title="Pearson r"),
                customdata=hdata, text=labels,
                hovertemplate=("X: %{x}<br>Y: %{y}<br>Pearson r: %{z:.3f}"
                               "<br>p (nominal, two-sided): %{customdata[0]}"
                               "<br>q (FDR-BH): %{customdata[1]}"
                               "<br>n (paired observations): %{customdata[2]}<extra></extra>"),
            )
            heatmap.texttemplate = "%{text}" if k <= 12 else None
            fig = go.Figure(data=[heatmap])
            fig.update_layout(height=max(440, k * 35 + 180),
                              margin=dict(l=10, r=10, t=35, b=10),
                              font=dict(family="Arial", color="#234153"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("* marks off-diagonal correlations with FDR-BH q < 0.05. Hover over a cell for nominal p, adjusted q and paired n. Diagonal cells have no hypothesis test.")
            with st.expander("Pairwise correlations · p-values and FDR-adjusted q-values", expanded=False):
                st.dataframe(pair_table, use_container_width=True, hide_index=True)
                st.download_button("Download correlation tests (.csv)",
                    pair_table.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"mapbiomas_correlations_{level}_{period.lower()}.csv", mime="text/csv")
        else:
            st.info("No eligible correlation pairs: each test needs at least three paired observations and variation in both metrics.")
    else:
        st.info("Choose at least two metrics to calculate Pearson correlations.")
with corr_cols[1]:
    st.markdown("**Cross-variable analysis**")
    st.caption("The Pearson matrix uses all observations in the selected time and territorial filters. The blue–neutral–warm diverging scale is always fixed at −1…+1.")
    st.caption("The p-value tests H₀: Pearson r = 0 (two-sided). q-values control the Benjamini–Hochberg false discovery rate across the tested pairs in the visible matrix.")
    st.caption("These p-values assume independent observations and are exploratory for time-series and repeated territories. Multiple-testing adjustment does not correct temporal or spatial dependence.")
    if period == "Monthly" and len(selected):
        st.markdown("**Monthly seasonal profile**")
        seas_var = st.selectbox("Seasonal metric", selected, format_func=lambda m: metric_label(m, meta), key=f"season_{level}")
        seas = filtered[["territory_label", "month", seas_var]].copy()
        seas[seas_var] = pd.to_numeric(seas[seas_var], errors="coerce")
        seas = seas.groupby(["territory_label", "month"], as_index=False)[seas_var].mean()
        fig = px.line(seas, x="month", y=seas_var, color="territory_label",
            color_discrete_sequence=SERIES_COLORS, markers=True)
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTH_NAMES)
        st.plotly_chart(prettify(fig, 350), use_container_width=True)
        st.caption("Mean of monthly observations over selected years; does not represent annual totals.")

# --------------------------- Data / provenance ----------------------------
with st.expander("View observations · export CSV", expanded=False):
    time_cols = ["year", "month"] if period == "Monthly" else ["year"] if period == "Annual" else ["period"]
    cols = ["territory_id", "territory_name"] + time_cols + list(selected)
    displayed = filtered[cols].copy()
    st.dataframe(displayed, use_container_width=True, hide_index=True, height=350)
    st.download_button("Download filtered values (.csv)", displayed.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"mapbiomas_atmosphere_{level}_{period.lower()}_filtered.csv", mime="text/csv")
with st.expander("Methods · metric dictionary · licensing"):
    descriptions = pd.DataFrame([r for r in dictionary if r.get("period") == period and r.get("column_name") in selected])
    st.dataframe(descriptions, hide_index=True, use_container_width=True)
    st.markdown("""**Dataset:** values are read from the existing XLSX-derived Parquet files; the app performs no Earth Engine/GCS queries.

**Geography:** territorial geometries are genuine GEE exports, normalized only to polygonal GeoJSON types for Plotly display.

**Visualization only:** paired differences, percentages, z-scores, Pearson correlations and ordinary least-squares regressions are calculated dynamically from the selected observations. No transformations overwrite the underlying data.

**Statistical inference:** nominal two-sided p-values for Pearson r and the simple OLS slope are computed using SciPy; the slope 95% confidence interval uses the Student t distribution. Correlation-matrix q-values use Benjamini–Hochberg adjustment across the valid off-diagonal variable pairs. These significance calculations presume independent observations and do **not** correct for trends, serial correlation, shared territory, heteroskedasticity or spatial dependence. Treat them as exploratory until an appropriate time-series/panel-data model is fitted.

**Palettes:** brand colors are documented in the [MapBiomas identity guide](https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2024/10/manual-de-marca_MapBiomas_PT_versao-site_16.10.24.pdf). The [land-cover class legend](https://brasil.mapbiomas.org/iniciativas-e-produtos/cobertura-e-uso-da-terra/cobertura-30m/cobertura/) is categorical and does not determine continuous atmospheric color scales. Continuous colors in this dashboard are designed to be semantically consistent; they are not claimed to reproduce the exact map styles of every official MapBiomas Atmosphere layer.
""")
st.caption("MapBiomas Atmosphere · Community explorer · Original data retain their source attribution and licensing.")
