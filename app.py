import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tamil-nadu-mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_file

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
HTML_FILE = BASE / "templates" / "index.html"
CSV_FILE = DATA_DIR / "master_dataset.csv"
GEO_FILE = DATA_DIR / "tamil-nadu-districts.geojson"

app = Flask(__name__)
DF = pd.read_csv(CSV_FILE)
for _column in DF.columns:
    if _column == "District":
        continue
    _converted = pd.to_numeric(DF[_column], errors="coerce")
    if _converted.notna().sum() == DF[_column].notna().sum() and _converted.notna().any():
        DF[_column] = _converted

DISTRICTS = sorted(DF["District"].dropna().unique().tolist(), key=str.casefold)
SUMMARY = DF.groupby("District", sort=False).first().reindex(DISTRICTS)
NUMERIC = [c for c in DF.select_dtypes(include="number").columns if c != "District"]
CATEGORICAL = [c for c in DF.columns if c not in NUMERIC]
with GEO_FILE.open(encoding="utf-8") as _f:
    GEOJSON = json.load(_f)
BOUNDARY_ALIASES = {
    "Kanniyakumari": "Kanyakumari", "Sivaganga": "Sivagangai",
    "Thiruvarur": "Tiruvarur", "Tirupathur": "Tirupattur",
    "Tuticorin": "Thoothukudi", "Kanchipuram": "Kancheepuram",
}
MAP_FEATURES = [(BOUNDARY_ALIASES.get(f["properties"].get("dtname"), f["properties"].get("dtname")), f["geometry"])
                for f in GEOJSON["features"]]
MATPLOTLIB_LOCK = threading.Lock()

INK = "#19372a"
GREEN = "#347452"
PALE = "#a8cc96"
ORANGE = "#d98851"
MUTED = "#718078"
GRID = "#e7ece7"


def display_name(value):
    return re.sub(r"\bph\b", "pH", str(value).replace("_", " "), flags=re.I)


def py_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records_json(frame):
    return [{str(k): py_value(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def district_or_404(name):
    if name not in DISTRICTS:
        abort(404, description="Unknown district")
    return name


def population(scope, district):
    if scope == "district":
        return DF[DF["District"] == district].copy()
    if scope == "districts":
        return SUMMARY.reset_index().copy()
    if scope == "crops":
        return DF.copy()
    abort(400, description="scope must be district, districts, or crops")


def checked_field(name, fields):
    if name not in fields:
        abort(400, description="Unknown field")
    return name


def finish_figure(fig):
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.seek(0)
    return output


def style_title(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=12)


def draw_bar(ax, labels, values, title, colors_list=None, horizontal=False, ylabel=None, xlabel=None):
    palette = colors_list or [GREEN] * len(labels)
    if horizontal:
        ax.barh(labels, values, color=palette, height=.68)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values, color=palette, width=.68)
        ax.tick_params(axis="x", labelrotation=35)
    style_title(ax, title)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)


def district_record(district):
    return SUMMARY.loc[district]


def map_svg(metric, selected):
    if metric not in NUMERIC:
        metric = "S_Sufficient_Pct"
    vals = SUMMARY[metric].dropna().astype(float)
    low, high = float(vals.min()), float(vals.max())
    cmap_name = "RdYlBu_r" if metric == "pH_Mean" else "Blues" if "Rainfall" in metric else "YlGn"
    cmap = matplotlib.colormaps[cmap_name]
    norm = colors.Normalize(low, high if high != low else low + 1)
    all_points = []

    def collect(coords):
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            all_points.append(coords)
        else:
            for child in coords:
                collect(child)

    for _, geom in MAP_FEATURES:
        collect(geom["coordinates"])
    min_lon = min(p[0] for p in all_points); max_lon = max(p[0] for p in all_points)
    min_lat = min(p[1] for p in all_points); max_lat = max(p[1] for p in all_points)
    width, height, margin = 620, 500, 22
    scale = min((width - margin * 2) / (max_lon - min_lon), (height - margin * 2) / (max_lat - min_lat))
    map_width = (max_lon - min_lon) * scale
    map_height = (max_lat - min_lat) * scale
    x_offset = (width - map_width) / 2
    y_offset = (height - map_height) / 2

    def point(pair):
        x = x_offset + (pair[0] - min_lon) * scale
        y = y_offset + (max_lat - pair[1]) * scale
        return x, y

    def ring_path(ring):
        if len(ring) < 3:
            return ""
        pts = [point(p) for p in ring]
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Tamil Nadu district map">',
             f'<rect width="{width}" height="{height}" rx="12" fill="#e5eff3"/>']
    for name, geom in MAP_FEATURES:
        if name not in DISTRICTS:
            continue
        value = float(district_record(name)[metric])
        fill = colors.to_hex(cmap(norm(value)))
        if geom["type"] == "Polygon":
            polygons = [geom["coordinates"]]
        else:
            polygons = geom["coordinates"]
        d = " ".join(ring_path(ring) for polygon in polygons for ring in polygon)
        stroke = "#173c2b" if name == selected else "#ffffff"
        stroke_width = 3.2 if name == selected else 1.9
        unit = "%" if metric.endswith("_Pct") else " mm" if "Rainfall" in metric else ""
        parts.append(f'<path d="{d}" data-district="{escape(name)}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linejoin="round" fill-rule="evenodd"><title>{escape(name)} · {escape(display_name(metric))}: {value:g}{unit}</title></path>')
    parts.append("</svg>")
    return "".join(parts)


def chart_figure(kind, district, args):
    r = district_record(district)
    crop_rows = DF[DF["District"] == district]
    state_context = args.get("context") == "state"
    if kind in {"nutrient", "rainfall", "categories", "state_ph", "chemistry", "shc_grade", "shc_ph", "shc_ec", "micronutrients", "soil_distribution", "crop_categories", "duration", "water", "zone_match", "climate_rain", "climate_seasons", "humidity", "temperature", "rel_scatter", "corr_heatmap", "crop_scatter"}:
        if kind in {"chemistry", "soil_distribution"}:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        elif kind == "corr_heatmap":
            fig, axes = plt.subplots(figsize=(8.6, 7.0))
        else:
            fig, axes = plt.subplots(figsize=(8.4, 3.8))
    elif kind in {"eda_hist", "eda_category", "eda_scatter", "eda_completeness"}:
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        axes = [ax]
    else:
        abort(404, description="Unknown chart")

    ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
    if kind == "nutrient":
        keys = ["S", "Zn", "Fe", "Cu", "Mn", "B"]
        labels = ["Sulfur", "Zinc", "Iron", "Copper", "Manganese", "Boron"]
        vals = [r.get(f"{k}_Sufficient_Pct", np.nan) for k in keys]
        datasets = [(district, vals, GREEN)]
        if state_context:
            means = [SUMMARY[f"{k}_Sufficient_Pct"].mean() for k in keys]
            datasets.append(("State mean", means, PALE))
        x = np.arange(len(keys)); bar_w = .34 if len(datasets) == 2 else .62
        for i, (label, values, color) in enumerate(datasets):
            ax.bar(x + (i - (len(datasets)-1)/2)*bar_w, values, bar_w, label=label, color=color)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, max(100, np.nanmax(vals) * 1.12 if np.isfinite(vals).any() else 100))
        ax.set_ylabel("Sufficiency value")
        style_title(ax, "Nutrient sufficiency")
        if state_context: ax.legend(frameon=False, fontsize=8)
    elif kind in {"rainfall", "climate_rain"}:
        keys = ["Mean_Annual_Rainfall_mm", "SW_Monsoon_Jun_Sep_mm", "NE_Monsoon_Oct_Dec_mm"]
        labels = ["Annual", "SW monsoon", "NE monsoon"]
        vals = [r.get(k, np.nan) for k in keys]
        x = np.arange(3); width = .34 if state_context else .62
        ax.bar(x - width/2 if state_context else x, vals, width, color=GREEN, label=district)
        if state_context:
            means = [SUMMARY[k].mean() for k in keys]
            ax.bar(x + width/2, means, width, color=PALE, label="State mean")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Rainfall (mm)")
        style_title(ax, "Rainfall by season")
    elif kind in {"categories", "crop_categories"}:
        counts = crop_rows["Crop_Category"].fillna("Unknown").value_counts().sort_values()
        draw_bar(ax, counts.index.tolist(), counts.values, "Crop profiles by category", horizontal=True)
        ax.set_xlabel("Crop profile records")
    elif kind == "state_ph":
        s = SUMMARY["pH_Mean"].dropna().sort_values()
        ax.bar(s.index, s.values, color=[ORANGE if d == district else PALE for d in s.index])
        ax.tick_params(axis="x", labelrotation=75, labelsize=7)
        ax.set_ylabel("Mean soil pH")
        style_title(ax, "Soil pH across districts")
    elif kind == "chemistry":
        ax1, ax2 = axes
        fields1 = ["pH_Mean", "EC_dSm", "OC_Pct"]
        fields2 = ["N_kg_ac", "P_kg_ac", "K_kg_ac"]
        draw_bar(ax1, ["pH", "EC", "OC"], [r.get(k, 0) for k in fields1], "Soil measures")
        ax1.set_ylabel("Reported value")
        draw_bar(ax2, ["N", "P", "K"], [r.get(k, 0) for k in fields2], "Macronutrients")
        ax2.set_ylabel("kg/ac")
    elif kind == "shc_grade":
        nutrients = ["N", "P", "K", "OC"]
        grades = ["High", "Medium", "Low"]
        x = np.arange(len(nutrients)); base = np.zeros(len(nutrients))
        for grade, color in zip(grades, [GREEN, "#9fc77e", "#e3c988"]):
            values = [r.get(f"SHC_{n}_{grade}", 0) for n in nutrients]
            ax.bar(x, values, bottom=base, label=grade, color=color)
            base += np.nan_to_num(values)
        ax.set_xticks(x, nutrients); ax.set_ylabel("Reported SHC value")
        ax.legend(frameon=False, ncol=3, fontsize=8); style_title(ax, "N · P · K · organic carbon classification")
    elif kind == "shc_ph":
        keys = ["SHC_pH_Alkaline", "SHC_pH_Neutral", "SHC_pH_Acidic"]
        draw_bar(ax, ["Alkaline", "Neutral", "Acidic"], [r.get(k, 0) for k in keys], "SHC soil pH classes", ["#d29c64", "#93b989", "#7695ad"])
        ax.set_ylabel("Reported SHC value")
    elif kind == "shc_ec":
        keys = ["SHC_EC_Non_Saline", "SHC_EC_Saline"]
        draw_bar(ax, ["Non-saline", "Saline"], [r.get(k, 0) for k in keys], "SHC electrical conductivity", [GREEN, ORANGE])
        ax.set_ylabel("Reported SHC value")
    elif kind == "micronutrients":
        keys = ["S", "Fe", "Zn", "Cu", "B", "Mn"]
        x = np.arange(len(keys)); width = .36
        for offset, status, color in [(-width/2, "Sufficient", GREEN), (width/2, "Deficient", ORANGE)]:
            values = [r.get(f"SHC_{k}_{status}", np.nan) for k in keys]
            ax.bar(x + offset, values, width, label=status, color=color)
        ax.set_xticks(x, keys); ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel("Reported SHC value"); style_title(ax, "Sulfur and micronutrient status")
    elif kind == "soil_distribution":
        ax1, ax2 = axes
        for a, keys, labels, title in [(ax1, ["pH_Mean", "EC_dSm", "OC_Pct"], ["pH", "EC", "OC"], "Soil indicators by district"),
                                       (ax2, ["N_kg_ac", "P_kg_ac", "K_kg_ac"], ["N", "P", "K"], "Nutrient values by district")]:
            data = [SUMMARY[k].dropna().values for k in keys]
            a.boxplot(data, tick_labels=labels, patch_artist=True,
                      boxprops={"facecolor": "#c9dec2", "edgecolor": GREEN},
                      medianprops={"color": ORANGE, "linewidth": 1.5})
            style_title(a, title)
    elif kind == "duration":
        rows = crop_rows.dropna(subset=["Duration_Min_Days", "Duration_Max_Days"]).sort_values("Duration_Max_Days")
        x = np.arange(len(rows)); lows = rows["Duration_Min_Days"].to_numpy(); highs = rows["Duration_Max_Days"].to_numpy()
        ax.bar(x, lows, color=PALE, label="Minimum"); ax.bar(x, highs-lows, bottom=lows, color=GREEN, label="Range to maximum")
        ax.set_xticks(x, rows["Crop_Name"], rotation=70, ha="right", fontsize=7); ax.set_ylabel("Days")
        ax.legend(frameon=False, fontsize=8); style_title(ax, "Crop duration ranges")
    elif kind == "water":
        rows = crop_rows.dropna(subset=["Water_Min_mm", "Water_Max_mm"]).sort_values("Water_Max_mm")
        x = np.arange(len(rows)); lows = rows["Water_Min_mm"].to_numpy(); highs = rows["Water_Max_mm"].to_numpy()
        ax.bar(x, lows, color="#b7d3dd", label="Minimum"); ax.bar(x, highs-lows, bottom=lows, color="#6699a9", label="Range to maximum")
        ax.set_xticks(x, rows["Crop_Name"], rotation=70, ha="right", fontsize=7); ax.set_ylabel("Water (mm)")
        ax.legend(frameon=False, fontsize=8); style_title(ax, "Crop water requirements")
    elif kind == "zone_match":
        counts = crop_rows["Zone_Match"].fillna("Unknown").value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", colors=[GREEN, "#d9b36f", "#b8c4c0"][:len(counts)],
               textprops={"fontsize": 8, "color": INK}, wedgeprops={"width": .42, "edgecolor": "white"})
        style_title(ax, "Zone match flags")
    elif kind == "climate_seasons":
        counts = {"Kharif": 0, "Rabi": 0, "Whole year": 0}
        for val in crop_rows["Primary_TN_Seasons"].dropna().astype(str):
            low = val.casefold()
            for season in counts:
                if season.casefold() in low or (season == "Whole year" and "year" in low): counts[season] += 1
        draw_bar(ax, list(counts), list(counts.values()), "Crop profiles by season")
        ax.set_ylabel("Crop profile records")
    elif kind in {"humidity", "temperature"}:
        minimum, maximum, ylabel, title = (("Humidity_Min_Pct", "Humidity_Max_Pct", "Relative humidity (%)", "Humidity ranges by crop") if kind == "humidity"
                                           else ("Temp_Min_C", "Temp_Max_C", "Temperature (°C)", "Temperature ranges by crop"))
        rows = crop_rows.dropna(subset=[minimum, maximum]).sort_values(maximum)
        x = np.arange(len(rows)); lows = rows[minimum].to_numpy(); highs = rows[maximum].to_numpy()
        ax.bar(x, lows, color="#c4dce1", label="Minimum"); ax.bar(x, highs-lows, bottom=lows, color="#6699a9" if kind == "humidity" else ORANGE, label="Range to maximum")
        ax.set_xticks(x, rows["Crop_Name"], rotation=70, ha="right", fontsize=7); ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8); style_title(ax, title)
    elif kind == "rel_scatter":
        ax.scatter(SUMMARY["pH_Mean"], SUMMARY["OC_Pct"], c=[ORANGE if d == district else GREEN for d in SUMMARY.index], alpha=.8, s=35)
        ax.set_xlabel("Mean soil pH"); ax.set_ylabel("Organic carbon (%)"); style_title(ax, "Soil pH vs organic carbon")
    elif kind == "corr_heatmap":
        keys = ["pH_Mean", "EC_dSm", "OC_Pct", "N_kg_ac", "P_kg_ac", "K_kg_ac", "Mean_Annual_Rainfall_mm", "S_Sufficient_Pct", "Zn_Sufficient_Pct", "Fe_Sufficient_Pct"]
        labels = [display_name(k) for k in keys]
        corr = SUMMARY[keys].corr(method="pearson")
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(keys)), labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(keys)), labels, fontsize=7)
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(corr.iloc[i,j]) > .6 else INK)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.03); ax.set_title("District-level Pearson correlation", loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.grid(False)
    elif kind == "crop_scatter":
        rows = crop_rows.dropna(subset=["Duration_Min_Days", "Duration_Max_Days", "Water_Min_mm", "Water_Max_mm"])
        xx = (rows["Duration_Min_Days"] + rows["Duration_Max_Days"]) / 2
        yy = (rows["Water_Min_mm"] + rows["Water_Max_mm"]) / 2
        ax.scatter(xx, yy, color=GREEN, alpha=.75, s=28)
        ax.set_xlabel("Average crop duration (days)"); ax.set_ylabel("Average water requirement (mm)"); style_title(ax, "Crop duration vs water requirement")
    elif kind.startswith("eda_"):
        scope = args.get("scope", "districts")
        pop = population(scope, district)
        if kind == "eda_hist":
            field = checked_field(args.get("metric", "pH_Mean"), NUMERIC)
            vals = pop[field].dropna().astype(float)
            ax.hist(vals, bins=min(18, max(5, int(np.sqrt(max(len(vals), 1))))), color=GREEN, edgecolor="white")
            ax.set_xlabel(display_name(field)); ax.set_ylabel("Records"); style_title(ax, f"Distribution · {display_name(field)}")
        elif kind == "eda_category":
            field = checked_field(args.get("category", "Crop_Category"), CATEGORICAL)
            counts = pop[field].fillna("Missing").astype(str).value_counts().head(18).sort_values()
            draw_bar(ax, counts.index.tolist(), counts.values, f"Frequency · {display_name(field)}", horizontal=True)
            ax.set_xlabel("Records")
        elif kind == "eda_scatter":
            x_field = checked_field(args.get("x", "pH_Mean"), NUMERIC)
            y_field = checked_field(args.get("y", "OC_Pct"), NUMERIC)
            pairs = pop[[x_field, y_field]].dropna()
            ax.scatter(pairs[x_field], pairs[y_field], color=GREEN, alpha=.65, s=24)
            corr = pairs[x_field].corr(pairs[y_field]) if len(pairs) > 1 else np.nan
            ax.set_xlabel(display_name(x_field)); ax.set_ylabel(display_name(y_field))
            style_title(ax, f"{display_name(x_field)} vs {display_name(y_field)} · Pearson r = {corr:.2f}" if pd.notna(corr) else "Insufficient numeric pairs")
        else:
            missing = pop.isna().sum().sort_values(ascending=False)
            incomplete = missing[missing > 0]
            if incomplete.empty:
                ax.axis("off")
                ax.text(.5, .65, "100% COMPLETE", ha="center", va="center", fontsize=22, color=GREEN, fontweight="bold", transform=ax.transAxes)
                ax.text(.5, .4, f"{len(pop):,} records · {len(DF.columns)} fields checked · no missing values", ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)
            else:
                rates = (100 * incomplete / max(len(pop), 1)).sort_values()
                ax.barh([display_name(k) for k in rates.index], rates.values, color=ORANGE)
                ax.set_xlim(0, 100); ax.set_xlabel("Missing records (%)"); style_title(ax, "Fields with missing values")
    return fig


@app.get("/")
def home():
    return send_file(HTML_FILE, mimetype="text/html", max_age=0)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.get("/api/meta")
def meta():
    map_metrics = ["S_Sufficient_Pct", "Zn_Sufficient_Pct", "Fe_Sufficient_Pct", "Cu_Sufficient_Pct", "Mn_Sufficient_Pct", "B_Sufficient_Pct", "pH_Mean", "Mean_Annual_Rainfall_mm", "OC_Pct"]
    ranges = {k: [float(SUMMARY[k].min()), float(SUMMARY[k].max())] for k in map_metrics if k in SUMMARY}
    return jsonify({"districts": DISTRICTS, "columns": DF.columns.tolist(), "numeric": NUMERIC,
                    "categorical": CATEGORICAL, "map_metrics": map_metrics, "map_ranges": ranges,
                    "rows": len(DF), "district_count": len(DISTRICTS)})


@app.get("/api/profile/<path:district>")
def profile(district):
    district_or_404(district)
    r = district_record(district)
    return jsonify({"district": district, "zone": py_value(r.get("Agro_Climatic_Zone")),
                    "soil": py_value(r.get("Major_Soil_Type")), "row_count": int((DF["District"] == district).sum()),
                    "profile": {k: py_value(v) for k, v in r.items()},
                    "crop_categories": crop_category_counts(district)})


def crop_category_counts(district):
    c = DF.loc[DF["District"] == district, "Crop_Category"].fillna("Unknown").astype(str).value_counts()
    return c.to_dict()


@app.get("/api/rows/<path:district>")
def rows(district):
    district_or_404(district)
    frame = DF[DF["District"] == district]
    return jsonify({"columns": DF.columns.tolist(), "rows": records_json(frame)})


@app.get("/api/eda/stats")
def eda_stats():
    district = district_or_404(request.args.get("district", DISTRICTS[0]))
    scope = request.args.get("scope", "districts")
    pop = population(scope, district)
    stats = []
    for column in NUMERIC:
        series = pop[column].dropna().astype(float)
        stats.append({"field": column, "n": int(series.count()), "missing": int(pop[column].isna().sum()),
                      "mean": py_value(series.mean()), "std": py_value(series.std()), "min": py_value(series.min()),
                      "q1": py_value(series.quantile(.25)), "median": py_value(series.median()),
                      "q3": py_value(series.quantile(.75)), "max": py_value(series.max())})
    return jsonify({"scope": scope, "rows": len(pop), "stats": stats})


@app.get("/api/map.svg")
def map_endpoint():
    district = request.args.get("district", DISTRICTS[0])
    district_or_404(district)
    metric = request.args.get("metric", "S_Sufficient_Pct")
    return Response(map_svg(metric, district), mimetype="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/api/chart/<kind>.png")
def chart_endpoint(kind):
    district = district_or_404(request.args.get("district", DISTRICTS[0]))
    args = request.args
    with MATPLOTLIB_LOCK:
        fig = chart_figure(kind, district, args)
        png = finish_figure(fig)
    return send_file(png, mimetype="image/png", max_age=0)


@app.get("/api/export/<path:district>.csv")
def export_csv(district):
    district_or_404(district)
    content = DF[DF["District"] == district].to_csv(index=False).encode("utf-8-sig")
    return Response(content, mimetype="text/csv", headers={"Content-Disposition": f'attachment; filename="{district.lower().replace(" ", "-")}-district-data.csv"'})


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": error.description}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": error.description or "Not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
