"""
Render the Strava Training Dashboard as a single static HTML file at
docs/index.html. Used for GitHub Pages hosting — no Python server required.

Replaces Dash's server-side filter callback with client-side JS that toggles
visibility of pre-rendered figure divs (one per filter value). All 18
filter-aware figures are baked into the page on render.

Run locally:   python build_static.py
"""
import json
import re
from pathlib import Path

import plotly.io as pio

import app  # imports everything: data, helpers, figures

REPO_ROOT = Path(__file__).parent
DOCS = REPO_ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Collect every figure we'll embed in the page
# ---------------------------------------------------------------------------

FILTER_VALUES = ["7", "30", "90", "180", "365", "all"]
FILTER_LABELS = {
    "7": "Past Week",
    "30": "Past Month",
    "90": "Past 3 Months",
    "180": "Past 6 Months",
    "365": "Past Year",
    "all": "All Time",
}
DEFAULT_FILTER = "365"

figs = {}

# Filter-aware (24 figures total — 4 charts × 6 filter values)
for val in FILTER_VALUES:
    start = app._filter_start_from_value(val)
    figs[f"fitness-{val}"] = app.build_fitness_fig(start)
    figs[f"heatmap-{val}"] = app.build_heatmap_fig(start)
    figs[f"cumulative-{val}"] = app.build_cumulative_time_fig(start)

# Pies
figs["pie-all"] = app._make_pie(app.pie_df(),
                                f"🌟 All-Time · Since {int(app.df['year'].min())}")
figs["pie-prev"] = app._make_pie(app.pie_df("previous"),
                                 f"🔙 {app.prev_year} · {app.prev_year_hours:.1f}h")
figs["pie-ytd"] = app._make_pie(app.pie_df("YTD"),
                                f"🔥 {app.latest_year} YTD · {app.ytd_hours:.1f}h · "
                                f"{app.progress_percent} of year")

# Other named figures
figs["combined-form-acwr"] = app.combined_form_acwr_fig
figs["yoy"] = app.yoy_fig
figs["map"] = app.map_fig
figs["scatter-pace"] = app.scatter_pace_fig
figs["scatter-efficiency"] = app.scatter_efficiency_fig

# PR sparklines (keyed by slug of title)
def _slug(title):
    return re.sub(r"[^a-z0-9]+", "-",
                  title.encode("ascii", "ignore").decode().lower()).strip("-")

for title, _value, _subtitle, spark in app.prs_with_sparks:
    if spark is not None:
        figs[f"spark-{_slug(title)}"] = spark

# Serialise every figure to JSON once (Plotly's pio.to_json handles numpy types)
figures_json_str = "{\n" + ",\n".join(
    f"  {json.dumps(fid)}: {pio.to_json(fig)}" for fid, fig in figs.items()
) + "\n}"

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

# Reuse the palette from app.py for inline styles
BG = app.dark_bg_color
PAPER = app.dark_paper_color
GRID = app.dark_grid_color
TEXT = app.dark_text_color
MUTED = app.muted_text_color

CARD_STYLE = (
    f"background-color:{PAPER};"
    f"border:1px solid {GRID};"
    "border-radius:10px;"
)

# Summary bar for collapsible <details> sections (less-used plots).
SUMMARY_STYLE = (
    f"cursor:pointer;font-weight:600;color:{TEXT};"
    f"padding:8px 14px;background-color:{PAPER};"
    f"border:1px solid {GRID};border-radius:8px;"
)

def kpi_card_html(title, value, subtitle, value_color=None):
    color_attr = f"color:{value_color};" if value_color else f"color:{TEXT};"
    # Equal-width on desktop (col-md), two-up on mobile (col-6) — fits 7 cards.
    return f"""<div class="col-6 col-md mb-2">
  <div class="card text-center h-100 kpi-card" style="{CARD_STYLE}">
    <div class="card-body py-3">
      <div class="text-uppercase" style="color:{MUTED};font-size:0.72rem;letter-spacing:0.06em;font-weight:500">{title}</div>
      <div class="kpi-value mt-1 fw-bold" style="font-size:1.75rem;line-height:1.1;{color_attr}">{value}</div>
      <div style="color:{MUTED};font-size:0.78rem;margin-top:0.25rem">{subtitle}</div>
    </div>
  </div>
</div>"""

def insight_card_html(icon, text):
    # Convert simple markdown bold (**X**) to <strong>X</strong>
    text_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return f"""<div class="col-md-3 mb-2">
  <div class="card h-100" style="{CARD_STYLE}">
    <div class="card-body py-3 text-center">
      <div style="font-size:1.4rem;margin-bottom:4px">{icon}</div>
      <div style="color:{TEXT};font-size:0.9rem;line-height:1.45">{text_html}</div>
    </div>
  </div>
</div>"""

def pr_card_html(title, value, subtitle, spark_id=None):
    spark_html = ""
    if spark_id:
        spark_html = (
            f'<div id="{spark_id}" '
            'style="height:56px;margin-top:6px"></div>'
        )
    return f"""<div class="col-md-4 mb-3">
  <div class="card text-center h-100 pr-card" style="{CARD_STYLE}">
    <div class="card-body py-3">
      <div class="text-uppercase" style="color:{MUTED};font-size:0.72rem;letter-spacing:0.06em;font-weight:500">{title}</div>
      <div class="pr-value mt-1 fw-bold" style="font-size:1.35rem;line-height:1.2;color:{TEXT}">{value}</div>
      <div style="color:{MUTED};font-size:0.78rem;margin-top:0.25rem">{subtitle}</div>
      {spark_html}
    </div>
  </div>
</div>"""

def filter_aware_divs_html(chart_id_prefix):
    """Emit 6 divs (one per filter value); JS hides/shows the active one."""
    return "\n".join(
        f'<div id="{chart_id_prefix}-{val}" class="filter-aware filter-{chart_id_prefix}" '
        f'data-filter="{val}"></div>'
        for val in FILTER_VALUES
    )

def df_to_table_html(records, columns):
    rows = "\n".join(
        "<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in columns) + "</tr>"
        for r in records
    )
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
    return (
        '<table class="table table-dark table-sm mb-0" '
        f'style="background-color:{PAPER};border-radius:10px;overflow:hidden">'
        f"<thead>{header}</thead><tbody>{rows}</tbody></table>"
    )

# ---------------------------------------------------------------------------
# Build the page sections
# ---------------------------------------------------------------------------

# KPI cards
kpi_html = "".join([
    kpi_card_html("Fitness (CTL)", f"{app.current_fitness:.0f}", "42-day load avg"),
    kpi_card_html("Form (TSB)", f"{app.current_form:+.0f}", app.form_label,
                  value_color=app.form_color),
    kpi_card_html("Load Ratio", f"{app.acwr:.2f}", app.acwr_label,
                  value_color=app.acwr_color),
    kpi_card_html(
        "This Week",
        f"{app.this_week_hours:.1f}h",
        f"{app.week_sign}{app.week_delta:.1f}h vs 8-wk avg",
        value_color=app.week_color,
    ),
    kpi_card_html("Weekly Streak", f"{app.week_streak} weeks",
                  f"{app.week_streak_activities} activities"),
    kpi_card_html("Day Streak", f"{app.current_streak} days",
                  f"ending {app.last_activity_date.strftime('%b %d')}"),
])

# Insights
insights_html = "".join(insight_card_html(icon, text) for icon, text in app._insights)

# Tables (use the same DataFrames app.py builds)
individual_stats_table = df_to_table_html(
    app.individual_stats_df.to_dict("records"),
    list(app.individual_stats_df.columns),
)
recent_activities_table = df_to_table_html(
    app.recent_activities_df.to_dict("records"),
    list(app.recent_activities_df.columns),
)
comparative_stats_table = df_to_table_html(
    app.comparative_stats_df.to_dict("records"),
    list(app.comparative_stats_df.columns),
)
# Fitness explainer — shared markdown from app.py, converted to minimal HTML.
explainer_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", app.FITNESS_EXPLAINER_MD)
explainer_html = explainer_html.replace("\n\n", "<br><br>").replace("\n", "<br>")

# Filter dropdown options
filter_options_html = "\n".join(
    f'<option value="{val}"{" selected" if val == DEFAULT_FILTER else ""}>{label}</option>'
    for val, label in FILTER_LABELS.items()
)

# PR cards grid
pr_cards_html = "".join(
    pr_card_html(title, value, subtitle,
                 spark_id=f"spark-{_slug(title)}" if spark is not None else None)
    for title, value, subtitle, spark in app.prs_with_sparks
)

# ---------------------------------------------------------------------------
# Assemble the page
# ---------------------------------------------------------------------------

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strava Training Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
  :root {{
    --dash-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  }}
  html, body, .card, .card-body, h1, h2, h3, h4, h5, h6, p, span, div, a, button, label, table, td, th {{
    font-family: var(--dash-font) !important;
  }}
  body {{ background-color: {BG} !important; color: {TEXT}; }}
  .kpi-value, .pr-value {{
    font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; letter-spacing: 0.005em;
  }}
  .kpi-card, .pr-card {{ transition: transform 120ms ease, border-color 120ms ease; }}
  .kpi-card:hover, .pr-card:hover {{ transform: translateY(-1px); border-color: #374151 !important; }}
  h1.dashboard-title {{
    background: linear-gradient(135deg, #60A5FA, #A855F7);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700; letter-spacing: -0.01em;
  }}
  h4 {{ font-weight: 600; color: {TEXT}; }}
  hr {{ border-color: {GRID}; opacity: 0.6; }}
  .filter-aware {{ display: none; }}
  .filter-aware.active {{ display: block; }}
  table.table-dark {{ border-color: {GRID}; }}
  table.table-dark td, table.table-dark th {{ border-color: {GRID}; font-size: 0.9rem; }}
  table.table-dark thead th {{ background-color: #2C2C2C; font-weight: 600; }}
  .filter-bar {{
    background-color: {PAPER}; border: 1px solid {GRID}; border-radius: 6px;
  }}
  select.form-select {{
    background-color: {PAPER}; color: {TEXT}; border-color: {GRID};
  }}
</style>
</head>
<body>
<div class="container-fluid">

<h1 class="text-center my-4 dashboard-title">Strava Training Dashboard</h1>

<div class="row g-3 mb-2">
{kpi_html}
</div>

<div class="row g-3 mb-2">
{insights_html}
</div>

<hr>

<!-- Section A: Stats tables -->
<div class="row mt-4">
  <div class="col-md-6">
    <h4 class="text-center my-3">🏆 Overall Stats</h4>
    {individual_stats_table}
  </div>
  <div class="col-md-6">
    <h4 class="text-center my-3">⏱️ Recent Activities</h4>
    {recent_activities_table}
  </div>
</div>

<div class="row mt-4">
  <div class="col-md-3"></div>
  <div class="col-md-6">
    <h4 class="text-center my-3">📊 Year-over-Year Comparison</h4>
    {comparative_stats_table}
  </div>
</div>

<hr>

<!-- Section A: Form + Load explainer + chart -->
<div class="px-4 py-2 mb-2" style="line-height:1.6;color:{TEXT};font-size:0.92rem">
{explainer_html}
</div>

<div class="row"><div class="col-md-12"><div id="combined-form-acwr"></div></div></div>

<hr>

<!-- Section B: Filter + fitness/YoY + heatmap/cumulative -->
<div class="d-flex align-items-center justify-content-center my-3 px-3 py-2 filter-bar">
  <span class="me-3 fw-semibold" style="color:{TEXT}">Filter range:</span>
  <select id="global-filter" class="form-select w-auto">
{filter_options_html}
  </select>
</div>

<div class="row">
  <div class="col-md-6">
{filter_aware_divs_html('fitness')}
  </div>
  <div class="col-md-6"><div id="yoy"></div></div>
</div>

<hr>

<div class="row">
  <div class="col-md-6">
{filter_aware_divs_html('heatmap')}
  </div>
  <div class="col-md-6">
{filter_aware_divs_html('cumulative')}
  </div>
</div>

<hr>

<!-- Section C: Reference (less-used plots collapsed by default) -->
<div class="row"><div class="col-md-12"><div id="map"></div></div></div>

<hr>

<details class="mb-2">
  <summary style="{SUMMARY_STYLE}">🏃 Run Pace &amp; Efficiency</summary>
  <div class="row pt-2">
    <div class="col-md-6"><div id="scatter-pace"></div></div>
    <div class="col-md-6"><div id="scatter-efficiency"></div></div>
  </div>
</details>

<details class="mb-2">
  <summary style="{SUMMARY_STYLE}">🥧 Activity Mix</summary>
  <div class="row pt-2">
    <div class="col-md-4"><div id="pie-all"></div></div>
    <div class="col-md-4"><div id="pie-prev"></div></div>
    <div class="col-md-4"><div id="pie-ytd"></div></div>
  </div>
</details>

<hr>

<h4 class="text-center my-3">🏆 Personal Records</h4>
<div class="row g-3">
{pr_cards_html}
</div>

</div>

<script>
const FIGURES = {figures_json_str};

const PLOTLY_CONFIG = {{responsive: true, displaylogo: false}};

function renderFigure(id) {{
  const fig = FIGURES[id];
  const el = document.getElementById(id);
  if (fig && el) {{
    Plotly.newPlot(id, fig.data, fig.layout, PLOTLY_CONFIG);
  }}
}}

function updateFilter() {{
  const value = document.getElementById('global-filter').value;
  document.querySelectorAll('.filter-aware').forEach(el => {{
    if (el.dataset.filter === value) {{
      el.classList.add('active');
      // Plotly figures only render properly while visible — re-call when shown
      Plotly.Plots.resize(el);
    }} else {{
      el.classList.remove('active');
    }}
  }});
}}

document.addEventListener('DOMContentLoaded', () => {{
  // Render every figure on load
  Object.keys(FIGURES).forEach(renderFigure);

  // Wire up filter dropdown
  const sel = document.getElementById('global-filter');
  sel.value = {DEFAULT_FILTER!r};
  updateFilter();
  sel.addEventListener('change', updateFilter);

  // Collapsible <details>: Plotly renders at 0px while hidden, so resize the
  // plots inside the moment a section is first opened.
  document.querySelectorAll('details').forEach(d => {{
    d.addEventListener('toggle', () => {{
      if (d.open) {{
        d.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
      }}
    }});
  }});
}});
</script>
</body>
</html>
"""

# Write the page
OUT = DOCS / "index.html"
OUT.write_text(HTML, encoding="utf-8")
print(f"✅ Wrote {OUT} ({len(HTML):,} bytes, {len(figs)} figures embedded)")

# Drop a small .nojekyll so GH Pages doesn't try to process the directory
(DOCS / ".nojekyll").touch()
print(f"✅ Wrote {DOCS / '.nojekyll'}")
