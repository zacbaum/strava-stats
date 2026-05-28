import ast
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dash_table, dcc, html
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#######################
# CONFIG
#######################

# Personal HR limits used by the TRIMP-based training-load formula.
# Tweak these to your own values for an honest fitness/fatigue model.
HR_MAX = 195
HR_REST = 55

#######################
# DATA PREPARATION
#######################

df = pd.read_csv("activities.csv", parse_dates=["start_date_local"])

# For runs with no recorded elevation gain, fall back to start_date (UTC) which
# tends to be more reliably populated than start_date_local on those entries.
df.loc[(df['type'] == 'Run') & (df['total_elevation_gain'] == 0), 'start_date_local'] = pd.to_datetime(df['start_date'])

# Polyline snapshot for the map. Trend-chart scoping is handled by the
# global filter dropdown, not by upstream slicing here.
polylines_for_map = df['summary_polyline'].copy()

df['type'] = df.apply(
    lambda row: 'Racquet Sports' if row['type'] == 'Workout' and 'pickleball' in str(row.get('name', '')).lower()
    else ('Racquet Sports' if row['type'] == 'Workout' and 'squash' in str(row.get('name', '')).lower()
        else ('Cardio' if row['type'] == 'Workout'
            else ('Weight Training' if row['type'] == 'WeightTraining' else row['type']))),
    axis=1
)

df['date'] = df['start_date_local'].dt.date
df['year'] = df['start_date_local'].dt.year
df['week_start'] = df['start_date_local'].dt.to_period('W').dt.start_time
df['duration_hr'] = df['elapsed_time'] / 3600
df['year_month'] = df['start_date_local'].dt.to_period('M').dt.to_timestamp()
df['weekday'] = df['start_date_local'].dt.day_name()
df['hour_of_day'] = df['start_date_local'].dt.hour
df['weekday_idx'] = df['start_date_local'].dt.weekday  # 0 = Monday

# Parse start_latlng — Strava stores it as a string like "[51.676, -0.607]" or "[]"
def _parse_latlng(s):
    if not isinstance(s, str) or s in ('', '[]'):
        return (None, None)
    try:
        coords = ast.literal_eval(s)
        if not coords or len(coords) < 2:
            return (None, None)
        return (float(coords[0]), float(coords[1]))
    except Exception:
        return (None, None)

_latlng = df['start_latlng'].apply(_parse_latlng)
df['lat'] = _latlng.apply(lambda x: x[0])
df['lng'] = _latlng.apply(lambda x: x[1])

# Google encoded polyline decoder — turns a string like "_p~iF~ps|U..." into
# a list of (lat, lng) tuples. Used to render activity routes on the map.
def decode_polyline(encoded):
    if not isinstance(encoded, str) or not encoded:
        return []
    coords = []
    index, lat, lng = 0, 0, 0
    n = len(encoded)
    while index < n:
        result, shift = 0, 0
        while index < n:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result, shift = 0, 0
        while index < n:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng
        coords.append((lat * 1e-5, lng * 1e-5))
    return coords

today = datetime.now().date()
latest_year = df['year'].max()
prev_year = latest_year - 1

#######################
# STYLING & THEMING
#######################

colors = px.colors.qualitative.Set2
activity_types = df['type'].unique()

# Activity colors — harmonized for dark backgrounds, consistent saturation
color_map = {
    'Run':              '#3B82F6',  # blue
    'Ride':             '#F59E0B',  # amber
    'Racquet Sports':   '#A855F7',  # purple
    'Cardio':           '#14B8A6',  # teal
    'Weight Training':  '#EC4899',  # pink
    'Hike':             '#94A3B8',  # slate
    'Walk':             '#84CC16',  # lime
}

# Fill any activity types that aren't pre-mapped, cycling through Set2
missing_types = [t for t in activity_types if t not in color_map]
for i, t in enumerate(missing_types):
    color_map[t] = colors[i % len(colors)]

# Surface palette — softer than pure white-on-black for less eye strain
dark_bg_color = "#0F1115"
dark_paper_color = "#181A20"
dark_text_color = "#E5E7EB"
dark_grid_color = "#262A33"
muted_text_color = "#9CA3AF"

# Semantic palette used across charts (form lines, rolling windows, trend dirs)
SERIES = {
    "fitness":    "#FFFFFF",
    "fatigue":    "#FB923C",
    "form":       "#4ADE80",
    "form_zero":  "#4B5563",
    "rolling_7":  "#60A5FA",
    "rolling_28": "#A78BFA",
    "rolling_365": "#FBBF24",
    "fresh":      "#4ADE80",
    "optimal":    "#60A5FA",
    "productive": "#FACC15",
    "overreach":  "#F87171",
}

# Single font stack used for HTML (via assets/style.css) and Plotly figures
chart_font_family = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
chart_font = dict(color=dark_text_color, family=chart_font_family)
title_font = dict(color=dark_text_color, family=chart_font_family, size=15)

# Master Plotly template — sets defaults so individual charts don't repeat them
dark_template = dict(
    layout=dict(
        paper_bgcolor=dark_paper_color,
        plot_bgcolor=dark_paper_color,
        font=chart_font,
        title=dict(x=0.5, xanchor='center', y=0.97, yanchor='top', font=title_font),
        margin=dict(l=20, r=20, t=60, b=20),
        colorway=list(color_map.values()),
        hoverlabel=dict(
            bgcolor="#1F2937",
            bordercolor=dark_grid_color,
            font=dict(color=dark_text_color, family=chart_font_family, size=12),
        ),
        legend=dict(
            font=chart_font,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
        ),
        xaxis=dict(
            gridcolor=dark_grid_color, zerolinecolor=dark_grid_color,
            linecolor=dark_grid_color, tickcolor=dark_grid_color,
            showgrid=False,
        ),
        yaxis=dict(
            gridcolor=dark_grid_color, zerolinecolor=dark_grid_color,
            linecolor=dark_grid_color, tickcolor=dark_grid_color,
            showgrid=True, gridwidth=1,
        ),
    )
)

#######################
# TRAINING LOAD MODEL
#######################
# Banister TRIMP using heart-rate reserve:
#   HRr   = (HR_avg - HR_REST) / (HR_MAX - HR_REST)        # 0..1
#   TRIMP = duration_min × HRr × 0.64 × exp(1.92 × HRr)
#
# Sessions with no recorded HR fall back to duration_hr × 100, matching
# the previous behaviour. Weight training and most racquet sessions land here.

def _trimp_vectorized(frame):
    duration_min = frame['duration_hr'] * 60
    hr = frame['average_heartrate']
    hrr = ((hr - HR_REST) / (HR_MAX - HR_REST)).clip(lower=0.0, upper=1.0)
    with_hr = duration_min * hrr * 0.64 * np.exp(1.92 * hrr)
    no_hr = frame['duration_hr'] * 100
    return np.where(hr.notna(), with_hr, no_hr)

df['training_load'] = _trimp_vectorized(df)

# Daily series with CTL (Fitness, 42d), ATL (Fatigue, 7d), TSB (Form = CTL - ATL)
date_range = pd.date_range(start=df['date'].min(), end=today, freq='D')
date_df_full = pd.DataFrame({"date": date_range.date})
daily_load = df.groupby('date')['training_load'].sum().reset_index()
daily_scores = pd.merge(date_df_full, daily_load, on='date', how='left').fillna(0)
daily_scores['fitness'] = daily_scores['training_load'].ewm(span=42, min_periods=1).mean()
daily_scores['fatigue'] = daily_scores['training_load'].ewm(span=7, min_periods=1).mean()
daily_scores['form'] = daily_scores['fitness'] - daily_scores['fatigue']

#######################
# YEAR / PROJECTION METRICS
#######################

def pie_df(year=None):
    if year == "YTD":
        return df[df["year"] == latest_year].groupby("type")["duration_hr"].sum().reset_index()
    elif year == "previous":
        return df[df["year"] == prev_year].groupby("type")["duration_hr"].sum().reset_index()
    else:
        return df.groupby("type")["duration_hr"].sum().reset_index()

current_ytd_df = df[df["year"] == latest_year]
prev_year_df = df[df["year"] == prev_year]

day_of_year = today.timetuple().tm_yday
year_progress = day_of_year / (366 if today.year % 4 == 0 else 365)

ytd_hours = current_ytd_df["duration_hr"].sum()
prev_year_hours = prev_year_df["duration_hr"].sum()

# Pattern-based projection: assume the remainder of this year mirrors what last
# year did over the equivalent remaining days. Far more honest than linear
# extrapolation when the previous year had a strongly seasonal load.
prev_year_at_same_day = prev_year_df[
    prev_year_df['start_date_local'].dt.dayofyear <= day_of_year
]['duration_hr'].sum()
prev_year_remaining = max(prev_year_hours - prev_year_at_same_day, 0)
projected_hours = ytd_hours + prev_year_remaining

on_track = projected_hours > prev_year_hours
trend_emoji = "🔼" if on_track else "🔽"
progress_percent = f"{year_progress:.1%}"

#######################
# KPI METRICS
#######################

latest_scores = daily_scores.iloc[-1]
current_fitness = latest_scores['fitness']
current_form = latest_scores['form']

def form_status(tsb):
    # Coggan-style training-stress balance bands.
    if tsb > 5:
        return "Fresh", "#4ade80"
    elif tsb > -10:
        return "Optimal", "#60a5fa"
    elif tsb > -30:
        return "Productive", "#facc15"
    else:
        return "Overreaching", "#f87171"

form_label, form_color = form_status(current_form)

# This week (Mon–Sun) vs trailing 8-week average
this_week_start = today - timedelta(days=today.weekday())
this_week_hours = df[df['date'] >= this_week_start]['duration_hr'].sum()
eight_weeks_ago = this_week_start - timedelta(weeks=8)
trailing_8wk_hours = df[(df['date'] >= eight_weeks_ago) & (df['date'] < this_week_start)]['duration_hr'].sum()
trailing_8wk_avg = trailing_8wk_hours / 8
week_delta = this_week_hours - trailing_8wk_avg
week_color = "#4ade80" if week_delta >= 0 else "#f87171"
week_sign = "+" if week_delta >= 0 else ""

# Current streak: consecutive days ending at the most recent activity
sorted_unique_dates = sorted(df['date'].unique())
current_streak = 1 if sorted_unique_dates else 0
for i in range(len(sorted_unique_dates) - 1, 0, -1):
    if (sorted_unique_dates[i] - sorted_unique_dates[i - 1]).days == 1:
        current_streak += 1
    else:
        break

last_activity_date = df['date'].max()
days_since_last = (today - last_activity_date).days

# ACWR (Acute:Chronic Workload Ratio) = avg-daily-load over 7 days / 28 days.
# Sports-science consensus: 0.8–1.3 = optimal, 1.3–1.5 = caution, >1.5 = high
# injury risk, <0.8 = undertraining / detraining.
_acute_7d = daily_scores[daily_scores['date'] > today - timedelta(days=7)]['training_load'].mean()
_chronic_28d = daily_scores[daily_scores['date'] > today - timedelta(days=28)]['training_load'].mean()
acwr = float(_acute_7d / _chronic_28d) if _chronic_28d > 0 else 0.0

def acwr_status(ratio):
    if ratio < 0.8:
        return "Undertraining", "#60a5fa"
    elif ratio <= 1.3:
        return "Optimal", "#4ade80"
    elif ratio <= 1.5:
        return "Caution", "#facc15"
    else:
        return "High Risk", "#f87171"

acwr_label, acwr_color = acwr_status(acwr)

#######################
# WEEKLY STREAK (consecutive weeks with at least one activity)
#######################

_active_weeks = {ws.date() for ws in df['week_start'].dropna().unique()}
_most_recent_active_week = last_activity_date - timedelta(days=last_activity_date.weekday())
week_streak = 0
_check = _most_recent_active_week
while _check in _active_weeks:
    week_streak += 1
    _check -= timedelta(days=7)
_streak_start_week = _most_recent_active_week - timedelta(days=7 * (week_streak - 1)) if week_streak else None
week_streak_activities = (
    int((df['week_start'].dt.date >= _streak_start_week).sum())
    if _streak_start_week else 0
)

#######################
# MONTHLY ACTIVITY
#######################

monthly_activity = df.groupby([pd.Grouper(key='start_date_local', freq='ME'), 'type'])['duration_hr'].sum().reset_index()
monthly_activity['month'] = monthly_activity['start_date_local'].dt.to_period('M').dt.to_timestamp()

#######################
# WEEKDAY × HOUR HEATMAP
#######################

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_pivot = (
    df.groupby(['weekday_idx', 'hour_of_day'])['duration_hr'].sum()
    .unstack(fill_value=0)
    .reindex(index=range(7), columns=range(24), fill_value=0)
)

#######################
# RUN PERFORMANCE
#######################

run_df = df[(df["type"] == "Run") & (df["distance"] > 0)].copy()
run_df["pace_min_per_km"] = run_df["moving_time"] / 60 / (run_df["distance"] / 1000)
run_df["year_str"] = run_df["year"].astype(int).astype(str)

# Discrete year-color map for the run efficiency scatter
year_palette = px.colors.qualitative.Plotly
sorted_years = sorted(run_df["year"].dropna().astype(int).unique())
year_color_map = {str(y): year_palette[i % len(year_palette)] for i, y in enumerate(sorted_years)}

#######################
# BONUS STATS
#######################

longest = df.loc[df["duration_hr"].idxmax()]
longest_session = {
    "duration_hr": round(longest["duration_hr"], 2),
    "date": longest["date"],
    "type": longest["type"]
}

daily_hours = df.groupby("date")["duration_hr"].sum()
longest_day = daily_hours.idxmax()
longest_day_hours = round(daily_hours.max(), 2)

weekly_hours = df.groupby("week_start")["duration_hr"].sum()
busiest_week_start = weekly_hours.idxmax()
busiest_week_total = round(weekly_hours.max(), 2)
busiest_week_end = busiest_week_start + timedelta(days=6)

last_year = df[df["year"] == prev_year]
current_ytd = df[df["year"] == latest_year]

avg_last_year = round(last_year.groupby("week_start")["duration_hr"].sum().mean(), 2)
avg_ytd = round(current_ytd.groupby("week_start")["duration_hr"].sum().mean(), 2)

df_sorted = df.sort_values("start_date_local")
df_sorted["prev_date"] = df_sorted["start_date_local"].shift(1)
df_sorted["gap"] = (df_sorted["start_date_local"] - df_sorted["prev_date"]).dt.days
longest_break = df_sorted["gap"].max()
gap_row = df_sorted.loc[df_sorted["gap"].idxmax()]
break_start = gap_row["prev_date"].date()
break_end = gap_row["start_date_local"].date()

unique_dates = pd.Series(df_sorted['date'].unique()).sort_values()
unique_dates_pd = pd.to_datetime(unique_dates)
date_diffs = unique_dates_pd.diff().dt.days
streak_groups = (date_diffs > 1).cumsum()
streak_lengths = unique_dates.groupby(streak_groups).count()
longest_streak_len = streak_lengths.max()
longest_streak_idx = streak_lengths.idxmax()
streak_dates = unique_dates[streak_groups == longest_streak_idx]
streak_start = streak_dates.min()
streak_end = streak_dates.max()

total_days = (df['date'].max() - df['date'].min()).days + 1
active_days = df['date'].nunique()
active_days_last_year = last_year['date'].nunique()
active_days_ytd = current_ytd['date'].nunique()
active_days_percent = round((active_days / total_days) * 100, 1)
active_days_percent_last_year = round((active_days_last_year / 365) * 100, 1)
days_ytd = (today - datetime(latest_year, 1, 1).date()).days + 1
active_days_percent_ytd = round((active_days_ytd / days_ytd) * 100, 1)

individual_stats_df = pd.DataFrame([
    {"Label": "🔥 Longest Session", "Value": f'{longest_session["duration_hr"]}h on {longest_session["date"]}'},
    {"Label": "📆 Biggest Day", "Value": f'{longest_day_hours}h on {longest_day}'},
    {"Label": "📈 Busiest Week", "Value": f'{busiest_week_total}h ({busiest_week_start.date()}–{busiest_week_end.date()})'},
    {"Label": "🛌 Longest Break", "Value": f'{longest_break} days ({break_start} to {break_end})'},
    {"Label": "💪 Longest Streak", "Value": f'{longest_streak_len} days ({streak_start} to {streak_end})'},
])

comparative_stats_df = pd.DataFrame([
    {"Metric": "📊 Avg Weekly Hours",
     "Overall": f'{round(df.groupby("week_start")["duration_hr"].sum().mean(), 2)}h',
     "Last Year": f'{avg_last_year}h',
     "YTD": f'{avg_ytd}h'},
    {"Metric": "📅 Active Days",
     "Overall": f'{active_days} days ({active_days_percent}%)',
     "Last Year": f'{active_days_last_year} days ({active_days_percent_last_year}%)',
     "YTD": f'{active_days_ytd} days ({active_days_percent_ytd}%)'},
])

#######################
# PERSONAL RECORDS
#######################

def _format_pace(pace):
    return f"{int(pace)}:{int(round((pace % 1) * 60)):02d}"

def _format_duration(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _longest_by_type(t):
    sub = df[(df['type'] == t) & df['distance'].notna() & (df['distance'] > 0)]
    if sub.empty:
        return None
    return sub.loc[sub['distance'].idxmax()]

def _fastest_in_band(min_dist, max_dist):
    band = run_df[(run_df['distance'] >= min_dist) & (run_df['distance'] <= max_dist)
                  & run_df['pace_min_per_km'].notna() & np.isfinite(run_df['pace_min_per_km'])]
    if band.empty:
        return None
    return band.loc[band['pace_min_per_km'].idxmin()]

pr_longest_run = _longest_by_type('Run')
pr_longest_ride = _longest_by_type('Ride')
pr_longest_hike = _longest_by_type('Hike')
pr_fastest_5k = _fastest_in_band(4000, 6000)
pr_fastest_10k = _fastest_in_band(9000, 11000)
pr_fastest_long = _fastest_in_band(15000, 1e9)

# Hardest session by TRIMP (training load) — duration-aware unlike raw HR.
pr_hardest = df.loc[df['training_load'].idxmax()] if not df.empty else None

# Steepest hike by vertical metres per km
_hike_pool = df[(df['type'] == 'Hike') & (df['distance'] > 1000) & (df['total_elevation_gain'] > 0)].copy()
if not _hike_pool.empty:
    _hike_pool['m_per_km'] = _hike_pool['total_elevation_gain'] / (_hike_pool['distance'] / 1000)
    pr_steepest_hike = _hike_pool.loc[_hike_pool['m_per_km'].idxmax()]
else:
    pr_steepest_hike = None

# Yearly best paces for the sparkline series on Fastest 5K / 10K / Long cards
def _yearly_best_pace(min_dist, max_dist):
    band = run_df[(run_df['distance'] >= min_dist) & (run_df['distance'] <= max_dist)
                  & run_df['pace_min_per_km'].notna() & np.isfinite(run_df['pace_min_per_km'])]
    if band.empty:
        return None
    out = band.groupby('year')['pace_min_per_km'].min().reset_index()
    out['year'] = out['year'].astype(int)
    return out

pace_progression_5k = _yearly_best_pace(4000, 6000)
pace_progression_10k = _yearly_best_pace(9000, 11000)
pace_progression_long = _yearly_best_pace(15000, 1e9)

#######################
# DASHBOARD LAYOUTS
#######################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Strava Training Dashboard"

# 0. KPI Cards

def kpi_card(title, value, subtitle="", value_color=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-uppercase",
                     style={"color": muted_text_color, "fontSize": "0.72rem",
                            "letterSpacing": "0.06em", "fontWeight": 500}),
            html.Div(value, className="kpi-value mt-1",
                     style={"fontSize": "1.75rem", "fontWeight": 600, "lineHeight": 1.1,
                            "color": value_color or dark_text_color}),
            html.Div(subtitle,
                     style={"color": muted_text_color, "fontSize": "0.78rem",
                            "marginTop": "0.25rem"}),
        ], className="py-3"),
        className="text-center h-100 kpi-card",
        style={"backgroundColor": dark_paper_color,
               "border": f"1px solid {dark_grid_color}",
               "borderRadius": "10px"}
    )

layout_kpi = dbc.Row([
    dbc.Col(kpi_card("Fitness (CTL)", f"{current_fitness:.0f}", "42-day load avg"), md=2),
    dbc.Col(kpi_card("Form (TSB)", f"{current_form:+.0f}", form_label, value_color=form_color), md=2),
    dbc.Col(kpi_card("Load Ratio", f"{acwr:.2f}", acwr_label, value_color=acwr_color), md=2),
    dbc.Col(kpi_card("This Week", f"{this_week_hours:.1f}h",
                     f"{week_sign}{week_delta:.1f}h vs 8-wk avg",
                     value_color=week_color), md=2),
    dbc.Col(kpi_card("Weekly Streak", f"{week_streak} weeks",
                     f"{week_streak_activities} activities"), md=2),
    dbc.Col(kpi_card("Day Streak", f"{current_streak} days",
                     f"ending {last_activity_date.strftime('%b %d')}"), md=2),
], className="g-3 mb-2")

# 1. Activity Type Distribution Charts

def _make_pie(data, title):
    fig = px.pie(
        data, names="type", values="duration_hr",
        title=title, color="type", color_discrete_map=color_map, hole=0.55,
    )
    fig.update_traces(
        texttemplate="%{value:.1f}h",
        textfont=dict(color=dark_text_color, family=chart_font_family, size=12),
        hovertemplate="%{label}: %{value:.1f}h (%{percent})<extra></extra>",
        marker=dict(line=dict(color=dark_paper_color, width=2)),
    )
    fig.update_layout(template=dark_template, showlegend=False, height=320,
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig

layout_pie = dbc.Row([
    dbc.Col(dcc.Graph(figure=_make_pie(pie_df(), f"🌟 All-Time · Since {int(df['year'].min())}")), md=4),
    dbc.Col(dcc.Graph(figure=_make_pie(pie_df("previous"),
        f"🔙 {prev_year} · {prev_year_hours:.1f}h")), md=4),
    dbc.Col(dcc.Graph(figure=_make_pie(pie_df("YTD"),
        f"🔥 {latest_year} YTD · {ytd_hours:.1f}h · {progress_percent} of year")), md=4),
])

# Global range filter — controls fitness, rolling, heatmap, and cumulative charts

def _filter_start_from_value(value):
    if not value or value == "all":
        return None
    try:
        return today - timedelta(days=int(value))
    except (ValueError, TypeError):
        return None

# Initial filter window for the four filter-aware charts on first render —
# matches the dropdown's default value so the page doesn't flash all-time
# data before the callback fires.
INITIAL_FILTER_START = today - timedelta(days=365)

filter_dropdown = html.Div([
    html.Span("Filter range:", className="me-3 fw-semibold",
              style={"color": dark_text_color}),
    dbc.RadioItems(
        id='global-filter',
        options=[
            {"label": "Past Week", "value": "7"},
            {"label": "Past Month", "value": "30"},
            {"label": "Past 3 Months", "value": "90"},
            {"label": "Past 6 Months", "value": "180"},
            {"label": "Past Year", "value": "365"},
            {"label": "All Time", "value": "all"},
        ],
        value="365",
        inline=True,
        className="d-inline-block",
        labelClassName="me-3",
    )
], className="d-flex align-items-center justify-content-center my-3 px-3 py-2",
   style={"backgroundColor": dark_paper_color, "border": f"1px solid {dark_grid_color}",
          "borderRadius": "6px"})

# 2. Fitness / Fatigue / Form — explainer + 30-day focused form view + full chart

# Single source of truth for the fitness explainer markdown — used here by the
# Dash app and by build_static.py for the static HTML export.
FITNESS_EXPLAINER_MD = (
    "**Fitness (CTL)** — 42-day rolling training load. Your long-term base.\n"
    "**Fatigue (ATL)** — 7-day rolling load. How loaded your body is right now.\n"
    "**Form (TSB) = Fitness − Fatigue.** Positive = fresh, negative = tired.\n"
    "**Load Ratio (ACWR)** — 7-day avg load ÷ 28-day avg load. Tells you whether "
    "you're ramping up safely or piling on too fast.\n\n"
    "**Form bands:** above **+5** fresh / peaked &nbsp;·&nbsp; **+5 to −10** "
    "optimal &nbsp;·&nbsp; **−10 to −30** productive (building) &nbsp;·&nbsp; "
    "below **−30** overreaching.\n"
    "**Load Ratio bands:** below **0.8** undertraining &nbsp;·&nbsp; **0.8 to 1.3** "
    "optimal &nbsp;·&nbsp; **1.3 to 1.5** caution &nbsp;·&nbsp; above **1.5** "
    "elevated injury risk."
)

fitness_explainer = dcc.Markdown(
    FITNESS_EXPLAINER_MD,
    className="px-4 py-2 mb-2",
    style={"color": dark_text_color, "fontSize": "0.92rem", "lineHeight": "1.6"}
)

# Last-30-day Form chart with zone shading
form_recent_start = today - timedelta(days=30)
form_recent = daily_scores[daily_scores['date'] >= form_recent_start].copy()

zone_specs = [
    (5, 200, "Fresh", SERIES["fresh"]),
    (-10, 5, "Optimal", SERIES["optimal"]),
    (-30, -10, "Productive", SERIES["productive"]),
    (-200, -30, "Overreaching", SERIES["overreach"]),
]

form_y_min = int(min(form_recent['form'].min(), -35)) - 5
form_y_max = int(max(form_recent['form'].max(), 12)) + 5

# Combined Form + Load Ratio chart: Form (TSB) on the primary y-axis with
# its zone shading; Load Ratio (ACWR) overlaid on a secondary y-axis on the
# right so both metrics share a single x-axis and today-marker context.
daily_scores['acwr'] = (
    daily_scores['training_load'].rolling(7, min_periods=1).mean()
    / daily_scores['training_load'].rolling(28, min_periods=1).mean().replace(0, np.nan)
)
acwr_recent = daily_scores[daily_scores['date'] >= form_recent_start].copy()

# Load Ratio axis aligned to the Form colour bands. Two-point linear pin:
#   0.8 ↔ Form +5  (Fresh / Optimal boundary)
#   1.5 ↔ Form −30 (Productive / Overreaching boundary)
# Slope: 35 form units = 0.7 load units → 50 form per 1 load. The inner
# cutoff 1.3 lands at Form ≈ −20 instead of −10 (~6% chart offset), an
# accepted tradeoff for getting the outer "danger" lines to align exactly.
_acwr_axis_top    = 0.8 + (5 - form_y_max) / 50      # low load, top of chart
_acwr_axis_bottom = 0.8 + (5 - form_y_min) / 50      # high load, bottom of chart

combined_form_acwr_fig = make_subplots(specs=[[{"secondary_y": True}]])

# Form zone shading on the PRIMARY y-axis. With the secondary (ACWR) axis
# inverted, the bottom red stripe applies to Overreaching Form AND High-Risk
# Load simultaneously. Using add_shape with explicit yref="y" — add_hrect on
# a make_subplots-with-secondary_y figure can drop these bands silently.
for y0, y1, _, hex_color in zone_specs:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    combined_form_acwr_fig.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, x1=1, y0=y0, y1=y1,
        fillcolor=f"rgba({r},{g},{b},0.32)",
        line_width=0, layer="below",
    )

# Form line + marker (left axis)
combined_form_acwr_fig.add_trace(
    go.Scatter(
        x=form_recent['date'], y=form_recent['form'].round(1),
        mode="lines",
        line=dict(color=SERIES["fitness"], width=2.5, shape="spline", smoothing=0.4),
        hovertemplate="Form: %{y:+.1f}<extra></extra>",
        name="Form (TSB)",
    ),
    secondary_y=False,
)
combined_form_acwr_fig.add_trace(
    go.Scatter(
        x=[form_recent['date'].iloc[-1]],
        y=[round(form_recent['form'].iloc[-1], 1)],
        mode="markers",
        marker=dict(color=form_color, size=14, line=dict(color=dark_paper_color, width=2)),
        hovertemplate=f"Today's Form: {current_form:+.1f} · {form_label}<extra></extra>",
        showlegend=False,
    ),
    secondary_y=False,
)

# Load Ratio line + marker (right axis)
combined_form_acwr_fig.add_trace(
    go.Scatter(
        x=acwr_recent['date'], y=acwr_recent['acwr'].round(2),
        mode="lines",
        line=dict(color=SERIES["fatigue"], width=2, dash="dot"),
        hovertemplate="Load Ratio: %{y:.2f}<extra></extra>",
        name="Load Ratio (ACWR)",
    ),
    secondary_y=True,
)
combined_form_acwr_fig.add_trace(
    go.Scatter(
        x=[acwr_recent['date'].iloc[-1]],
        y=[round(acwr_recent['acwr'].iloc[-1], 2)
           if pd.notna(acwr_recent['acwr'].iloc[-1]) else None],
        mode="markers",
        marker=dict(color=acwr_color, size=12, line=dict(color=dark_paper_color, width=2)),
        hovertemplate=f"Today's Load: {acwr:.2f} · {acwr_label}<extra></extra>",
        showlegend=False,
    ),
    secondary_y=True,
)

# Zero line for Form on the left axis
combined_form_acwr_fig.add_hline(
    y=0, line=dict(color=SERIES["form_zero"], width=1, dash="dash"),
    secondary_y=False,
)

# Zone labels INSIDE the plot area at the left edge — keeps them readable and
# stops them clipping outside when chart is half-width in a side-by-side row.
zone_label_positions = [
    (max(form_y_max - 4, 8), "Fresh", SERIES["fresh"]),
    (-2.5, "Optimal", SERIES["optimal"]),
    (-20, "Productive", SERIES["productive"]),
    (min(form_y_min + 4, -36), "Overreaching", SERIES["overreach"]),
]
for y, label, color in zone_label_positions:
    if form_y_min <= y <= form_y_max:
        combined_form_acwr_fig.add_annotation(
            xref="paper", yref="y",
            x=0.01, y=y, xanchor="left", yanchor="middle",
            text=label, showarrow=False,
            font=dict(color=color, size=10, family=chart_font_family, weight=600),
        )

# Load Ratio (ACWR) zone labels — RHS, mirroring the Form labels. The right
# axis is inverted, so "Risk" sits at the bottom (aligned with Overreaching)
# and "Undertraining" at the top (aligned with Fresh).
acwr_zone_label_positions = [
    (0.7, "Undertraining", SERIES["fresh"]),
    (1.05, "Optimal", SERIES["optimal"]),
    (1.4, "Caution", SERIES["productive"]),
    (1.6, "Risk", SERIES["overreach"]),
]
for y, label, color in acwr_zone_label_positions:
    if _acwr_axis_top <= y <= _acwr_axis_bottom:
        combined_form_acwr_fig.add_annotation(
            xref="paper", yref="y2",
            x=0.99, y=y, xanchor="right", yanchor="middle",
            text=label, showarrow=False,
            font=dict(color=color, size=10, family=chart_font_family, weight=600),
        )

combined_form_acwr_fig.update_layout(
    template=dark_template,
    title_text=(
        f"🎯 Form & Load Ratio · Last 30 Days &nbsp;&nbsp;"
        f"<span style='color:{form_color}; font-weight:600'>Form {current_form:+.0f} · {form_label}</span>"
        f"&nbsp;&nbsp;<span style='color:{acwr_color}; font-weight:600'>Load {acwr:.2f} · {acwr_label}</span>"
    ),
    xaxis=dict(title_text=None, tickformat="%b %d"),
    height=380,
    hovermode="x unified",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    # Wider l/r margins so both axis titles render fully without clipping
    margin=dict(l=80, r=80, t=70, b=20),
)
combined_form_acwr_fig.update_yaxes(
    title_text="Form (TSB)", secondary_y=False,
    range=[form_y_min, form_y_max], tickformat=".0f",
)
# Right axis is INVERTED so high Load Ratio (high injury risk = bad) sits at
# the BOTTOM, aligning with low Form (overreaching = also bad). The "danger"
# end is the same direction on both axes, which makes the dual-line read
# intuitively: lines that drop together = stress accumulating.
combined_form_acwr_fig.update_yaxes(
    title_text="Load Ratio (ACWR)", secondary_y=True,
    range=[_acwr_axis_bottom, _acwr_axis_top],
    tickmode='array', tickvals=[0.8, 1.0, 1.3, 1.5], tickformat=".1f",
    showgrid=False,
)

layout_form_recent = dbc.Row([dbc.Col(dcc.Graph(figure=combined_form_acwr_fig), md=12)])

# 2b. Full Fitness Chart (CTL / ATL / TSB + monthly bars) — global-filter aware

def build_fitness_fig(start_date):
    if start_date:
        ma = monthly_activity[monthly_activity['month'] >= pd.Timestamp(start_date)]
        ds = daily_scores[daily_scores['date'] >= start_date]
    else:
        ma = monthly_activity
        ds = daily_scores

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for activity_type in df['type'].unique():
        activity_data = ma[ma['type'] == activity_type]
        fig.add_trace(
            go.Bar(
                x=activity_data['month'], y=activity_data['duration_hr'],
                name=activity_type,
                marker=dict(
                    color=color_map.get(activity_type, '#666666'),
                    line=dict(width=1, color=dark_paper_color),
                ),
                opacity=0.9,
                hovertemplate=f"<b>{activity_type}</b>: %{{y:.1f}}h<br>%{{x|%b %Y}}<extra></extra>",
                showlegend=False,
            ),
            secondary_y=False
        )

    fig.add_trace(go.Scatter(
        x=ds['date'], y=ds['fitness'].round(1), name="Fitness (CTL · 42d)",
        line=dict(color=SERIES["fitness"], width=2.5),
        hovertemplate="Fitness: %{y:.0f}<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=ds['date'], y=ds['fatigue'].round(1), name="Fatigue (ATL · 7d)",
        line=dict(color=SERIES["fatigue"], width=1.8, dash="dot"),
        hovertemplate="Fatigue: %{y:.0f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        template=dark_template,
        title_text="💪 Fitness · Fatigue · Monthly Volume",
        barmode='stack',
        bargap=0.15,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=70, r=70, t=70, b=20),
        height=440,
    )
    fig.update_xaxes(title_text=None, showgrid=False)
    fig.update_yaxes(title_text="Hours", secondary_y=False, rangemode="tozero",
                     tickformat=".0f")
    fig.update_yaxes(title_text="Score", secondary_y=True, showgrid=False,
                     tickformat=".0f")
    return fig

# layout_fitness_yoy_row (fitness chart side-by-side with YoY trajectory) is
# assembled after yoy_fig is built — see further down in this file.

# 3. Rolling Volume Chart — global-filter aware

# 4. Day-of-Week × Hour Heatmap — global-filter aware

def build_heatmap_fig(start_date):
    df_f = df[df['date'] >= start_date] if start_date else df
    pivot = (
        df_f.groupby(['weekday_idx', 'hour_of_day'])['duration_hr'].sum()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}" for h in range(24)],
        y=weekday_order,
        colorscale=[
            [0.00, dark_paper_color],
            [0.05, "#1F2937"],
            [0.30, "#4338CA"],
            [0.60, "#A855F7"],
            [0.85, "#F97316"],
            [1.00, "#FACC15"],
        ],
        hovertemplate="<b>%{y} · %{x}:00</b><br>%{z:.1f} hours<extra></extra>",
        colorbar=dict(title=dict(text="Hours", font=dict(size=11)),
                      thickness=12, len=0.8, outlinewidth=0,
                      tickfont=dict(size=10)),
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        template=dark_template,
        title_text="🕓 When Do You Train? · Day × Hour",
        height=400,
    )
    fig.update_xaxes(title_text=None, tickfont=dict(size=11), showgrid=False)
    fig.update_yaxes(title_text=None, autorange="reversed", showgrid=False)
    return fig

# layout_heatmap defined below as part of layout_heatmap_cumulative

# 5. Cumulative Stats — global-filter aware (cumsum resets at filter window start)

def _empty_fig(msg):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=dark_paper_color, plot_bgcolor=dark_bg_color, font=chart_font,
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(color=muted_text_color, size=14,
                                                     family=chart_font_family))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10), height=320
    )
    return fig

def _build_cumulative_pivot(start_date, value_col):
    """Cumulative pivot of `value_col` per type, starting cumsum at the filter window."""
    src = df.sort_values("start_date_local").copy()
    src["activity_count"] = 1
    if start_date:
        src = src[src['date'] >= start_date]
    if src.empty:
        return None
    start_d = start_date if start_date else df['date'].min()
    end_d = df['date'].max()
    dates_idx = pd.date_range(start=start_d, end=end_d, freq='D').date
    date_df_local = pd.DataFrame({"date": dates_idx})

    grouped = src.groupby(["date", "type"])[value_col].sum().reset_index()
    pivot = grouped.pivot(index='date', columns='type', values=value_col).fillna(0)
    pivot = pd.merge(date_df_local, pivot.reset_index(), on='date', how='left').fillna(0)
    pivot = pivot.sort_values('date')
    for col in pivot.columns:
        if col != 'date':
            pivot[col] = pivot[col].cumsum()
    return pivot

def build_cumulative_time_fig(start_date):
    pivot = _build_cumulative_pivot(start_date, "duration_hr")
    if pivot is None:
        return _empty_fig("No activities in range")
    fig = px.area(
        pivot, x="date", y=pivot.columns[1:],
        title="⏱️ Training Hours · Cumulative by Activity",
        color_discrete_map=color_map,
        labels={"value": "Hours", "date": "", "variable": "Activity Type"},
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b>: %{y:.2f}h<br>%{x|%b %d, %Y}<extra></extra>",
        line=dict(width=0),
        opacity=0.9,
    )
    # Annotate the total at the right edge so the headline number is visible
    _final_total = float(pivot[pivot.columns[1:]].iloc[-1].sum())
    fig.add_annotation(
        x=pivot['date'].iloc[-1], y=_final_total,
        text=f"<b>{_final_total:.0f}h</b>",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color=dark_text_color, size=12, family=chart_font_family),
        bgcolor=dark_paper_color, bordercolor=dark_grid_color, borderwidth=1, borderpad=4,
    )
    fig.update_layout(
        template=dark_template,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=10)),
        height=400,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=70, b=20),
    )
    fig.update_xaxes(title_text=None, showgrid=False)
    fig.update_yaxes(title_text="Hours", rangemode="tozero", tickformat=".0f")
    return fig

# Heatmap + cumulative time side-by-side
layout_heatmap_cumulative = dbc.Row([
    dbc.Col(dcc.Graph(id='heatmap-graph', figure=build_heatmap_fig(INITIAL_FILTER_START)), md=6),
    dbc.Col(dcc.Graph(id='cumulative-time-graph', figure=build_cumulative_time_fig(INITIAL_FILTER_START)), md=6),
])

# 6. Run Performance — pace axis formatted as MM:SS, year as discrete colors
pace_ticks = list(np.arange(3.0, 9.5, 0.5))
pace_tick_labels = [_format_pace(p) for p in pace_ticks]

def _safe_pace_str(p):
    if pd.isna(p) or not np.isfinite(p):
        return "—"
    return _format_pace(p)

# Named figures (extracted so the static-export script can grab them).
scatter_pace_fig = px.scatter(
    run_df,
    x="date", y="pace_min_per_km",
    size="distance", color="average_heartrate",
    title="🏃 Run Pace Over Time",
    labels={"pace_min_per_km": "Pace", "date": "",
            "average_heartrate": "Avg HR", "distance": "Distance (m)"},
    color_continuous_scale="RdYlBu_r",
    hover_data=["name", "distance", "average_heartrate", "moving_time"]
).update_traces(
    marker=dict(opacity=0.75, line=dict(width=0.5, color=dark_paper_color), sizemin=4),
    selector=dict(mode='markers'),
    hovertemplate=
        "<b>%{customdata[0]} /km</b><br>"
        "%{customdata[1]:.1f} km · %{customdata[3]}<br>"
        "Avg HR: %{customdata[2]}<extra></extra>",
    customdata=run_df.apply(lambda x: [
        _safe_pace_str(x['pace_min_per_km']),
        (x['distance'] or 0) / 1000,
        f"{int(x['average_heartrate'])} bpm" if pd.notna(x['average_heartrate']) else "—",
        f"{int(x['moving_time']//60):02d}:{int(x['moving_time']%60):02d}"
    ], axis=1).tolist()
).update_layout(
    template=dark_template,
    height=440,
    xaxis=dict(title_text=None),
    yaxis=dict(title_text="Pace (min/km)", autorange="reversed",
               tickvals=pace_ticks, ticktext=pace_tick_labels),
    coloraxis_colorbar=dict(
        title=dict(text="Avg HR", font=dict(size=11)),
        thickness=12, len=0.8, outlinewidth=0,
        tickformat="d", tickfont=dict(size=10),
    ),
)

scatter_efficiency_fig = px.scatter(
    run_df,
    x="average_heartrate", y="pace_min_per_km",
    color="year_str", size="distance",
    color_discrete_map=year_color_map,
    category_orders={"year_str": [str(y) for y in sorted_years]},
    title="❤️ Run Efficiency · Pace vs HR",
    labels={"pace_min_per_km": "Pace", "average_heartrate": "Avg HR (bpm)",
            "year_str": "Year", "distance": "Distance (m)"},
    hover_data=["name", "date", "distance", "moving_time"]
).update_traces(
    marker=dict(opacity=0.78, line=dict(width=0.5, color=dark_paper_color), sizemin=4),
    selector=dict(mode='markers'),
    hovertemplate=
        "<b>%{customdata[0]} /km @ %{x:.0f} bpm</b><br>"
        "%{customdata[2]:.1f} km · %{customdata[3]}<br>"
        "%{customdata[1]}<extra></extra>",
    customdata=run_df.apply(lambda x: [
        _safe_pace_str(x['pace_min_per_km']),
        x['date'],
        (x['distance'] or 0) / 1000,
        f"{int(x['moving_time']//60):02d}:{int(x['moving_time']%60):02d}"
    ], axis=1).tolist()
).update_layout(
    template=dark_template,
    height=440,
    xaxis=dict(title_text="Avg HR (bpm)", showgrid=True),
    yaxis=dict(title_text="Pace (min/km)", autorange="reversed",
               tickvals=pace_ticks, ticktext=pace_tick_labels),
    legend=dict(title=dict(text="Year")),
)

layout_scatter = dbc.Row([
    dbc.Col(dcc.Graph(figure=scatter_pace_fig), md=6),
    dbc.Col(dcc.Graph(figure=scatter_efficiency_fig), md=6),
])

# 7. Year activity heatmap — past 365 days × 7-row weekday grid. Two
# interchangeable views:
#   • Load view (default): single-colour cell per day, intensity by TRIMP
#   • Activities view: each day's cell is split into sub-rectangles, one per
#     activity that day, coloured by activity type
# Both views share the same axis layout and an enriched tooltip per cell.
_year_end = today
_year_start_aligned = _year_end - timedelta(days=365 - 1)
_year_start_aligned -= timedelta(days=_year_start_aligned.weekday())  # back to Monday

_daily_load_full = df.groupby('date')['training_load'].sum()

# Pre-compute per-day activity breakdowns for both tooltips and the split cells
_acts_by_date = {
    d: g.sort_values('duration_hr', ascending=False)
    for d, g in df[(df['date'] >= _year_start_aligned) & (df['date'] <= _year_end)].groupby('date')
}

def _cell_tooltip(day, acts):
    """Format a per-cell hover tooltip: date, each activity's hours/load, totals."""
    if acts is None or acts.empty:
        return f"<b>{day.strftime('%a · %b %d, %Y')}</b><br>Rest day"
    lines = [f"<b>{day.strftime('%a · %b %d, %Y')}</b>"]
    for _, act in acts.iterrows():
        lines.append(
            f"&nbsp;&nbsp;{act['type']}: {act['duration_hr']:.2f}h · "
            f"load {act['training_load']:.0f}"
        )
    total_dur = float(acts['duration_hr'].sum())
    total_load = float(acts['training_load'].sum())
    lines.append(f"<b>Total: {total_dur:.2f}h · load {total_load:.0f}</b>")
    return "<br>".join(lines)

_week_starts = []
_year_z = [[] for _ in range(7)]            # load values for the colour view
_year_tooltips = [[] for _ in range(7)]     # rich tooltip per cell, shared
_cursor = _year_start_aligned
while _cursor <= _year_end:
    _week_starts.append(_cursor)
    for _d in range(7):
        _day = _cursor + timedelta(days=_d)
        if _day > _year_end:
            _year_z[_d].append(None)
            _year_tooltips[_d].append("")
        else:
            _year_z[_d].append(float(_daily_load_full.get(_day, 0)))
            _year_tooltips[_d].append(_cell_tooltip(_day, _acts_by_date.get(_day)))
    _cursor += timedelta(days=7)

# Tick at the first week of each month
_month_tick_idx = []
_month_tick_text = []
_seen_months = set()
for i, ws in enumerate(_week_starts):
    key = (ws.year, ws.month)
    if key not in _seen_months:
        _seen_months.add(key)
        _month_tick_idx.append(i)
        _month_tick_text.append(ws.strftime('%b'))

# Shared axis kwargs so both views look identical
_year_xaxis_kwargs = dict(
    tickmode='array', tickvals=_month_tick_idx, ticktext=_month_tick_text,
    showgrid=False, title_text=None, zeroline=False, tickfont=dict(size=10),
    range=[-0.5, len(_week_starts) - 0.5],
)
_year_yaxis_kwargs = dict(
    tickmode='array', tickvals=list(range(7)),
    ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    showgrid=False, title_text=None, autorange='reversed', tickfont=dict(size=10),
    # Suppress the default zeroline at y=0 — otherwise it bisects Monday cells
    # in the Activities view (shapes are layer="below" so the line draws over).
    zeroline=False,
)

# --- Load view: a Heatmap coloured by daily TRIMP, custom-tooltip per cell.
year_heatmap_load_fig = go.Figure(data=go.Heatmap(
    z=_year_z,
    x=list(range(len(_week_starts))),
    y=list(range(7)),  # numerical so it lines up with the activities-view shapes
    customdata=_year_tooltips,
    colorscale=[
        [0.00, "#1F2937"],     # no activity → dark grey
        [0.0001, "#FED7AA"],   # any activity → light amber
        [0.30, "#FB923C"],     # moderate → orange
        [0.70, "#DC2626"],     # high → red
        [1.00, "#7F1D1D"],     # peak → deep red
    ],
    hovertemplate="%{customdata}<extra></extra>",
    xgap=2, ygap=2,
    colorbar=dict(title=dict(text="Load", font=dict(size=11)),
                  thickness=12, len=0.8, outlinewidth=0,
                  tickfont=dict(size=10)),
))
year_heatmap_load_fig.update_layout(
    template=dark_template,
    title_text="📅 Training Year · Past 365 Days",
    height=240,
    xaxis=_year_xaxis_kwargs, yaxis=_year_yaxis_kwargs,
)

# --- Activities view: rectangle shapes split horizontally inside each cell —
# one strip per activity on that day, coloured by activity type. A transparent
# scatter layer above provides hover for the shared tooltip.
year_heatmap_activities_fig = go.Figure()
_cell_pad = 0.42  # half-height/width of a cell (slight margin for visual gap)
_hover_x, _hover_y, _hover_text = [], [], []
for _w_idx, _ws in enumerate(_week_starts):
    for _d in range(7):
        _day = _ws + timedelta(days=_d)
        if _day > _year_end:
            continue
        _acts = _acts_by_date.get(_day)
        if _acts is not None and not _acts.empty:
            _n = len(_acts)
            for _i, (_, _act) in enumerate(_acts.iterrows()):
                _frac_lo = _i / _n
                _frac_hi = (_i + 1) / _n
                year_heatmap_activities_fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=_w_idx - _cell_pad + (_frac_lo * 2 * _cell_pad),
                    x1=_w_idx - _cell_pad + (_frac_hi * 2 * _cell_pad),
                    y0=_d - _cell_pad, y1=_d + _cell_pad,
                    fillcolor=color_map.get(_act['type'], '#666666'),
                    line_width=0, layer="below",
                )
        _hover_x.append(_w_idx)
        _hover_y.append(_d)
        _hover_text.append(_cell_tooltip(_day, _acts))

# Invisible square markers anchor the hover tooltips
year_heatmap_activities_fig.add_trace(go.Scatter(
    x=_hover_x, y=_hover_y, mode='markers',
    marker=dict(size=18, opacity=0, symbol='square'),
    text=_hover_text,
    hovertemplate="%{text}<extra></extra>",
    showlegend=False,
))

# Legend chips — one transparent-marker trace per activity type so the colour
# key sits in a horizontal legend above the chart.
_legend_types_present = sorted(set(
    t for _acts in _acts_by_date.values() for t in _acts['type'].unique()
))
for _t in _legend_types_present:
    year_heatmap_activities_fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color=color_map.get(_t, '#666666'), symbol='square'),
        name=_t, showlegend=True, hoverinfo='skip',
    ))

year_heatmap_activities_fig.update_layout(
    template=dark_template,
    title_text="📅 Training Year · Past 365 Days · By Activity",
    height=280,  # taller to make room for the legend
    xaxis=_year_xaxis_kwargs, yaxis=_year_yaxis_kwargs,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                font=dict(size=10)),
)

# Layout with a view toggle that flips between the two figures
layout_year_heatmap = html.Div([
    html.Div([
        dbc.RadioItems(
            id='year-heatmap-view',
            options=[
                {'label': 'Load', 'value': 'load'},
                {'label': 'Activity', 'value': 'activities'},
            ],
            value='load',
            inline=True,
            labelClassName="me-3 small",
            inputClassName="form-check-input",
        ),
    ], className="d-flex justify-content-end mb-1"),
    dcc.Graph(id='year-heatmap-graph', figure=year_heatmap_load_fig),
])


# 8. Bonus stats tables — activity feed: 5 rows, Strava-style detail
recent_activities = df.sort_values("start_date_local", ascending=False).head(5)
activity_emoji = {
    'Run': '🏃', 'Ride': '🚴', 'Racquet Sports': '🏓', 'Cardio': '💪',
    'Weight Training': '🏋️', 'Hike': '🥾', 'Workout': '🤸', 'Walk': '🚶'
}

def _feed_distance(row):
    if pd.notna(row['distance']) and row['distance'] > 0:
        return f"{row['distance']/1000:.2f} km"
    return "—"

def _feed_pace_or_hr(row):
    """For runs/walks/hikes: pace. Otherwise: avg HR if available."""
    if row['type'] in ('Run', 'Walk', 'Hike') and pd.notna(row.get('distance')) and row['distance'] > 0:
        pace = row['moving_time'] / 60 / (row['distance'] / 1000)
        if np.isfinite(pace):
            return f"{_format_pace(pace)} /km"
    if pd.notna(row.get('average_heartrate')):
        return f"{int(row['average_heartrate'])} bpm"
    return "—"

def _feed_vert(row):
    v = row.get('total_elevation_gain')
    if pd.notna(v) and v > 0:
        return f"↑ {int(v)} m"
    return "—"

recent_activities_df = pd.DataFrame({
    "Date": recent_activities["start_date_local"].dt.strftime("%b %d"),
    "Activity": recent_activities.apply(
        lambda row: f"{activity_emoji.get(row['type'], '🏋️')} {row['name']}", axis=1
    ),
    "Duration": recent_activities["elapsed_time"].apply(lambda s: _format_duration(s)),
    "Distance": recent_activities.apply(_feed_distance, axis=1),
    "Pace / HR": recent_activities.apply(_feed_pace_or_hr, axis=1),
    "Vert": recent_activities.apply(_feed_vert, axis=1),
})

table_cell_style = {
    "textAlign": "left", "padding": "5px",
    "backgroundColor": dark_paper_color, "color": dark_text_color
}
table_header_style = {
    "fontWeight": "bold", "backgroundColor": "#2C2C2C", "color": dark_text_color
}

layout_bonus = dbc.Row([
    dbc.Col([
        html.H4("🏆 Overall Stats", className="text-center my-3"),
        dash_table.DataTable(
            data=individual_stats_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in individual_stats_df.columns],
            style_table={"width": "100%", "margin": "0 auto"},
            style_cell=table_cell_style,
            style_header=table_header_style,
        )
    ], md=6, className="mb-4"),

    dbc.Col([
        html.H4("⏱️ Recent Activities", className="text-center my-3"),
        dash_table.DataTable(
            data=recent_activities_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in recent_activities_df.columns],
            style_table={"width": "100%", "margin": "0 auto", "marginBottom": "20px"},
            style_cell=table_cell_style,
            style_header=table_header_style,
        )
    ], md=6)
], className="mt-4")

layout_yoy = dbc.Row([
    dbc.Col([], md=3),
    dbc.Col([
        html.H4("📊 Year-over-Year Comparison", className="text-center my-3"),
        dash_table.DataTable(
            data=comparative_stats_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in comparative_stats_df.columns],
            style_table={"width": "100%", "margin": "0 auto"},
            style_cell=table_cell_style,
            style_header=table_header_style,
        )
    ], md=6),
    dbc.Col([], md=3),
], className="mt-4")

# 9. Personal Records grid

def _pace_sparkline(progression, accent_color):
    """Tiny line chart of yearly-best pace for a distance band. Lower = better, so y-axis is reversed."""
    if progression is None or progression.empty or len(progression) < 2:
        return None
    fig = go.Figure(go.Scatter(
        x=progression['year'].astype(str),
        y=progression['pace_min_per_km'].round(2),
        mode='lines+markers',
        line=dict(color=accent_color, width=2),
        marker=dict(size=5, color=accent_color),
        hovertemplate="%{x}: %{customdata}<extra></extra>",
        customdata=[_format_pace(p) for p in progression['pace_min_per_km']],
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=4),
        height=56,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=chart_font,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   autorange='reversed', fixedrange=True),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1F2937", bordercolor=dark_grid_color,
                        font=dict(family=chart_font_family, color=dark_text_color, size=11)),
    )
    return fig

def _distribution_spark(values, pr_value, accent_color, unit="km", value_fmt=".1f"):
    """Small histogram of all values for context, with a vertical marker at the PR.
    Visualises how rare/extreme the PR is relative to the rest of the distribution."""
    if values is None or pr_value is None:
        return None
    values = pd.Series(values).dropna()
    if len(values) < 3:
        return None
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=max(10, min(24, len(values) // 5)),
        marker=dict(color="#374151", line=dict(width=0)),
        opacity=0.9,
        showlegend=False,
        hovertemplate=f"%{{x:{value_fmt}}} {unit}: %{{y}} sessions<extra></extra>",
    ))
    fig.add_vline(x=pr_value, line=dict(color=accent_color, width=2))
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=4),
        height=56,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=chart_font,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
        showlegend=False,
        bargap=0.06,
        hoverlabel=dict(bgcolor="#1F2937", bordercolor=dark_grid_color,
                        font=dict(family=chart_font_family, color=dark_text_color, size=11)),
    )
    return fig

def pr_card(title, value, subtitle="", spark=None):
    body = [
        html.Div(title, className="text-uppercase",
                 style={"color": muted_text_color, "fontSize": "0.72rem",
                        "letterSpacing": "0.06em", "fontWeight": 500}),
        html.Div(value, className="pr-value mt-1",
                 style={"fontSize": "1.35rem", "fontWeight": 600, "lineHeight": 1.2,
                        "color": dark_text_color}),
        html.Div(subtitle,
                 style={"color": muted_text_color, "fontSize": "0.78rem",
                        "marginTop": "0.25rem"}),
    ]
    if spark is not None:
        body.append(dcc.Graph(
            figure=spark,
            config={'displayModeBar': False, 'staticPlot': False},
            style={'height': '56px', 'marginTop': '6px'},
        ))
    return dbc.Card(
        dbc.CardBody(body, className="py-3"),
        className="text-center h-100 pr-card",
        style={"backgroundColor": dark_paper_color,
               "border": f"1px solid {dark_grid_color}",
               "borderRadius": "10px"}
    )

def _fmt_longest(row, include_vert=False):
    if row is None:
        return ("—", "")
    main = f"{row['distance']/1000:.2f} km"
    if include_vert and pd.notna(row.get('total_elevation_gain')) and row['total_elevation_gain'] > 0:
        main += f" · {int(row['total_elevation_gain'])} m"
    return (main, f"{row['date']}")

def _fmt_pace(row):
    if row is None:
        return ("—", "")
    return (f"{_format_pace(row['pace_min_per_km'])} /km",
            f"{row['distance']/1000:.2f} km in {_format_duration(row['moving_time'])} on {row['date']}")

def _fmt_steepest(row):
    if row is None:
        return ("—", "")
    return (f"{row['m_per_km']:.0f} m/km",
            f"{int(row['total_elevation_gain'])} m over {row['distance']/1000:.2f} km on {row['date']}")

def _fmt_hardest(row):
    if row is None:
        return ("—", "", "")
    score = row['training_load']
    main = f"{int(round(score))} TRIMP"
    sub_bits = []
    if pd.notna(row.get('average_heartrate')):
        sub_bits.append(f"{int(row['average_heartrate'])} bpm avg")
    sub_bits.append(_format_duration(row['elapsed_time']))
    sub = f"{row['type']} · " + " · ".join(sub_bits) + f" on {row['date']}"
    return (main, sub)

# Distribution sparks for the extreme-value PR cards: show how rare/extreme the
# record is relative to the full set of sessions of that type.
_run_distances_km = df.loc[(df['type'] == 'Run') & (df['distance'] > 0), 'distance'] / 1000
_hike_distances_km = df.loc[(df['type'] == 'Hike') & (df['distance'] > 0), 'distance'] / 1000

_hike_for_steepness = df[(df['type'] == 'Hike') & (df['distance'] > 1000)
                         & (df['total_elevation_gain'] > 0)].copy()
_hike_for_steepness['m_per_km'] = (
    _hike_for_steepness['total_elevation_gain'] / (_hike_for_steepness['distance'] / 1000)
)

dist_spark_run = _distribution_spark(
    _run_distances_km,
    pr_longest_run['distance'] / 1000 if pr_longest_run is not None else None,
    color_map.get('Run'),
)
dist_spark_hike = _distribution_spark(
    _hike_distances_km,
    pr_longest_hike['distance'] / 1000 if pr_longest_hike is not None else None,
    color_map.get('Hike'),
)
dist_spark_steep = _distribution_spark(
    _hike_for_steepness['m_per_km'],
    pr_steepest_hike['m_per_km'] if pr_steepest_hike is not None else None,
    color_map.get('Hike'),
    unit="m/km", value_fmt=".0f",
)

# Hardest session: TRIMP distribution across all activities — the PR will sit
# in the long right tail showing how exceptional it is.
dist_spark_hardest = _distribution_spark(
    df['training_load'],
    pr_hardest['training_load'] if pr_hardest is not None else None,
    SERIES["overreach"],
    unit="TRIMP", value_fmt=".0f",
)

# Biggest week: weekly hours distribution with PR marked.
dist_spark_week = _distribution_spark(
    weekly_hours,
    busiest_week_total,
    SERIES["fresh"],
    unit="h", value_fmt=".1f",
)

prs_with_sparks = [
    ("🏃 Longest Run", *_fmt_longest(pr_longest_run), dist_spark_run),
    ("🚴 Longest Ride", *_fmt_longest(pr_longest_ride), None),
    ("🥾 Longest Hike", *_fmt_longest(pr_longest_hike, include_vert=True), dist_spark_hike),
    ("⚡ Fastest 5K", *_fmt_pace(pr_fastest_5k),
     _pace_sparkline(pace_progression_5k, SERIES["rolling_7"])),
    ("⚡ Fastest 10K", *_fmt_pace(pr_fastest_10k),
     _pace_sparkline(pace_progression_10k, SERIES["rolling_28"])),
    ("⚡ Fastest Long Run", *_fmt_pace(pr_fastest_long),
     _pace_sparkline(pace_progression_long, SERIES["rolling_365"])),
    ("⛰️ Steepest Hike", *_fmt_steepest(pr_steepest_hike), dist_spark_steep),
    ("📈 Biggest Week", f"{busiest_week_total}h", f"{busiest_week_start.date()}", dist_spark_week),
    ("🔥 Hardest Session", *_fmt_hardest(pr_hardest), dist_spark_hardest),
]

layout_prs = html.Div([
    html.H4("🏆 Personal Records", className="text-center my-3"),
    dbc.Row(
        [dbc.Col(pr_card(t, v, s, spark=spark), md=4, className="mb-3")
         for (t, v, s, spark) in prs_with_sparks],
        className="g-3"
    )
])

#######################
# NARRATIVE INSIGHTS
#######################

_insights = []

# 1) Fitness change in the last 30 days
_thirty_back = today - timedelta(days=30)
_past_row = daily_scores[daily_scores['date'] == _thirty_back]
if not _past_row.empty:
    _past_fitness = _past_row['fitness'].iloc[0]
    _delta = current_fitness - _past_fitness
    _verb = "climbed" if _delta >= 0 else "dropped"
    _sign = "+" if _delta >= 0 else "−"
    _insights.append(
        ("🏋️",
         f"Your **fitness score** (CTL — your 42-day rolling training load) has "
         f"**{_verb} from {_past_fitness:.0f} to {current_fitness:.0f}** in the last 30 days "
         f"({_sign}{abs(_delta):.0f}).")
    )

# 2) Most common training day
if not df.empty:
    _wkday_counts = df['weekday'].value_counts()
    _top_day = _wkday_counts.idxmax()
    _top_pct = round(_wkday_counts.max() / len(df) * 100)
    _insights.append(
        ("📅", f"Your busiest training day is **{_top_day}** — {_top_pct}% of all sessions.")
    )

# 3) Pace at moderate effort (~150 bpm) compared to last year
def _avg_pace_at_hr(year, hr_target=150, tolerance=10):
    mask = ((run_df['year'] == year)
            & run_df['average_heartrate'].between(hr_target - tolerance, hr_target + tolerance)
            & run_df['pace_min_per_km'].notna()
            & np.isfinite(run_df['pace_min_per_km']))
    if not mask.any():
        return None
    return run_df.loc[mask, 'pace_min_per_km'].mean()

_pace_this = _avg_pace_at_hr(latest_year)
_pace_last = _avg_pace_at_hr(prev_year)
if _pace_this is not None and _pace_last is not None:
    _diff_seconds = round((_pace_last - _pace_this) * 60)
    if _diff_seconds == 0:
        _insights.append(("🏃", f"Your pace at ~150 bpm is **unchanged** vs {prev_year}."))
    else:
        _direction = "faster" if _diff_seconds > 0 else "slower"
        _insights.append(
            ("🏃", f"At ~150 bpm you're **{abs(_diff_seconds)}s/km {_direction}** than {prev_year}.")
        )
elif _pace_this is not None:
    _insights.append(
        ("🏃", f"Average pace at ~150 bpm in {latest_year}: **{_format_pace(_pace_this)} /km**.")
    )

# 4) Vertical metres climbed this year (with Everest reference)
_year_vert_m = float(df[df['year'] == latest_year]['total_elevation_gain'].sum())
if _year_vert_m > 0:
    _everest_pct = _year_vert_m / 8849 * 100
    _insights.append(
        ("⛰️",
         f"You've climbed **{int(_year_vert_m):,} m**".replace(',', ' ') +
         f" in {latest_year} — **{_everest_pct:.0f}%** of Everest's height.")
    )

def _insight_card(icon, text):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(icon, style={"fontSize": "1.4rem", "marginBottom": "4px"}),
                dcc.Markdown(text, className="m-0",
                             style={"color": dark_text_color, "fontSize": "0.9rem", "lineHeight": 1.45}),
            ], className="py-3 text-center"),
            style={"backgroundColor": dark_paper_color,
                   "border": f"1px solid {dark_grid_color}",
                   "borderRadius": "10px"},
            className="h-100"
        ),
        md=3, className="mb-2"
    )

layout_insights = (
    dbc.Row([_insight_card(i, t) for (i, t) in _insights], className="g-3 mb-2")
    if _insights else html.Div()
)

#######################
# YEAR-OVER-YEAR TRAJECTORY
#######################

def _yoy_cumulative(year):
    yr = df[df['year'] == year]
    if yr.empty:
        return None
    by_day = yr.groupby(yr['start_date_local'].dt.dayofyear)['duration_hr'].sum()
    return by_day.reindex(range(1, 367), fill_value=0).cumsum()

cum_this_year = _yoy_cumulative(latest_year)
cum_last_year = _yoy_cumulative(prev_year)
day_now = today.timetuple().tm_yday

yoy_fig = go.Figure()

# Older history first so it sits behind the prominent lines
older_years = sorted([int(y) for y in df['year'].unique() if y < prev_year])
for year in older_years:
    cum = _yoy_cumulative(year)
    if cum is None:
        continue
    yoy_fig.add_trace(go.Scatter(
        x=list(range(1, 367)), y=cum.round(2).values,
        name=str(year),
        line=dict(color="#4B5563", width=1.2, dash="dot"),
        opacity=0.55,
        hovertemplate=f"<b>{year}</b>: %{{y:.1f}}h<extra></extra>"
    ))

# Previous year — secondary, dotted
if cum_last_year is not None:
    yoy_fig.add_trace(go.Scatter(
        x=list(range(1, 367)), y=cum_last_year.round(2).values,
        name=str(prev_year),
        line=dict(color=muted_text_color, width=2, dash="dot"),
        hovertemplate=f"<b>{prev_year}</b>: %{{y:.1f}}h<extra></extra>"
    ))

# Current year — primary, solid, capped at today
if cum_this_year is not None:
    _y = cum_this_year.round(2).copy()
    _y.iloc[day_now:] = np.nan  # don't extrapolate beyond today
    yoy_fig.add_trace(go.Scatter(
        x=list(range(1, 367)), y=_y.values,
        name=str(latest_year),
        line=dict(color=SERIES["optimal"], width=2.5),
        hovertemplate=f"<b>{latest_year}</b>: %{{y:.1f}}h<extra></extra>"
    ))

# Today indicator + delta annotation
if cum_this_year is not None and cum_last_year is not None and day_now >= 1:
    yoy_fig.add_vline(x=day_now, line_dash="dash", line_color=SERIES["form_zero"],
                      line_width=1)
    same_day_delta = cum_this_year.iloc[day_now - 1] - cum_last_year.iloc[day_now - 1]
    delta_label = f"+{same_day_delta:.1f}h ahead" if same_day_delta >= 0 else f"{abs(same_day_delta):.1f}h behind"
    delta_color = SERIES["fresh"] if same_day_delta >= 0 else SERIES["overreach"]
    yoy_fig.add_annotation(
        x=day_now, y=cum_this_year.iloc[day_now - 1],
        text=f"  {delta_label}  ",
        font=dict(color=delta_color, size=11, family=chart_font_family, weight=600),
        showarrow=False,
        bgcolor=dark_paper_color, bordercolor=delta_color, borderwidth=1, borderpad=4,
        xanchor='left', yanchor='middle',
    )

month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

yoy_fig.update_layout(
    template=dark_template,
    title_text=f"📅 Year-over-Year · Cumulative Hours · {prev_year} vs {latest_year}",
    hovermode="x unified",
    height=440,
    xaxis=dict(tickmode='array', tickvals=month_starts, ticktext=month_labels,
               range=[0, 366], showgrid=False),
    yaxis=dict(title_text="Hours", rangemode="tozero", tickformat=".0f"),
)

# Fitness chart (filter-aware) side-by-side with YoY trajectory (full history).
# Filter dropdown sits above this row in app.layout so the relationship between
# the filter and the fitness chart on the left half stays visually clear.
layout_fitness_yoy_row = dbc.Row([
    dbc.Col(dcc.Graph(id='fitness-graph', figure=build_fitness_fig(INITIAL_FILTER_START)), md=6),
    dbc.Col(dcc.Graph(figure=yoy_fig), md=6),
])

#######################
# MAP — where you train
#######################

all_lats = []
all_lngs = []
n_routes = 0
for _poly in polylines_for_map.dropna():
    if not isinstance(_poly, str) or not _poly:
        continue
    _coords = decode_polyline(_poly)
    if not _coords:
        continue
    n_routes += 1
    for _lat, _lng in _coords:
        all_lats.append(_lat)
        all_lngs.append(_lng)
    # None separator so segments don't connect across activities
    all_lats.append(None)
    all_lngs.append(None)

if n_routes:
    # Activity-weighted home detection. Step 1: take the median of all decoded
    # points — robust to a few far-flung trips. Step 2: keep only points
    # within ~50 km of that median (filters out travel). Step 3: average them
    # to get the home centroid. Zoom is sized to that home cluster's spread
    # so the initial view frames your everyday training, not the whole world.
    _valid_lats = np.array([v for v in all_lats if v is not None])
    _valid_lngs = np.array([v for v in all_lngs if v is not None])
    _rough_lat = float(np.median(_valid_lats))
    _rough_lng = float(np.median(_valid_lngs))

    HOME_RADIUS_DEG = 0.5  # ~55 km at the equator, enough for a metro area
    _deltas = np.sqrt((_valid_lats - _rough_lat)**2 + (_valid_lngs - _rough_lng)**2)
    _home_mask = _deltas < HOME_RADIUS_DEG
    if _home_mask.sum() >= 10:
        _home_lats = _valid_lats[_home_mask]
        _home_lngs = _valid_lngs[_home_mask]
        _home_lat = float(_home_lats.mean())
        _home_lng = float(_home_lngs.mean())
        _spread = max(
            float(_home_lats.max() - _home_lats.min()),
            float(_home_lngs.max() - _home_lngs.min()),
            0.05,
        )
        # Zoom so the home cluster fills most of the viewport. Bumped +1.5
        # over the natural log2 fit to land near city-level (10–11) for
        # typical metros instead of regional view.
        import math as _math
        _zoom = max(9.5, min(13.0, _math.log2(180.0 / _spread) + 1.5))
    else:
        _home_lat, _home_lng = _rough_lat, _rough_lng
        _zoom = 10.5
    _center = dict(lat=_home_lat, lon=_home_lng)

    # Region clustering — group representative points (every 50th decoded
    # polyline point) into clusters within 1° (~110 km). Each cluster becomes
    # a faint circle marker visible at world/continent zoom and hidden once
    # the user zooms in past city-level so it doesn't obscure the routes.
    # Centre each circle on the cluster's *medoid* (the actual sampled point
    # closest to the running centroid) so every circle sits on a real route
    # point — never in dead space between distinct metros.
    SAMPLE_STEP = 50
    REGION_THRESHOLD_DEG = 1.0
    _clusters_raw = []
    for _lat, _lng in zip(_valid_lats[::SAMPLE_STEP], _valid_lngs[::SAMPLE_STEP]):
        _merged = False
        for _c in _clusters_raw:
            _d = ((_c['lat_c'] - _lat) ** 2 + (_c['lng_c'] - _lng) ** 2) ** 0.5
            if _d < REGION_THRESHOLD_DEG:
                _c['points'].append((float(_lat), float(_lng)))
                _n = len(_c['points'])
                _c['lat_c'] = sum(p[0] for p in _c['points']) / _n
                _c['lng_c'] = sum(p[1] for p in _c['points']) / _n
                _merged = True
                break
        if not _merged:
            _clusters_raw.append({
                'points': [(float(_lat), float(_lng))],
                'lat_c': float(_lat),
                'lng_c': float(_lng),
            })

    _region_clusters = []
    for _c in _clusters_raw:
        if len(_c['points']) < 2:
            continue
        _pts = np.array(_c['points'])
        _cx = float(_pts[:, 0].mean())
        _cy = float(_pts[:, 1].mean())
        _dists = np.sqrt((_pts[:, 0] - _cx) ** 2 + (_pts[:, 1] - _cy) ** 2)
        _idx = int(_dists.argmin())
        _region_clusters.append({
            'lat': float(_pts[_idx, 0]),
            'lng': float(_pts[_idx, 1]),
            'count': len(_pts),
        })

    # Two-pass rendering for log-like saturation:
    #   Pass 1 (base): warm amber at moderate alpha — guarantees every route
    #                  is clearly visible even on solo trips.
    #   Pass 2 (heat): deep red at low alpha — additively darkens areas where
    #                  many routes overlap, producing the heatmap effect
    #                  without crushing single-line visibility.
    BASE_COLOR = "rgba(255, 165, 60, 0.25)"
    HEAT_COLOR = "rgba(252, 30, 0, 0.12)"

    _ScatterCls = getattr(go, 'Scattermap', None) or go.Scattermapbox
    _is_new_map_api = _ScatterCls is getattr(go, 'Scattermap', None)

    map_fig = go.Figure()
    map_fig.add_trace(_ScatterCls(
        lat=all_lats, lon=all_lngs,
        mode='lines',
        line=dict(color=BASE_COLOR, width=1.4),
        hoverinfo='skip',
        showlegend=False,
        name='base',
    ))
    map_fig.add_trace(_ScatterCls(
        lat=all_lats, lon=all_lngs,
        mode='lines',
        line=dict(color=HEAT_COLOR, width=1.4),
        hoverinfo='skip',
        showlegend=False,
        name='heat',
    ))

    _coverage = (
        f"<span style='font-size: 0.65em; color:{muted_text_color}; font-weight:400'>"
        f"{n_routes} routes plotted · indoor sessions don't record GPS"
        "</span>"
    )
    # Region circles via a static GeoJSON layer with maxzoom so they fade
    # out automatically when the user zooms in past world/continent view.
    _regions_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [_c['lng'], _c['lat']]},
                "properties": {},
            }
            for _c in _region_clusters
        ],
    }
    _region_layer = {
        "sourcetype": "geojson",
        "source": _regions_geojson,
        "type": "circle",
        "color": "rgba(255, 200, 120, 0.45)",
        "circle": {"radius": 24},
        "maxzoom": 6.5,
    }
    _map_layout_args = dict(
        style='carto-darkmatter',
        zoom=_zoom,
        center=_center,
        layers=[_region_layer] if _region_clusters else [],
    )
    map_fig.update_layout(
        title_text=f"🌍 Where Do You Train?<br>{_coverage}",
        paper_bgcolor=dark_paper_color,
        plot_bgcolor=dark_paper_color,
        font=chart_font,
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
        **({'map': _map_layout_args} if _is_new_map_api else {'mapbox': _map_layout_args}),
    )
    layout_map = dbc.Row([dbc.Col(dcc.Graph(figure=map_fig), md=12)])
else:
    layout_map = html.Div()

# Final app layout
# Section A — Current state (daily glance): KPIs · Insights · Form+Load.
# Section B — Recent activity: Year heatmap · Bonus tables.
# Section C — Trends: Filter · Fitness+YoY trajectory · Heatmap+Cumulative.
# Section D — Reference / all-time: Run scatters · Map · Pies · PRs.
app.layout = dbc.Container([
    html.H1("Strava Training Dashboard", className="text-center my-4 dashboard-title"),
    # Section A
    layout_kpi,
    layout_insights,
    html.Hr(),
    fitness_explainer,
    layout_form_recent,
    html.Hr(),
    # Section B
    layout_year_heatmap,
    html.Hr(),
    layout_bonus,
    layout_yoy,
    html.Hr(),
    # Section C
    filter_dropdown,
    layout_fitness_yoy_row,
    html.Hr(),
    layout_heatmap_cumulative,
    html.Hr(),
    # Section D
    layout_scatter,
    html.Hr(),
    layout_map,
    html.Hr(),
    layout_pie,
    html.Hr(),
    layout_prs,
], fluid=True, style={"backgroundColor": dark_bg_color})

@app.callback(
    Output('fitness-graph', 'figure'),
    Output('heatmap-graph', 'figure'),
    Output('cumulative-time-graph', 'figure'),
    Input('global-filter', 'value'),
)
def _update_filtered_charts(filter_value):
    start = _filter_start_from_value(filter_value)
    return (
        build_fitness_fig(start),
        build_heatmap_fig(start),
        build_cumulative_time_fig(start),
    )

@app.callback(
    Output('year-heatmap-graph', 'figure'),
    Input('year-heatmap-view', 'value'),
)
def _toggle_year_heatmap(view):
    return (year_heatmap_activities_fig if view == 'activities'
            else year_heatmap_load_fig)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
