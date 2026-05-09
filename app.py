import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
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

df = df[df["start_date_local"] >= "2023-01-01"]

# Walks weren't tracked consistently before 2024 — exclude them so they don't
# inflate active-day / cumulative metrics with sparse early-period data.
df = df[~((df['type'] == 'Walk') & (df['start_date_local'] < '2024-01-01'))]

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

today = datetime.now().date()
latest_year = df['year'].max()
prev_year = latest_year - 1

#######################
# STYLING & THEMING
#######################

colors = px.colors.qualitative.Set2
activity_types = df['type'].unique()
color_map = {
    'Run': '#36a2eb',
    'Ride': '#ff9f40',
    'Racquet Sports': '#b967ff',
    'Cardio': '#4bc0c0',
    'Weight Training': '#ff6384',
    'Hike': '#a8a8a8',
    'Walk': '#97e084',
}

# Fill any activity types that aren't pre-mapped, cycling through Set2
missing_types = [t for t in activity_types if t not in color_map]
for i, t in enumerate(missing_types):
    color_map[t] = colors[i % len(colors)]

dark_bg_color = "#121212"
dark_paper_color = "#1E1E1E"
dark_text_color = "#FFFFFF"
dark_grid_color = "#333333"
muted_text_color = "#888888"

# Single font stack used for HTML (via assets/style.css) and Plotly figures
chart_font_family = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
chart_font = dict(color=dark_text_color, family=chart_font_family)

dark_template = dict(
    layout=dict(
        paper_bgcolor=dark_paper_color,
        plot_bgcolor=dark_bg_color,
        font=chart_font,
        xaxis=dict(gridcolor=dark_grid_color, zerolinecolor=dark_grid_color),
        yaxis=dict(gridcolor=dark_grid_color, zerolinecolor=dark_grid_color),
        legend=dict(font=chart_font)
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

def _trimp(row):
    hr = row.get('average_heartrate')
    if pd.isnull(hr):
        return row['duration_hr'] * 100
    hrr = (hr - HR_REST) / (HR_MAX - HR_REST)
    hrr = max(0.0, min(1.0, hrr))
    duration_min = row['duration_hr'] * 60
    return duration_min * hrr * 0.64 * float(np.exp(1.92 * hrr))

df['training_load'] = df.apply(_trimp, axis=1)

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
projected_hours = ytd_hours / year_progress if year_progress > 0 else 0

on_track = projected_hours > prev_year_hours
trend_emoji = "🔼" if on_track else "🔽"
progress_percent = f"{year_progress:.1%}"

#######################
# KPI METRICS
#######################

latest_scores = daily_scores.iloc[-1]
current_fitness = latest_scores['fitness']
current_fatigue = latest_scores['fatigue']
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

#######################
# MONTHLY ACTIVITY
#######################

monthly_activity = df.groupby([pd.Grouper(key='start_date_local', freq='ME'), 'type'])['duration_hr'].sum().reset_index()
monthly_activity['month'] = monthly_activity['start_date_local'].dt.to_period('M').dt.to_timestamp()

#######################
# CUMULATIVE STATS
#######################

cumulative_df = df.copy()
cumulative_df = cumulative_df.sort_values("start_date_local")
cumulative_df["activity_count"] = 1

all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
date_df = pd.DataFrame({"date": all_dates.date})

count_df = cumulative_df.groupby(["date", "type"])["activity_count"].sum().reset_index()
count_pivot = count_df.pivot(index='date', columns='type', values='activity_count').fillna(0)
count_pivot = pd.merge(date_df, count_pivot.reset_index(), on='date', how='left').fillna(0)
count_pivot = count_pivot.sort_values('date')
for col in count_pivot.columns:
    if col != 'date':
        count_pivot[col] = count_pivot[col].cumsum()

time_df = cumulative_df.groupby(["date", "type"])["duration_hr"].sum().reset_index()
time_pivot = time_df.pivot(index='date', columns='type', values='duration_hr').fillna(0)
time_pivot = pd.merge(date_df, time_pivot.reset_index(), on='date', how='left').fillna(0)
time_pivot = time_pivot.sort_values('date')
for col in time_pivot.columns:
    if col != 'date':
        time_pivot[col] = time_pivot[col].cumsum()

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
# ROLLING VOLUME (hours per week)
#######################

daily_hours_series = (
    df.groupby('date')['duration_hr'].sum()
    .reindex(date_range.date, fill_value=0)
)
rolling_7d = daily_hours_series.rolling(7, min_periods=1).sum()
rolling_28d = daily_hours_series.rolling(28, min_periods=1).sum() / 4
rolling_365d = daily_hours_series.rolling(365, min_periods=1).sum() / (365 / 7)

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

_hardest_pool = df[df['average_heartrate'].notna()]
pr_hardest = _hardest_pool.loc[_hardest_pool['average_heartrate'].idxmax()] if not _hardest_pool.empty else None

#######################
# DASHBOARD LAYOUTS
#######################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Strava Training Dashboard"

# 0. KPI Cards

def kpi_card(title, value, subtitle="", value_color=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-uppercase small", style={"color": muted_text_color}),
            html.Div(value, className="fw-bold",
                     style={"fontSize": "1.8rem", "color": value_color or dark_text_color}),
            html.Div(subtitle, className="small", style={"color": muted_text_color}),
        ]),
        className="text-center h-100",
        style={"backgroundColor": dark_paper_color, "border": f"1px solid {dark_grid_color}"}
    )

layout_kpi = dbc.Row([
    dbc.Col(kpi_card("Fitness (CTL)", f"{current_fitness:.0f}", "42-day load avg"), md=2),
    dbc.Col(kpi_card("Form (TSB)", f"{current_form:+.0f}", form_label, value_color=form_color), md=3),
    dbc.Col(kpi_card("This Week", f"{this_week_hours:.1f}h",
                     f"{week_sign}{week_delta:.1f}h vs 8-wk avg ({trailing_8wk_avg:.1f}h)",
                     value_color=week_color), md=3),
    dbc.Col(kpi_card("Current Streak", f"{current_streak} days",
                     f"Last activity: {last_activity_date}"), md=2),
    dbc.Col(kpi_card("Days Off", f"{days_since_last}",
                     "since last activity"), md=2),
], className="g-3 mb-2")

# 1. Activity Type Distribution Charts
layout_pie = dbc.Row([
    dbc.Col(dcc.Graph(
        figure=px.pie(
            pie_df(),
            names="type",
            values="duration_hr",
            title="🌟 All-Time Activity (Since 2023) 🌟",
            color="type",
            color_discrete_map=color_map,
            hole=0.4
        ).update_traces(
            texttemplate="%{value:.1f}h",
            hovertemplate="%{label}: %{value:.1f}h<extra></extra>"
        ).update_layout(
            template=dark_template,
            showlegend=False
        )
    ), md=4),
    dbc.Col([
        dcc.Graph(
            figure=px.pie(
                pie_df("previous"),
                names="type",
                values="duration_hr",
                title=f"🔙 {prev_year} Activity ({prev_year_hours:.1f}h) 🔙",
                color="type",
                color_discrete_map=color_map,
                hole=0.4
            ).update_traces(
                texttemplate="%{value:.1f}h",
                hovertemplate="%{label}: %{value:.1f}h<extra></extra>"
            ).update_layout(
                template=dark_template,
                showlegend=False
            )
        ),
    ], md=4),
    dbc.Col([
        dcc.Graph(
            figure=px.pie(
                pie_df("YTD"),
                names="type",
                values="duration_hr",
                title=f"🔥 {latest_year} YTD ({ytd_hours:.1f}h, {progress_percent} of year) 🔥",
                color="type",
                color_discrete_map=color_map,
                hole=0.4
            ).update_traces(
                texttemplate="%{value:.1f}h",
                hovertemplate="%{label}: %{value:.1f}h<extra></extra>"
            ).update_layout(
                template=dark_template,
                showlegend=False
            )
        ),
        html.Div([
            html.H5([
                f"✨ Projected {latest_year}: {projected_hours:.1f}h ",
                html.Span(f"{trend_emoji} {abs(projected_hours - prev_year_hours):.1f}h vs {prev_year}",
                         style={"color": "lightgreen" if on_track else "#ff6b6b"})
            ], className="text-center mt-2")
        ])
    ], md=4),
])

# 2. Fitness / Fatigue / Form — explainer + 30-day focused form view + full chart

fitness_explainer = dcc.Markdown(
    """
**Fitness (CTL)** — 42-day rolling training load. Your long-term base.
**Fatigue (ATL)** — 7-day rolling load. How loaded your body is right now.
**Form (TSB) = Fitness − Fatigue.** Positive = fresh, negative = tired.

Bands: above **+5** fresh / peaked &nbsp;·&nbsp; **+5 to −10** optimal training zone &nbsp;·&nbsp;
**−10 to −30** productive (building, accept the tired) &nbsp;·&nbsp; below **−30** overreaching.
    """,
    className="px-4 py-2 mb-2",
    style={"color": dark_text_color, "fontSize": "0.92rem", "lineHeight": "1.6"}
)

# Last-30-day Form chart with zone shading
form_recent_start = today - timedelta(days=30)
form_recent = daily_scores[daily_scores['date'] >= form_recent_start].copy()

zone_specs = [
    (5, 200, "Fresh", "#4ade80"),
    (-10, 5, "Optimal", "#60a5fa"),
    (-30, -10, "Productive", "#facc15"),
    (-200, -30, "Overreaching", "#f87171"),
]

form_recent_fig = go.Figure()

for y0, y1, _, hex_color in zone_specs:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    form_recent_fig.add_hrect(
        y0=y0, y1=y1, fillcolor=f"rgba({r},{g},{b},0.13)",
        line_width=0, layer="below"
    )

form_recent_fig.add_trace(go.Scatter(
    x=form_recent['date'], y=form_recent['form'],
    mode="lines",
    line=dict(color="#ffffff", width=3),
    hovertemplate="%{x|%b %d}<br>Form: %{y:+.1f}<extra></extra>",
    name="Form (TSB)",
    showlegend=False
))

# Highlight today's value
form_recent_fig.add_trace(go.Scatter(
    x=[form_recent['date'].iloc[-1]],
    y=[form_recent['form'].iloc[-1]],
    mode="markers",
    marker=dict(color=form_color, size=14, line=dict(color="white", width=2)),
    hovertemplate=f"Today: {current_form:+.1f}<br>{form_label}<extra></extra>",
    showlegend=False
))

form_recent_fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dash"))

# Zone labels on the right edge
form_y_min = min(form_recent['form'].min(), -35) - 5
form_y_max = max(form_recent['form'].max(), 12) + 5
zone_label_positions = [
    (max(form_y_max - 4, 8), "Fresh", "#4ade80"),
    (-2.5, "Optimal", "#60a5fa"),
    (-20, "Productive", "#facc15"),
    (min(form_y_min + 4, -36), "Overreaching", "#f87171"),
]
for y, label, color in zone_label_positions:
    if form_y_min <= y <= form_y_max:
        form_recent_fig.add_annotation(
            xref="paper", x=1.0, y=y,
            xanchor="left", yanchor="middle",
            text=label, showarrow=False,
            font=dict(color=color, size=10, family=chart_font_family),
            xshift=8
        )

form_recent_fig.update_layout(
    title_text=f"🎯 Form — Last 30 Days  &nbsp;<span style='color:{form_color}'>{current_form:+.0f} · {form_label}</span>",
    paper_bgcolor=dark_paper_color,
    plot_bgcolor=dark_bg_color,
    font=chart_font,
    xaxis=dict(showgrid=False, title_text=None),
    yaxis=dict(
        title_text="Form (TSB)",
        gridcolor=dark_grid_color, zerolinecolor="#666",
        range=[form_y_min, form_y_max]
    ),
    margin=dict(l=10, r=90, t=60, b=10),
    height=320
)

layout_form_recent = dbc.Row([dbc.Col(dcc.Graph(figure=form_recent_fig), md=12)])

# 2b. Full Fitness Chart (CTL / ATL / TSB + monthly bars)
fitness_fig = make_subplots(specs=[[{"secondary_y": True}]])

# Stacked monthly bars (no legend — colors echo the pies)
for activity_type in df['type'].unique():
    activity_data = monthly_activity[monthly_activity['type'] == activity_type]
    fitness_fig.add_trace(
        go.Bar(
            x=activity_data['month'],
            y=activity_data['duration_hr'],
            name=activity_type,
            marker_color=color_map.get(activity_type, '#333333'),
            hovertemplate=f"{activity_type}: %{{y:.1f}}h<br>%{{x|%b %Y}}<extra></extra>",
            showlegend=False
        ),
        secondary_y=False
    )

# Fitness (CTL) — primary line
fitness_fig.add_trace(
    go.Scatter(
        x=daily_scores['date'], y=daily_scores['fitness'],
        name="Fitness (CTL, 42d)",
        line=dict(color="#ffffff", width=2.5),
        hovertemplate="Fitness: %{y:.0f}<extra></extra>"
    ),
    secondary_y=True
)
# Fatigue (ATL)
fitness_fig.add_trace(
    go.Scatter(
        x=daily_scores['date'], y=daily_scores['fatigue'],
        name="Fatigue (ATL, 7d)",
        line=dict(color="#ff9f40", width=1.5, dash="dot"),
        hovertemplate="Fatigue: %{y:.0f}<extra></extra>"
    ),
    secondary_y=True
)
# Form (TSB) — can go negative; zero line below highlights crossings
fitness_fig.add_trace(
    go.Scatter(
        x=daily_scores['date'], y=daily_scores['form'],
        name="Form (TSB)",
        line=dict(color="#4ade80", width=1.5),
        hovertemplate="Form: %{y:+.0f}<extra></extra>"
    ),
    secondary_y=True
)
fitness_fig.add_hline(y=0, line=dict(color="#666666", width=1, dash="dash"), secondary_y=True)

fitness_fig.update_layout(
    title_text="💪 Fitness, Fatigue & Form 📈",
    barmode='stack',
    hovermode="x unified",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=chart_font),
    paper_bgcolor=dark_paper_color,
    plot_bgcolor=dark_bg_color,
    font=chart_font,
    xaxis=dict(showgrid=False, zerolinecolor=dark_bg_color, title_text=None),
    margin=dict(l=10, r=10, t=60, b=10)
)
fitness_fig.update_yaxes(
    title_text="Hours", secondary_y=False, showgrid=True,
    gridwidth=1, gridcolor=dark_grid_color, minor_showgrid=False, rangemode="tozero"
)
fitness_fig.update_yaxes(
    title_text="Score", secondary_y=True, title_font=dict(color="#ffffff"),
    showgrid=False, minor_showgrid=False
)

layout_fitness = dbc.Row([dbc.Col(dcc.Graph(figure=fitness_fig), md=12)])

# 3. Rolling Volume Chart
rolling_fig = go.Figure()
rolling_fig.add_trace(go.Scatter(
    x=list(daily_hours_series.index), y=rolling_7d.values,
    name="Week (7d)", line=dict(color="#36a2eb", width=2),
    hovertemplate="7d: %{y:.1f} h/wk<extra></extra>"
))
rolling_fig.add_trace(go.Scatter(
    x=list(daily_hours_series.index), y=rolling_28d.values,
    name="Month (28d avg)", line=dict(color="#ff9f40", width=2),
    hovertemplate="28d avg: %{y:.1f} h/wk<extra></extra>"
))
rolling_fig.add_trace(go.Scatter(
    x=list(daily_hours_series.index), y=rolling_365d.values,
    name="Year (365d avg)", line=dict(color="#b967ff", width=2),
    hovertemplate="365d avg: %{y:.1f} h/wk<extra></extra>"
))
rolling_fig.update_layout(
    title_text="⏳ Rolling Training Volume (hours per week) ⏳",
    hovermode="x unified",
    paper_bgcolor=dark_paper_color,
    plot_bgcolor=dark_bg_color,
    font=chart_font,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=chart_font),
    xaxis=dict(showgrid=False, title_text=None),
    yaxis=dict(showgrid=True, gridcolor=dark_grid_color, title_text="Hours / week",
               rangemode="tozero"),
    margin=dict(l=10, r=10, t=60, b=10)
)
layout_rolling = dbc.Row([dbc.Col(dcc.Graph(figure=rolling_fig), md=12)])

# 4. Day-of-Week × Hour Heatmap (replaces weekday bar + hourly bar)
heatmap_fig = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[f"{h:02d}:00" for h in range(24)],
    y=weekday_order,
    colorscale="Plasma",
    hovertemplate="%{y} %{x}<br>%{z:.1f} hours<extra></extra>",
    colorbar=dict(title="Hours")
))
heatmap_fig.update_layout(
    title_text="🔥 When Do You Train? (Day × Hour) 🔥",
    paper_bgcolor=dark_paper_color,
    plot_bgcolor=dark_bg_color,
    font=chart_font,
    xaxis=dict(showgrid=False, title_text=None, tickfont=dict(size=10)),
    yaxis=dict(showgrid=False, title_text=None, autorange="reversed"),  # Mon at top
    margin=dict(l=10, r=10, t=60, b=10),
    height=360
)
layout_heatmap = dbc.Row([dbc.Col(dcc.Graph(figure=heatmap_fig), md=12)])

# 5. Cumulative Stats
layout_cumulative = dbc.Row([
    dbc.Col(dcc.Graph(
        figure=px.area(
            count_pivot,
            x="date", y=count_pivot.columns[1:],
            title="🏆 What Activities Are You Doing Most? 🏆",
            color_discrete_map=color_map,
            labels={"value": "Count", "date": "", "variable": "Activity Type"}
        ).update_traces(
            hovertemplate="%{y:.1f} activities - %{fullData.name}<br>%{x|%b %Y}<extra></extra>"
        ).update_layout(
            template=dark_template,
            showlegend=False,
            xaxis=dict(showgrid=False, title_text=None),
            yaxis=dict(showgrid=True, title_text=None, gridwidth=1,
                       gridcolor=dark_grid_color, minor_showgrid=False)
        )
    ), md=6),
    dbc.Col(dcc.Graph(
        figure=px.area(
            time_pivot,
            x="date", y=time_pivot.columns[1:],
            title="⏱️ Training Hours Accumulation ⏱️",
            color_discrete_map=color_map,
            labels={"value": "Hours", "date": "", "variable": "Activity Type"}
        ).update_traces(
            hovertemplate="%{y:.1f} hours - %{fullData.name}<br>%{x|%b %Y}<extra></extra>"
        ).update_layout(
            template=dark_template,
            showlegend=False,
            xaxis=dict(showgrid=False, title_text=None),
            yaxis=dict(showgrid=True, title_text=None, gridwidth=1,
                       gridcolor=dark_grid_color, minor_showgrid=False)
        )
    ), md=6)
])

# 6. Run Performance — pace axis formatted as MM:SS, year as discrete colors
pace_ticks = list(np.arange(3.0, 9.5, 0.5))
pace_tick_labels = [_format_pace(p) for p in pace_ticks]

def _safe_pace_str(p):
    if pd.isna(p) or not np.isfinite(p):
        return "—"
    return _format_pace(p)

layout_scatter = dbc.Row([
    dbc.Col(dcc.Graph(
        figure=px.scatter(
            run_df,
            x="date", y="pace_min_per_km",
            size="distance", color="average_heartrate",
            title="🏃‍♂️ Run Pace Timeline 🏃‍♀️",
            labels={"pace_min_per_km": "Pace", "date": "",
                    "average_heartrate": "Avg HR", "distance": "Distance (m)"},
            color_continuous_scale="RdYlBu_r",
            hover_data=["name", "distance", "average_heartrate", "moving_time"]
        ).update_traces(
            marker=dict(opacity=0.7),
            selector=dict(mode='markers'),
            hovertemplate=
                "Pace: %{customdata[0]} /km<br>"
                "Distance: %{customdata[1]:.1f} km<br>"
                "Avg HR: %{customdata[2]}<br>"
                "Time: %{customdata[3]}<extra></extra>",
            customdata=run_df.apply(lambda x: [
                _safe_pace_str(x['pace_min_per_km']),
                (x['distance'] or 0) / 1000,
                f"{int(x['average_heartrate'])} bpm" if pd.notna(x['average_heartrate']) else "—",
                f"{int(x['moving_time']//60):02d}:{int(x['moving_time']%60):02d}"
            ], axis=1).tolist()
        ).update_layout(
            template=dark_template,
            xaxis=dict(showgrid=False, title_text=None),
            yaxis=dict(
                showgrid=True, title_text="Pace (min/km)",
                gridwidth=1, gridcolor=dark_grid_color, minor_showgrid=False,
                autorange="reversed",
                tickvals=pace_ticks, ticktext=pace_tick_labels
            ),
            coloraxis_colorbar=dict(tickformat="d")
        )
    ), md=6),

    dbc.Col(dcc.Graph(
        figure=px.scatter(
            run_df,
            x="average_heartrate", y="pace_min_per_km",
            color="year_str", size="distance",
            color_discrete_map=year_color_map,
            category_orders={"year_str": [str(y) for y in sorted_years]},
            title="❤️ Run Efficiency ⚡",
            labels={"pace_min_per_km": "Pace", "average_heartrate": "Avg HR (bpm)",
                    "year_str": "Year", "distance": "Distance (m)"},
            hover_data=["name", "date", "distance", "moving_time"]
        ).update_traces(
            marker=dict(opacity=0.75, line=dict(width=0.5, color='white')),
            selector=dict(mode='markers'),
            hovertemplate=
                "%{customdata[0]} /km @ %{x:.0f} bpm<br>"
                "%{customdata[1]}<br>"
                "Distance: %{customdata[2]:.1f} km<br>"
                "Time: %{customdata[3]}<extra></extra>",
            customdata=run_df.apply(lambda x: [
                _safe_pace_str(x['pace_min_per_km']),
                x['date'],
                (x['distance'] or 0) / 1000,
                f"{int(x['moving_time']//60):02d}:{int(x['moving_time']%60):02d}"
            ], axis=1).tolist()
        ).update_layout(
            template=dark_template,
            xaxis=dict(showgrid=True, title_text="Avg HR (bpm)",
                       gridwidth=1, gridcolor=dark_grid_color, title_font=dict(size=12)),
            yaxis=dict(
                showgrid=True, title_text="Pace (min/km)",
                gridwidth=1, gridcolor=dark_grid_color, minor_showgrid=False,
                autorange="reversed", title_font=dict(size=12),
                tickvals=pace_ticks, ticktext=pace_tick_labels
            ),
            legend=dict(title="Year", orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1)
        )
    ), md=6)
])

# 7. Bubble calendar — last 8 weeks
most_recent_date = df['date'].max()
recent_date = max(most_recent_date, today)
days_since_monday = recent_date.weekday()
most_recent_monday = recent_date - timedelta(days=days_since_monday)
start_date = most_recent_monday - timedelta(weeks=8)

recent_df = df[df['date'] >= start_date].copy()
recent_df['date_dt'] = pd.to_datetime(recent_df['date'])
recent_df['week_number'] = ((recent_df['date_dt'] - pd.to_datetime(start_date)).dt.days // 7)
recent_df['weekday'] = recent_df['start_date_local'].dt.weekday

daily_stats = recent_df.groupby(['date', 'week_number', 'weekday']).apply(
    lambda x: pd.Series({
        'duration_hr': x['duration_hr'].sum(),
        'avg_hr': np.average(
            x['average_heartrate'].fillna(0),
            weights=x['duration_hr'],
            axis=0
        ) if x['average_heartrate'].notna().any() else 0,
        'activity_types': ', '.join(x['type'].unique())
    })
).reset_index()

bubble_date_range = pd.date_range(start=start_date, end=recent_date, freq='D')
start_date_ts = pd.Timestamp(start_date)
calendar_df = pd.DataFrame({
    'date': bubble_date_range.date,
    'week_number': [(d - start_date_ts).days // 7 for d in bubble_date_range],
    'weekday': [d.weekday() for d in bubble_date_range]
})
calendar_df = calendar_df.merge(daily_stats, on=['date', 'week_number', 'weekday'], how='left')
calendar_df['duration_hr'] = calendar_df['duration_hr'].fillna(0)
calendar_df['avg_hr'] = calendar_df['avg_hr'].fillna(0)
calendar_df['activity_types'] = calendar_df['activity_types'].fillna('')
calendar_df['bubble_size'] = calendar_df['duration_hr'] * 60

min_hr = max(50, calendar_df[calendar_df['avg_hr'] > 0]['avg_hr'].min()) if (calendar_df['avg_hr'] > 0).any() else 60
max_hr = min(200, calendar_df['avg_hr'].max()) if (calendar_df['avg_hr'] > 0).any() else 180
midpoint_hr = (min_hr + max_hr) / 2

bubble_fig = px.scatter(
    calendar_df,
    x='weekday', y='week_number',
    size='bubble_size', color='avg_hr',
    color_continuous_scale='RdYlBu_r',
    title='📅 Workout Calendar — Last 8 Weeks 🗓️',
    labels={'weekday': 'Day of Week', 'week_number': 'Week',
            'avg_hr': 'Avg HR (bpm)', 'duration_hr': 'Hours'},
    size_max=60,
    color_continuous_midpoint=midpoint_hr,
    range_color=[min_hr, max_hr]
)
bubble_fig.update_traces(
    hovertemplate="Date: %{customdata[0]}<br>Time: %{customdata[1]}<br>"
                  "Avg HR: %{customdata[2]}<br>Activities: %{customdata[3]}",
    customdata=calendar_df.apply(lambda x: [
        x['date'].strftime('%b %d, %Y'),
        f"{int(x['duration_hr'] * 60)}:{int((x['duration_hr'] * 60 % 1) * 60):02d}" if x['duration_hr'] > 0 else "0:00",
        f"{int(x['avg_hr'])}" if x['avg_hr'] > 0 else "N/A",
        x['activity_types'] if x['activity_types'] else "None"
    ], axis=1).tolist()
)
# Week labels: actual week-of dates rather than "Week 0"
week_starts = [start_date + timedelta(weeks=w) for w in range(9)]
week_tick_labels = [w.strftime("Wk of %b %d") for w in week_starts]

bubble_fig.update_layout(
    template=dark_template,
    xaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        showgrid=False, title_text=None, zeroline=False,
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(0, 9)),
        ticktext=week_tick_labels,
        showgrid=True, title_text=None,
        gridwidth=1, gridcolor=dark_grid_color, minor_showgrid=False,
        autorange='reversed'
    )
)
layout_bubble = dbc.Row([dbc.Col(dcc.Graph(figure=bubble_fig), md=12)])

# 8. Bonus stats tables
recent_activities = df.sort_values("start_date_local", ascending=False).head(5)
activity_emoji = {
    'Run': '🏃', 'Ride': '🚴', 'Racquet Sports': '🏓', 'Cardio': '💪',
    'Weight Training': '🏋️', 'Hike': '🥾', 'Workout': '🤸', 'Walk': '🚶'
}

max_distance_idx = recent_activities[recent_activities['distance'].notna()]['distance'].idxmax() \
    if not recent_activities[recent_activities['distance'].notna()].empty else None
max_hr_idx = recent_activities[recent_activities['average_heartrate'].notna()]['average_heartrate'].idxmax() \
    if not recent_activities[recent_activities['average_heartrate'].notna()].empty else None

recent_activities_df = pd.DataFrame({
    "Date": recent_activities["start_date_local"].dt.strftime("%b %d"),
    "Activity": recent_activities.apply(
        lambda row: f"{activity_emoji.get(row['type'], '🏋️')} {row['type']}",
        axis=1
    ),
    "Duration": recent_activities["duration_hr"].apply(
        lambda x: f"{int(x * 60)}:{int((x * 60 % 1) * 60):02d} min"
    ),
    "Details": recent_activities.apply(
        lambda row: f"{row['distance']/1000:.1f}km {'🔥' if row.name == max_distance_idx else ''}"
            if row['type'] in ['Run', 'Ride', 'Hike']
        else f"{row.get('average_heartrate', 0):.0f} bpm {'❤️' if row.name == max_hr_idx else ''}"
            if pd.notnull(row.get('average_heartrate'))
        else "",
        axis=1
    )
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

def pr_card(title, value, subtitle=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-uppercase small", style={"color": muted_text_color}),
            html.Div(value, className="fw-bold", style={"fontSize": "1.4rem"}),
            html.Div(subtitle, className="small", style={"color": muted_text_color}),
        ]),
        className="text-center h-100",
        style={"backgroundColor": dark_paper_color, "border": f"1px solid {dark_grid_color}"}
    )

def _fmt_longest(row):
    if row is None:
        return ("—", "")
    return (f"{row['distance']/1000:.1f} km", f"{row['date']}")

def _fmt_pace(row):
    if row is None:
        return ("—", "")
    return (f"{_format_pace(row['pace_min_per_km'])} /km",
            f"{row['distance']/1000:.1f} km in {_format_duration(row['moving_time'])} on {row['date']}")

prs = [
    ("🏃 Longest Run", *_fmt_longest(pr_longest_run)),
    ("🚴 Longest Ride", *_fmt_longest(pr_longest_ride)),
    ("🥾 Longest Hike", *_fmt_longest(pr_longest_hike)),
    ("⚡ Fastest 5K", *_fmt_pace(pr_fastest_5k)),
    ("⚡ Fastest 10K", *_fmt_pace(pr_fastest_10k)),
    ("⚡ Fastest Long Run", *_fmt_pace(pr_fastest_long)),
    ("📈 Biggest Week", f"{busiest_week_total}h", f"{busiest_week_start.date()}"),
    ("📆 Biggest Day", f"{longest_day_hours}h", f"{longest_day}"),
    ("❤️ Hardest Session",
     f"{pr_hardest['average_heartrate']:.0f} bpm avg" if pr_hardest is not None else "—",
     f"{pr_hardest['type']} on {pr_hardest['date']}" if pr_hardest is not None else ""),
]

layout_prs = html.Div([
    html.H4("🏆 Personal Records", className="text-center my-3"),
    dbc.Row(
        [dbc.Col(pr_card(t, v, s), md=4, className="mb-3") for (t, v, s) in prs],
        className="g-3"
    )
])

# Final app layout
app.layout = dbc.Container([
    html.H1("Strava Activity Dashboard 🏃‍♂️📊", className="text-center my-4"),
    layout_kpi,
    html.Hr(),
    layout_pie,
    html.Hr(),
    layout_bonus,
    layout_yoy,
    html.Hr(),
    fitness_explainer,
    layout_form_recent,
    layout_fitness,
    html.Hr(),
    layout_rolling,
    html.Hr(),
    layout_heatmap,
    html.Hr(),
    layout_cumulative,
    html.Hr(),
    layout_scatter,
    html.Hr(),
    layout_bubble,
    html.Hr(),
    layout_prs,
], fluid=True, style={"backgroundColor": dark_bg_color})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
