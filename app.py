import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pytz

#######################
# DATA PREPARATION
#######################

# Load data and preprocess
df = pd.read_csv("activities.csv", parse_dates=["start_date_local"])

# for all runs with elevation_gain 0, use start_date instead of start_date_local
df.loc[(df['type'] == 'Run') & (df['total_elevation_gain'] == 0), 'start_date_local'] = pd.to_datetime(df['start_date'])

#df = df[df["type"] != "Walk"]
df = df[df["start_date_local"] >= "2023-01-01"]

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
df["weekday"] = df["start_date_local"].dt.day_name()

latest_year = df['year'].max()
prev_year = latest_year - 1

#######################
# STYLING & THEMING
#######################

colors = px.colors.qualitative.Set2
activity_types = df['type'].unique()
# Create a color map for each activity type
color_map = {
    'Run': '#36a2eb',         # Brighter blue
    'Ride': '#ff9f40',        # Lighter orange
    'Racquet Sports': '#b967ff',  # Brighter purple
    'Cardio': '#4bc0c0',      # Teal green
    'Weight Training': '#ff6384', # Bright pink
    'Hike': '#a8a8a8',        # Lighter gray
    'Walk': '#97e084',        # Light green
}

# Add any missing activity types dynamically from the data
for activity in activity_types:
    if activity not in color_map:
        # Assign from plotly colors if not explicitly defined
        color_idx = len(color_map) % len(colors)
        color_map[activity] = colors[color_idx]

# Define dark theme color settings
dark_bg_color = "#121212"
dark_paper_color = "#1E1E1E"
dark_text_color = "#FFFFFF"
dark_grid_color = "#333333"

# Create a template for dark mode plots
dark_template = dict(
    layout=dict(
        paper_bgcolor=dark_paper_color,
        plot_bgcolor=dark_bg_color,
        font=dict(color=dark_text_color),
        xaxis=dict(gridcolor=dark_grid_color, zerolinecolor=dark_grid_color),
        yaxis=dict(gridcolor=dark_grid_color, zerolinecolor=dark_grid_color),
        legend=dict(font=dict(color=dark_text_color))
    )
)

#######################
# METRICS CALCULATION
#######################

def pie_df(year=None):
    if year == "YTD":
        return df[df["year"] == latest_year].groupby("type")["duration_hr"].sum().reset_index()
    elif year == "previous":
        return df[df["year"] == prev_year].groupby("type")["duration_hr"].sum().reset_index()
    else:
        return df.groupby("type")["duration_hr"].sum().reset_index()

# Calculate YTD metrics and projections
current_ytd_df = df[df["year"] == latest_year]
prev_year_df = df[df["year"] == prev_year]

# Calculate day of year percentage
today = datetime.now()
day_of_year = today.timetuple().tm_yday
year_progress = day_of_year / (366 if today.year % 4 == 0 else 365)

# Calculate total hours
ytd_hours = current_ytd_df["duration_hr"].sum()
prev_year_hours = prev_year_df["duration_hr"].sum()
projected_hours = ytd_hours / year_progress if year_progress > 0 else 0

# On track status
on_track = projected_hours > prev_year_hours
trend_emoji = "🔼" if on_track else "🔽"
progress_percent = f"{year_progress:.1%}"

#######################
# FITNESS MODEL
#######################

# Calculate fitness, fatigue, and form based on daily effort (similar to Strava's fitness model)
# Compute Training Load for each activity (similar to Training Stress Score)
df['training_load'] = df.apply(
    lambda row: row['duration_hr'] * (row.get('average_heartrate', 0) / 150)**2 * 100 
    if pd.notnull(row.get('average_heartrate')) else row['duration_hr'] * 100,
    axis=1
)

# Create a complete date range for daily scores
date_range = pd.date_range(start=df['date'].min(), end=datetime.now().date(), freq='D')
date_df = pd.DataFrame({"date": date_range.date})

# Aggregate training load by day
daily_load = df.groupby('date')['training_load'].sum().reset_index()
daily_scores = pd.merge(date_df, daily_load, on='date', how='left').fillna(0)

# Calculate Fitness (CTL - Chronic Training Load) - 42 day exponentially weighted average
daily_scores['fitness'] = daily_scores['training_load'].ewm(span=42, min_periods=1).mean()

# Prepare monthly activity data
monthly_activity = df.groupby([pd.Grouper(key='start_date_local', freq='M'), 'type'])['duration_hr'].sum().reset_index()
monthly_activity['month'] = monthly_activity['start_date_local'].dt.to_period('M').dt.to_timestamp()

#######################
# CUMULATIVE STATS
#######################
# Create stacked cumulative plots
cumulative_df = df.copy()
cumulative_df = cumulative_df.sort_values("start_date_local")
cumulative_df["activity_count"] = 1

# Create a complete date range
all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
date_df = pd.DataFrame({"date": all_dates.date})

# Cumulative activity count - pivoted for stacking
count_df = cumulative_df.groupby(["date", "type"])["activity_count"].sum().reset_index()
count_pivot = count_df.pivot(index='date', columns='type', values='activity_count').fillna(0)
count_pivot = pd.merge(date_df, count_pivot.reset_index(), on='date', how='left').fillna(0)
count_pivot = count_pivot.sort_values('date')
# Calculate cumulative sums for each activity type
for col in count_pivot.columns:
    if col != 'date':
        count_pivot[col] = count_pivot[col].cumsum()

# Cumulative duration (hours) - pivoted for stacking
time_df = cumulative_df.groupby(["date", "type"])["duration_hr"].sum().reset_index()
time_pivot = time_df.pivot(index='date', columns='type', values='duration_hr').fillna(0)
time_pivot = pd.merge(date_df, time_pivot.reset_index(), on='date', how='left').fillna(0)
time_pivot = time_pivot.sort_values('date')
# Calculate cumulative sums for each activity type
for col in time_pivot.columns:
    if col != 'date':
        time_pivot[col] = time_pivot[col].cumsum()

#######################
# WEEKLY & HOURLY PATTERNS
#######################

# Weekly patterns
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_df = df.groupby(["weekday", "type"])["duration_hr"].sum().reset_index()
weekday_df["weekday"] = pd.Categorical(weekday_df["weekday"], categories=weekday_order, ordered=True)
weekday_df = weekday_df.sort_values("weekday")

# Hourly patterns
# Get hour of day from start_date_local
df['hour_of_day'] = df['start_date_local'].dt.hour

hourly_df = df.groupby(["hour_of_day", "type"])["duration_hr"].sum().reset_index()
hourly_df = hourly_df.sort_values("hour_of_day")

#######################
# RUN PERFORMANCE DATA
#######################

# Convert pace (min/km), filter for runs only with good HR and distance data
run_df = df[(df["type"] == "Run")]# & df["average_heartrate"].notna() & (df["distance"] > 0)]
run_df["pace_min_per_km"] = run_df["moving_time"] / 60 / (run_df["distance"] / 1000)

#######################
# BONUS STATS
#######################

# Longest session
longest = df.loc[df["duration_hr"].idxmax()]
longest_session = {
    "duration_hr": round(longest["duration_hr"], 2),
    "date": longest["date"],
    "type": longest["type"]
}

# Daily totals
daily_hours = df.groupby("date")["duration_hr"].sum()
longest_day = daily_hours.idxmax()
longest_day_hours = round(daily_hours.max(), 2)

# Weekly totals
weekly_hours = df.groupby("week_start")["duration_hr"].sum()
busiest_week_start = weekly_hours.idxmax()
busiest_week_total = round(weekly_hours.max(), 2)
busiest_week_end = busiest_week_start + timedelta(days=6)

# Weekly averages
last_year = df[df["year"] == prev_year]
current_ytd = df[df["year"] == latest_year]

avg_last_year = round(last_year.groupby("week_start")["duration_hr"].sum().mean(), 2)
avg_ytd = round(current_ytd.groupby("week_start")["duration_hr"].sum().mean(), 2)

# Longest break
df_sorted = df.sort_values("start_date_local")
df_sorted["prev_date"] = df_sorted["start_date_local"].shift(1)
df_sorted["gap"] = (df_sorted["start_date_local"] - df_sorted["prev_date"]).dt.days
longest_break = df_sorted["gap"].max()
gap_row = df_sorted.loc[df_sorted["gap"].idxmax()]
break_start = gap_row["prev_date"].date()
break_end = gap_row["start_date_local"].date()

# Longest streak calculation
# Get unique dates with activities and sort them
unique_dates = pd.Series(df_sorted['date'].unique()).sort_values()
# Convert to datetime for proper comparison
unique_dates_pd = pd.to_datetime(unique_dates)
# Calculate gaps between consecutive days with activities
date_diffs = unique_dates_pd.diff().dt.days
# Start a new streak when there's a gap larger than 1 day
streak_groups = (date_diffs > 1).cumsum()
# Count days in each streak
streak_lengths = unique_dates.groupby(streak_groups).count()
# Find the longest streak
longest_streak_len = streak_lengths.max()
longest_streak_idx = streak_lengths.idxmax()
# Get the dates of the longest streak
streak_dates = unique_dates[streak_groups == longest_streak_idx]
streak_start = streak_dates.min()
streak_end = streak_dates.max()

# Percent of all days with activities, given overall, for last year, and YTD
total_days = (df['date'].max() - df['date'].min()).days + 1
active_days = df['date'].nunique()
active_days_last_year = last_year['date'].nunique()
active_days_ytd = current_ytd['date'].nunique()
active_days_percent = round((active_days / total_days) * 100, 1)
active_days_percent_last_year = round((active_days_last_year / 365) * 100, 1)
days_ytd = (datetime.now().date() - datetime(latest_year, 1, 1).date()).days + 1
active_days_percent_ytd = round((active_days_ytd / days_ytd) * 100, 1)

# Create separate DataFrames for individual stats and comparative stats
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
# DASHBOARD LAYOUTS
#######################

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Strava Training Dashboard"

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

# 2. Fitness Chart
fitness_fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add fitness line with improved visibility for dark mode
fitness_fig.add_trace(
    go.Scatter(
        x=daily_scores['date'], 
        y=daily_scores['fitness'],
        name="Fitness Score",
        line=dict(color="#ffffff", width=2),  # Brighter white and thicker
        hovertemplate="Fitness: %{y:.1f}<extra></extra>"
    ),
    secondary_y=True
)

# Add activity bars by type
for activity_type in df['type'].unique():
    activity_data = monthly_activity[monthly_activity['type'] == activity_type]
    fitness_fig.add_trace(
        go.Bar(
            x=activity_data['month'],
            y=activity_data['duration_hr'],
            name=activity_type,
            marker_color=color_map.get(activity_type, '#333333'),
            hovertemplate=f"{activity_type}: %{{y:.1f}} hours<br>%{{x|%b %Y}}<extra></extra>"
        ),
        secondary_y=False
    )

# Update layout with enhanced dark mode
fitness_fig.update_layout(
    title_text="💪 Fitness Trend & Monthly Training Hours 📈",
    barmode='stack',
    hovermode="x unified",
    showlegend=False,  # Remove legend
    paper_bgcolor=dark_paper_color,
    plot_bgcolor=dark_bg_color,
    font=dict(color=dark_text_color),
    xaxis=dict(
        showgrid=False,
        zerolinecolor=dark_bg_color,
        title_text=None  # Removed x-axis title
    ),
    margin=dict(l=10, r=10, t=40, b=10)
)

# Set y-axes titles with major grid lines only and both starting at 0
fitness_fig.update_yaxes(
    title_text="Hours", 
    secondary_y=False, 
    showgrid=True,
    gridwidth=1,
    gridcolor=dark_grid_color,
    minor_showgrid=False,
    rangemode="tozero"  # Force axis to start at zero
)
fitness_fig.update_yaxes(
    title_text="Fitness Score", 
    secondary_y=True, 
    title_font=dict(color="#ffffff"), 
    showgrid=False,  # Removed gridlines for secondary y-axis
    minor_showgrid=False,
    rangemode="tozero"  # Force axis to start at zero
)

layout_fitness = dbc.Row([
    dbc.Col(dcc.Graph(figure=fitness_fig), md=12),
])

# 3. Weekly Training Patterns and Start Hour Distribution
# Create weekly pattern chart
weekday_fig = px.bar(
    weekday_df,
    x="weekday", y="duration_hr", color="type",
    title="🗓️ What Days Are You Active? 🗓️",
    color_discrete_map=color_map,
    labels={"duration_hr": "Hours", "weekday": ""}  # Removed x-axis label
).update_traces(
    hovertemplate="%{fullData.name}: %{y:.1f} hours<extra></extra>"
).update_layout(
    template=dark_template,
    showlegend=False,  # Remove legend
    xaxis=dict(
        title_text=None,  # Removed x-axis title
        tickfont=dict(size=10),
        showgrid=False
    ),
    yaxis=dict(
        showgrid=True,
        title_text=None,
        gridwidth=1,
        gridcolor=dark_grid_color,
        minor_showgrid=False
    ),
    margin=dict(l=10, r=10, t=40, b=10)
)
# Create activity start hour bar chart showing sum of hours
start_hour_fig = px.bar(
    hourly_df, 
    x='hour_of_day',
    y='duration_hr',
    color='type',
    title='⏰ Activity Timing ⏰',
    labels={'hour_of_day': '', 'duration_hr': 'Hours'},  # Removed x-axis title
    color_discrete_map=color_map,
).update_traces(
    hovertemplate="%{fullData.name}: %{y:.1f} hours<extra></extra>"
).update_layout(
    template=dark_template,
    showlegend=False,  # Remove legend
    bargap=0.1,
    xaxis=dict(
        range=[-0.5, 23.5],  # Ensure full hour range is shown
        tickmode='linear',
        tick0=0,
        dtick=2,  # Show every 2 hours
        showgrid=False,
        title_text=None
    ),
    yaxis=dict(
        showgrid=True,
        title_text=None,
        gridwidth=1,
        gridcolor=dark_grid_color,
        minor_showgrid=False
    ),
    margin=dict(l=10, r=10, t=40, b=10)
)

# Combined layout for weekly patterns and start hour
layout_weekly_patterns = dbc.Row([
    dbc.Col(dcc.Graph(figure=weekday_fig), md=6),
    dbc.Col(dcc.Graph(figure=start_hour_fig), md=6)
])

# 4. Cumulative Stats
layout_cumulative = dbc.Row([
    dbc.Col(dcc.Graph(
        figure=px.area(
            count_pivot,
            x="date", y=count_pivot.columns[1:],
            title="🏆 What Activities Are You Doing Most? 🏆",
            color_discrete_map=color_map,
            labels={"value": "Count", "date": "", "variable": "Activity Type"}  # Removed x-axis label
        ).update_traces(
            hovertemplate="%{y:.1f} activities - %{fullData.name}<br>%{x|%b %Y}<extra></extra>"
        ).update_layout(
            template=dark_template,
            showlegend=False,  # Remove legend
            xaxis=dict(
                showgrid=False,
                title_text=None  # Removed x-axis title
            ),
            yaxis=dict(
                showgrid=True,
                title_text=None,
                gridwidth=1,
                gridcolor=dark_grid_color,
                minor_showgrid=False
            )
        )
    ), md=6),
    dbc.Col(dcc.Graph(
        figure=px.area(
            time_pivot,
            x="date", y=time_pivot.columns[1:],
            title="⏱️ Training Hours Accumulation ⏱️",
            color_discrete_map=color_map,
            labels={"value": "Hours", "date": "", "variable": "Activity Type"}  # Removed x-axis label
        ).update_traces(
            hovertemplate="%{y:.1f} hours - %{fullData.name}<br>%{x|%b %Y}<extra></extra>"
        ).update_layout(
            template=dark_template,
            showlegend=False,  # Remove legend
            xaxis=dict(
                showgrid=False,
                title_text=None  # Removed x-axis title
            ),
            yaxis=dict(
                showgrid=True,
                title_text=None,
                gridwidth=1,
                gridcolor=dark_grid_color,
                minor_showgrid=False
            )
        )
    ), md=6)
])

# 5. Run Performance 
layout_scatter = dbc.Row([
    dbc.Col(dcc.Graph(
        figure=px.scatter(
            run_df,
            x="date", y="pace_min_per_km",
            size="distance", color="average_heartrate",
            title="🏃‍♂️ Run Pace Timeline 🏃‍♀️",
            labels={
                "pace_min_per_km": "Pace (min/km)",
                "date": "",  # Removed x-axis label
                "average_heartrate": "Avg HR",
                "distance": "Distance (m)"
            },
            color_continuous_scale="RdYlBu_r",
            hover_data=["name", "distance", "average_heartrate", "moving_time"]
        ).update_traces(
            marker=dict(opacity=0.7),
            selector=dict(mode='markers'),
            hovertemplate=
                          "Pace: %{customdata[0]} min/km<br>" +
                          "Distance: %{customdata[1]:.1f} km<br>" + 
                          "Avg HR: %{customdata[2]:.0f} bpm<br>" +
                          "Time: %{customdata[3]} min<extra></extra>",
            customdata=run_df.apply(lambda x: [
                f"{int(x['pace_min_per_km'])}:{int((x['pace_min_per_km'] % 1) * 60):02d}", 
                x['distance']/1000, 
                x['average_heartrate'],
                f"{int(x['moving_time']//60):02d}:{int(x['moving_time']%60):02d}"
            ], axis=1).tolist()
        ).update_layout(
            template=dark_template,
            xaxis=dict(
                showgrid=False,
                title_text=None  # Removed x-axis title
            ),
            yaxis=dict(
                showgrid=True,
                title_text=None,
                gridwidth=1,
                gridcolor=dark_grid_color,
                minor_showgrid=False,
                autorange="reversed"  # This flips the y-axis so lower values are higher
            ),
            coloraxis_colorbar=dict(
                tickformat="d"  # Format to show whole numbers
            )
        )
    ), md=6),
    
    dbc.Col(dcc.Graph(
        figure=px.scatter(
            run_df,
            x="average_heartrate", 
            y="pace_min_per_km",
            color="year", 
            color_continuous_scale="RdYlBu_r",
            size="distance",
            title="❤️ Run Efficiency ⚡",
            labels={
                "pace_min_per_km": "Pace (min/km)",
                "average_heartrate": "Average Heart Rate (bpm)",
                "year": "Year",
                "distance": "Distance (m)"
            },
            hover_data=["name", "date", "distance", "moving_time"]
        ).update_traces(
            marker=dict(opacity=0.7, line=dict(width=0.5, color='white')),
            selector=dict(mode='markers'),
            hovertemplate=
                          "%{customdata[0]} pace @ %{x:.0f} bpm<br>" +
                          "%{customdata[1]}<br>" +
                          "Distance: %{customdata[2]:.1f} km<br>" + 
                          "Time: %{customdata[3]} min<extra></extra>",
            customdata=run_df.apply(lambda x: [
                f"{int(x['pace_min_per_km'])}:{int((x['pace_min_per_km'] % 1) * 60):02d}", 
                x['date'],
                x['distance']/1000, 
                f"{int(x['moving_time']//60):02d}:{int(x['moving_time']%60):02d}"
            ], axis=1).tolist()
        ).update_layout(
            template=dark_template,
            xaxis=dict(
                showgrid=True,
                title_text=None,
                gridwidth=1,
                gridcolor=dark_grid_color,
                title_font=dict(size=12),
            ),
            yaxis=dict(
                showgrid=True,
                title_text=None,
                gridwidth=1,
                gridcolor=dark_grid_color,
                minor_showgrid=False,
                title_font=dict(size=12),
                autorange="reversed"  # Flip the y-axis to make lower paces appear higher (faster)
            ),
            legend=dict(
                title="Year",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            coloraxis_colorbar=dict(
                tickvals=list(range(int(min(run_df["year"])), 
                                  int(max(run_df["year"])) + 1, 
                                  1)),  # Show each year individually
                tickformat="d"  # Format to show whole numbers
            )
        )
    ), md=6)
])

# 6. Bubble Graph for Last 8 weeks of activities
# Get the current date and calculate the start date (8 weeks ago from the most recent Monday)
most_recent_date = df['date'].max()
today = datetime.now().date()
recent_date = max(most_recent_date, today)
# Find the most recent Monday
days_since_monday = recent_date.weekday()
most_recent_monday = recent_date - timedelta(days=days_since_monday)
start_date = most_recent_monday - timedelta(weeks=8)

# Filter data for last 8 weeks
recent_df = df[df['date'] >= start_date].copy()
# Convert date column to datetime for proper calculation
recent_df['date_dt'] = pd.to_datetime(recent_df['date'])
recent_df['week_number'] = ((recent_df['date_dt'] - pd.to_datetime(start_date)).dt.days // 7)
recent_df['weekday'] = recent_df['start_date_local'].dt.weekday  # 0 = Monday, 6 = Sunday

# Calculate total duration and weighted average heart rate per day
daily_stats = recent_df.groupby(['date', 'week_number', 'weekday']).apply(
    lambda x: pd.Series({
        'duration_hr': x['duration_hr'].sum(),
        'avg_hr': np.average(
            x['average_heartrate'].fillna(0), 
            weights=x['duration_hr'],
            axis=0
        ) if x['average_heartrate'].notna().any() else 0,
        'activity_types': ', '.join(x['type'].unique())  # Collect all activity types for the day
    })
).reset_index()

# Create a complete calendar grid with all days in the period
date_range = pd.date_range(start=start_date, end=recent_date, freq='D')
start_date_ts = pd.Timestamp(start_date)
calendar_df = pd.DataFrame({
    'date': date_range.date,
    'week_number': [(d - start_date_ts).days // 7 for d in date_range],
    'weekday': [d.weekday() for d in date_range]
})
calendar_df = calendar_df.merge(daily_stats, on=['date', 'week_number', 'weekday'], how='left')
calendar_df['duration_hr'] = calendar_df['duration_hr'].fillna(0)
calendar_df['avg_hr'] = calendar_df['avg_hr'].fillna(0)
calendar_df['activity_types'] = calendar_df['activity_types'].fillna('')

# Apply transformation to make bubble sizes more distinct
calendar_df['bubble_size'] = calendar_df['duration_hr'] * 60

# Get the actual heart rate range from the data for colorbar scaling
min_hr = max(50, calendar_df[calendar_df['avg_hr'] > 0]['avg_hr'].min())
max_hr = min(200, calendar_df['avg_hr'].max())
midpoint_hr = (min_hr + max_hr) / 2

# Create bubble chart with inverted y-axis (recent weeks at bottom)
bubble_fig = px.scatter(
    calendar_df,
    x='weekday',
    y='week_number',
    size='bubble_size',  # Use duration for size
    color='avg_hr',      # Use heart rate for color
    color_continuous_scale='RdYlBu_r',  # Red for high HR, blue for low HR
    title='📅 Workout Calendar - Last 2 Months 🗓️',
    labels={
        'weekday': 'Day of Week',
        'week_number': 'Week',
        'avg_hr': 'Avg HR (bpm)',
        'duration_hr': 'Hours'
    },
    size_max=60,  # Increase maximum bubble size
    color_continuous_midpoint=midpoint_hr,  # Dynamic midpoint based on data
    range_color=[min_hr, max_hr]  # Set color range to match actual heart rate range
)

# Format the hover template with HR, duration, and activity types information
bubble_fig.update_traces(
    hovertemplate="Date: %{customdata[0]}<br>Time: %{customdata[1]} min<br>Avg HR: %{customdata[2]} bpm<br>Activities: %{customdata[3]}",
    customdata=calendar_df.apply(lambda x: [
        x['date'].strftime('%b %d, %Y'),
        f"{int(x['duration_hr'] * 60)}:{int((x['duration_hr'] * 60 % 1) * 60):02d}" if x['duration_hr'] > 0 else "0:00",
        f"{int(x['avg_hr'])}" if x['avg_hr'] > 0 else "N/A",
        x['activity_types'] if x['activity_types'] else "None"
        ], axis=1).tolist()
)

# Customize layout
bubble_fig.update_layout(
    template=dark_template,
    xaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        showgrid=False,
        title_text=None,
        zeroline=False,
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(0, 9)),  # Show week numbers from 0 to 8
        ticktext=[f"Week {i}" for i in range(0, 9)],
        showgrid=True,
        title_text=None,
        gridwidth=1,
        gridcolor=dark_grid_color,
        minor_showgrid=False,
        autorange='reversed'  # Invert y-axis to show most recent weeks at the bottom
    )
)
# Layout for the bubble chart
layout_bubble = dbc.Row([
    dbc.Col(dcc.Graph(figure=bubble_fig), md=12)
])

# 7. Bonus Stats
# First, prepare data for recent activities
recent_activities = df.sort_values("start_date_local", ascending=False).head(5)
# Define activity type emojis
activity_emoji = {
    'Run': '🏃',
    'Ride': '🚴',
    'Racquet Sports': '🏓',
    'Cardio': '💪',
    'Weight Training': '🏋️',
    'Hike': '🥾',
    'Workout': '🤸',
    'Walk': '🚶'
}

# Find the "best" activity
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
    "Duration": recent_activities["duration_hr"].apply(lambda x: f"{int(x * 60)}:{int((x * 60 % 1) * 60):02d} min"),
    "Details": recent_activities.apply(
        lambda row: f"{row['distance']/1000:.1f}km {'🔥' if row.name == max_distance_idx else ''}" 
            if row['type'] in ['Run', 'Ride', 'Hike']
        else f"{row.get('average_heartrate', 0):.0f} bpm {'❤️' if row.name == max_hr_idx else ''}" 
            if pd.notnull(row.get('average_heartrate')) 
        else "", 
        axis=1
    )
})

# 7. Bonus Stats
layout_bonus = dbc.Row([
    dbc.Col([
        html.H4("🏆 Overall Stats", className="text-center my-3"),
        dash_table.DataTable(
            data=individual_stats_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in individual_stats_df.columns],
            style_table={"width": "100%", "margin": "0 auto"},
            style_cell={
                "textAlign": "left", 
                "padding": "5px",
                "backgroundColor": dark_paper_color,
                "color": dark_text_color
            },
            style_header={
                "fontWeight": "bold", 
                "backgroundColor": "#2C2C2C", 
                "color": dark_text_color
            },
        )
    ], md=6, className="mb-4"),
    
    dbc.Col([
        html.H4("⏱️ Recent Activities", className="text-center my-3"),
        dash_table.DataTable(
            data=recent_activities_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in recent_activities_df.columns],
            style_table={"width": "100%", "margin": "0 auto", "marginBottom": "20px"},
            style_cell={
                "textAlign": "left", 
                "padding": "5px",
                "backgroundColor": dark_paper_color,
                "color": dark_text_color
            },
            style_header={
                "fontWeight": "bold", 
                "backgroundColor": "#2C2C2C", 
                "color": dark_text_color
            },
        )
    ], md=6)
], className="mt-4")

# New row for YoY Stats centered
layout_yoy = dbc.Row([
    dbc.Col([], md=3),  # Empty column for centering
    dbc.Col([
        html.H4("📊 Year-over-Year Comparison", className="text-center my-3"),
        dash_table.DataTable(
            data=comparative_stats_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in comparative_stats_df.columns],
            style_table={"width": "100%", "margin": "0 auto"},
            style_cell={
                "textAlign": "left", 
                "padding": "5px",
                "backgroundColor": dark_paper_color,
                "color": dark_text_color
            },
            style_header={
                "fontWeight": "bold", 
                "backgroundColor": "#2C2C2C", 
                "color": dark_text_color
            },
        )
    ], md=6),
    dbc.Col([], md=3),  # Empty column for centering
], className="mt-4")

# Final app layout - following the order of the dashboard
app.layout = dbc.Container([
    html.H1("Strava Activity Dashboard 🏃‍♂️📊", className="text-center my-4"),
    layout_pie,
    html.Hr(),
    layout_bonus,
    layout_yoy,
    html.Hr(),
    layout_fitness,
    html.Hr(),
    layout_weekly_patterns,
    html.Hr(),
    layout_cumulative,
    html.Hr(),
    layout_scatter,
    html.Hr(),
    layout_bubble,
], fluid=True, style={"backgroundColor": dark_bg_color})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
