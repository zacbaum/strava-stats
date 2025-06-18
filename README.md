# Strava Stats

A tool for analyzing and visualizing my Strava activity data.

## Features

- Fetches my Strava activities using the Strava API
- Generate statistics and insights about my workouts
- Visualize my running, cycling, and other activity trends
- Track my progress over time

## Installation

```bash
git clone https://github.com/yourusername/strava-stats.git
cd strava-stats
pip install -r requirements.txt
```

## Usage

1. Set up Strava API credentials
2. Run the main script to fetch activities
3. Generate reports and visualizations

```bash
python get_data.py
python app.py
```

## Configuration

Create a `.env` file with your Strava API credentials:

```
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_REFRESH_TOKEN=your_refresh_token
```
