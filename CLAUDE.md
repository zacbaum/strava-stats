# Strava Stats Dashboard

A personal Strava training dashboard deployed on Render. Pulls activity data daily from the Strava API and renders an interactive Dash/Plotly view of training load, routes, and PRs.

## Files

| File | Purpose |
|---|---|
| `app.py` | Dash app. ~1700 LOC organised into `#######################` section blocks: imports, config, data prep, model, metrics, layouts, callback |
| `get_data.py` | Strava API fetcher. `FORCE_REFRESH=true` env var bypasses the existing-id skip to backfill new columns across all activities |
| `activities.csv` | Raw activity data, 2020–present. Includes `summary_polyline`, `timezone`, `location_*` columns |
| `assets/style.css` | Inter font + card hover styling (auto-loaded by Dash) |
| `requirements.txt` | pandas, numpy, plotly, dash, dash-bootstrap-components, requests, python-dotenv, pytz |
| `render.yaml` | Render deploy config |
| `.github/workflows/update_data.yml` | Daily 10:00 UTC cron pulling new Strava activities |
| `.github/workflows/ci.yml` | Syntax + import smoke test on every push to main |

## Local commands

```bash
# Syntax check (also runs in CI)
python -m py_compile app.py get_data.py

# Re-pull all activity history (slower; useful when adding columns)
FORCE_REFRESH=true PYTHONIOENCODING=utf-8 python get_data.py

# Incremental pull (what the cron runs)
python get_data.py

# Run the dashboard locally
python app.py   # serves on http://0.0.0.0:10000
```

## Architectural decisions worth knowing

- **Fitness model**: Banister TRIMP via HR reserve. `HR_MAX=195`, `HR_REST=55` (both configurable up top). CTL=42-day EWMA, ATL=7-day EWMA, TSB=CTL−ATL. No-HR fallback is `duration_hr × 100`.
- **Sport categories**: `SPORT_CATEGORIES` in `app.py` maps the full Strava sport-type enum into display buckets (Run, Ride, Walk, Hike, Racquet Sports, Weight Training, Cardio, Water Sports, Winter Sports, Mind & Body, Other). Generic "Workout" uploads are name-sniffed via `WORKOUT_NAME_CATEGORIES` (padel/pickleball/squash/tennis → Racquet Sports, etc.). Run/Ride/Hike/Walk names are load-bearing — PR cards and pace logic reference them directly.
- **Strain**: WHOOP-style 0–21 daily score (log map of training load, 99th-percentile day ≈ 21). Surfaced only as a KPI card — the strain time-series chart was removed (unreadable on mobile).
- **Year projection**: pattern-based (`ytd + (prev_year_total − prev_year_at_same_day)`), not linear. Lines up with the YoY trajectory chart.
- **Global filter dropdown** (top of trend section): controls only the 4 filter-aware charts (fitness, rolling, heatmap, cumulative). Pies/PRs/KPIs always show full history. Defaults to "Past Year".
- **Map**: 270 routes rendered as two passes (warm base + heat overlay) for log-like saturation. Activity-weighted home centring + dynamic zoom. Region cluster circles fade out past zoom 6.5.
- **Date range**: no hardcoded cutoffs. Data flows the full 2020–present span everywhere; the dropdown handles time windowing.

## Conventions

- **Ask before committing**. Make changes, summarise pending diffs, offer tweaks, then push when the user says "ship it."
- **All decimals ≤ 2 places** in hovers/tooltips/displayed numbers.
- **Form zones** (TSB): `>+5` Fresh · `−10 to +5` Optimal · `−30 to −10` Productive · `<−30` Overreaching.
- **Plotly traces use the `SERIES` dict** at the top of `app.py` for semantic colors (`fitness`, `fatigue`, `form`, `rolling_7/28/365`, etc.) so the same meaning gets the same colour everywhere.
- **Activity colours** live in `color_map` (harmonised saturation; Run blue, Ride amber, Hike slate, etc.).
- **Sparkline height** in PR cards: 56 px.
- **Windows quirk**: get_data.py prints emoji; set `PYTHONIOENCODING=utf-8` when running from PowerShell to avoid `UnicodeEncodeError`.

## Deployment

- Render auto-deploys on push to `main`.
- The daily cron writes to `activities.csv` and commits via a PAT token (see `update_data.yml`).
- CI smoke test catches deploy-breakers (e.g. import errors, syntax errors) before Render does.

## Known open items

- **Google Maps Timeline integration**: not started. Google killed API access in 2024; would require user to export via Google Takeout and a parser would read the JSON. Discussed, not pursued.
- **`comparative_stats_df` "Overall"** uses the full unfiltered dataset; could optionally tie to the global filter dropdown.
- **Pace-at-150bpm narrative insight** currently shows the user as slower vs prior year — real signal, not a bug.
