import requests
import urllib3
import pandas as pd
import time
import os
from dotenv import load_dotenv


def main():
    # Suppress SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Strava credentials
    load_dotenv()  # take environment variables from .env
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
    
    # API endpoints
    auth_url = "https://www.strava.com/oauth/token"
    activities_url = "https://www.strava.com/api/v3/athlete/activities"

    # FORCE_REFRESH=true bypasses the existing-id skip, re-fetching every
    # activity from scratch — useful when new columns are added below.
    force_refresh = os.getenv("FORCE_REFRESH", "").lower() in ("1", "true", "yes")

    # Columns to extract
    columns = [
        'id', 'type', 'name', 'distance', 'moving_time', 'elapsed_time',
        'total_elevation_gain', 'start_date', 'start_date_local', 'start_latlng', 'kilojoules',
        'average_heartrate', 'max_heartrate', 'elev_high', 'elev_low',
        'average_speed', 'max_speed',
        # Location text fields — Strava sometimes populates these (currently
        # returns None for most activities, but kept for future-proofing).
        'location_city', 'location_state', 'location_country',
        # IANA timezone is populated for every activity — used as the primary
        # fallback location when start_latlng is empty (indoor sessions etc.).
        'timezone',
        # Encoded GPS polyline of the activity's route (extracted from the
        # nested `map` object below). Used to render route lines on the map.
        'summary_polyline',
    ]

    # Step 1: Get access token using refresh token
    print("Requesting new access token...")
    auth_response = requests.post(auth_url, data={
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'refresh_token': REFRESH_TOKEN,
        'grant_type': 'refresh_token'
    }, verify=False)

    if auth_response.status_code != 200:
        print("❌ Authentication failed:", auth_response.text)
        exit(1)

    access_token = auth_response.json().get('access_token')
    headers = {'Authorization': f'Bearer {access_token}'}
    print("✅ Access token received.\n")

    # Step 2: Load existing activities if available
    existing_ids = set()
    if force_refresh:
        existing_df = pd.DataFrame(columns=columns)
        print("🔁 FORCE_REFRESH set — ignoring existing CSV, re-fetching all activities.")
    else:
        try:
            existing_df = pd.read_csv('activities.csv', dtype={'id': str})
            existing_ids = set(existing_df['id'].astype(str))
            print(f"📁 Loaded {len(existing_df)} existing activities.")
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=columns)
            print("📁 No existing CSV found. Starting from scratch.")

    # Step 3: Fetch new activities. Track every ID Strava currently returns so
    # we can detect deletions after the fetch completes.
    page = 1
    per_page = 200
    new_activities = []
    seen_ids = set()
    fetch_complete = False

    while True:
        response = requests.get(activities_url, headers=headers, params={
            'page': page, 'per_page': per_page
        })

        if response.status_code == 429:
            print("⚠️ Rate limit hit. Waiting 15 minutes...")
            time.sleep(15 * 60)
            continue

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        page_data = response.json()
        if not page_data:
            print(f"✅ No more activities after page {page - 1}.")
            fetch_complete = True
            break

        new_count = 0
        for act in page_data:
            act_id = str(act.get('id'))
            seen_ids.add(act_id)
            if act_id in existing_ids:
                continue
            row = {col: act.get(col, None) for col in columns}
            # summary_polyline is nested inside the `map` object
            row['summary_polyline'] = (act.get('map') or {}).get('summary_polyline') or None
            new_activities.append(row)
            new_count += 1

        print(f"📄 Page {page}: {len(page_data)} fetched, {new_count} new")
        page += 1

        if len(page_data) < per_page:
            fetch_complete = True
            break

    # Detect deletions: any ID in the CSV but not in Strava's response. Only
    # trust this if the fetch finished cleanly — otherwise a partial response
    # could falsely flag activities as deleted.
    deleted_ids = set()
    if fetch_complete and existing_ids:
        deleted_ids = existing_ids - seen_ids
        if deleted_ids:
            print(f"🗑️  Detected {len(deleted_ids)} activities removed from Strava — dropping from CSV")
            existing_df = existing_df[~existing_df['id'].astype(str).isin(deleted_ids)]

    # Step 4: Save results if anything changed (additions or deletions)
    if new_activities or deleted_ids:
        new_df = pd.DataFrame(new_activities)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # format='ISO8601' handles both pandas' default serialised form
        # ("2025-06-17 06:44:20+00:00") and Strava's API form ("2025-06-17T06:44:20Z")
        # in the same column — they differ after a FORCE_REFRESH writes the CSV
        # in pandas form and the next incremental pull adds API-form rows.
        combined_df['start_date'] = pd.to_datetime(combined_df['start_date'], format='ISO8601')
        combined_df = combined_df.sort_values(by='start_date', ascending=False)
        combined_df.to_csv('activities.csv', index=False, encoding='utf-8')
        print(f"\n✅ Added {len(new_activities)} · removed {len(deleted_ids)} · total now: {len(combined_df)}")
    else:
        print("📭 No new activities and no deletions detected.")

    print("📁 All activities saved to activities.csv")

if __name__ == "__main__":
    main()
