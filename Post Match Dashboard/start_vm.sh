#!/usr/bin/env bash

# Fetch match schedule from API
matches=$(python3 - <<EOF
import requests
import json
def get_match_date_and_id(team_id):
    # Get data from the API
    response = requests.get(f'https://www.fotmob.com/api/teams?id={team_id}&ccode3=USA_MA')
    data = json.loads(response.content)

    # Extract fixtures data
    fixtures = data['fixtures']['allFixtures']['fixtures']

    # Flatten the JSON and create a DataFrame
    df = json_normalize(fixtures)

    # Get today's date in the format: 'YYYY-MM-DD'
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Check if there is a match today for the specified team
    today_match = df[(df['status.finished'] == True) & (df['status.utcTime'].str.startswith(today_date))]

    if not today_match.empty:
        match_date_str = today_match['status.utcTime'].iloc[0]
        match_date = datetime.strptime(match_date_str, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%d-%m-%y')

        match_id = today_match['id'].iloc[0]

        return f'{match_date}', match_id
    else:
        return None, None

EOF
)

# Check if there is a match today
if [[ "$matches" == *"today's date"* ]]; then
   # Start the VM
   gcloud compute instances start INSTANCE_NAME --zone=ZONE
   sleep 120
   python3 main.py
fi


sudo gsutil cp -r gs://soccer-pipelines/Football_Analysis_Tools/* /var/www/html/


sudo gsutil cp -r gs://soccer-pipelines/Post_Match_Dashboard/ ~/soccer/

gs://soccer-pipelines/Post_Match_Dashboard/