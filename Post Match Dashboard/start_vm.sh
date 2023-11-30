

# Activate the virtual environment
source myenv/bin/activate

# Fetch match schedule from API
matches=$(python3 - <<EOF
import requests
import json
from datetime import datetime
from pandas import json_normalize

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

        print(f'Match today: {match_date}, Match ID: {match_id}')
    else:
        print('No match today')

get_match_date_and_id(8650)
EOF
)

# Check if there is a match today and start the VM
if [[ "$matches" == *"Match today"* ]]; then
    # Start the VM
    gcloud compute instances start soccer --zone=us-west4-b

    # Check if the VM start was successful
    if [ $? -eq 0 ]; then
        echo "VM started successfully"
    else
        echo "Failed to start VM"
    fi
fi

# Deactivate the virtual environment
deactivate
