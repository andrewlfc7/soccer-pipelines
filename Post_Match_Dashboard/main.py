import subprocess
import os

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Full paths to the Python scripts
stats_avg_script = os.path.join(current_directory, 'stats_avg.py')

post_match_script = os.path.join(current_directory, 'Post_match.py')
player_dashboard_script = os.path.join(current_directory, 'player_dashboard.py')

db_script_path = os.path.join(current_directory, 'pipeline', 'db.py')

# Run the scripts
subprocess.run(['python', db_script_path], check=True)


subprocess.run(['python', stats_avg_script], check=True)
subprocess.run(['python', post_match_script], check=True)
subprocess.run(['python', player_dashboard_script])


# twitter_api_script = os.path.join(current_directory, 'twitter-api.py')
# subprocess.run(['python', twitter_api_script])
