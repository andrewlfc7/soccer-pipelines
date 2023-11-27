import subprocess



# subprocess.run(['python', 'Post Match Dashboard/pipeline/main.py'],check=True)

subprocess.run(['python', 'Post_match.py'],check=True)
subprocess.run(['python', 'stats_avg.py'],check=True)
subprocess.run(['python', 'player_dashboard.py'])
