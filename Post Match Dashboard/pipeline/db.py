import pandas as pd
import subprocess
import datetime
from scraper import get_shots_data
from schedule import get_match_date_and_id

match_date, match_id = get_match_date_and_id(team_id=8650)
data_shots=get_shots_data(match_id)

date = datetime.datetime.now().strftime("%d-%m-%y")

# subprocess.run(['python', 'processing.py'])


# data = pd.read_csv(f'./Data/process_data/match_data{date}.csv')
data = pd.read_csv(f"../Data/process_data/match_data{date}.csv")

competition = data_shots['competition'].iloc[0]


data['competition'] = competition


from sqlalchemy import create_engine


engine = create_engine('')



data.to_sql('opta_event_data', engine, if_exists='append', index=False)

data_shots.to_sql('fotmob_shots_data', engine, if_exists='append', index=False)


#