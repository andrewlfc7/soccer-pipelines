from sqlalchemy import exc
import requests
import json
import pandas as pd
#%%
# Function to check if played games are already in the database
def does_matches_exist(table_name,engine):

    # Write your SQL query to retrieve data
    query = f"SELECT * FROM {table_name}"

    try:
        # Execute the query and fetch the data into a DataFrame
        shots_data = pd.read_sql(query, engine)
        return len(shots_data) > 0
    except exc.ProgrammingError as e:
        # Catch UndefinedTable error and return False
        return False


def get_matches_data(table_name,engine):
    # Write your SQL query to retrieve data
    query = f"SELECT * FROM {table_name}"

    try:
        # Execute the query and fetch the data into a DataFrame
        matches_data = pd.read_sql(query, engine)
        return matches_data
    except exc.ProgrammingError as e:
        # Catch UndefinedTable error and return None
        print(f"Error: {e}")
        return None

def calculate_xg_table(shots_data):
    # Group by match and team and sum expected goals for each team in each match
    team_goals = shots_data.groupby(['matchId', 'TeamName', 'Venue', 'teamId'])['expectedGoals'].sum().reset_index()

    # Create a new DataFrame with expected goals against for each team in each match
    opponent_goals = team_goals.groupby('matchId')['expectedGoals'].sum().reset_index()
    opponent_goals = opponent_goals.rename(columns={'TeamName': 'OpponentTeam', 'expectedGoals': 'ExpectedGoalsAgainst'})

    team_goals = team_goals.merge(opponent_goals, on='matchId', how='left')
    team_goals['ExpectedGoalsAgainst'] = team_goals['ExpectedGoalsAgainst'] - team_goals['expectedGoals']

    # Group by team and sum expected goals for and expected goals against for each team
    xg_table = team_goals.groupby(['TeamName', 'teamId'])[['expectedGoals', 'ExpectedGoalsAgainst']].sum()

    return xg_table

def calculate_per_90_metrics(sim_table):
    sim_table_copy = sim_table.copy()  # Create a copy to avoid modifying the original DataFrame
    sim_table_copy['played'] = sim_table_copy['w'] + sim_table_copy['d'] + sim_table_copy['l']
    sim_table_copy['ppg'] = sim_table_copy['points'] / sim_table_copy['played']
    sim_table_copy['gf_per90'] = sim_table_copy['gf'] / sim_table_copy['played']
    sim_table_copy['ga_per90'] = sim_table_copy['ga'] / sim_table_copy['played']
    return sim_table_copy

def get_team_id_mapping(league_id):
    # Make API request to Fotmob to get league information
    url = f'https://www.fotmob.com/api/leagues?id={league_id}&ccode3=USA_MA'
    response = requests.get(url)
    data = json.loads(response.content)

    # Extract team information from the response
    table = pd.DataFrame(data['table'][0]['data']['table']['all'])

    # Create a mapping between team names and team IDs
    team_id_mapping = table.set_index('name')['id'].to_dict()

    return team_id_mapping

def does_fixtures_exist(table_name,engine):

    # Write your SQL query to retrieve data
    query = f"SELECT * FROM {table_name}"

    try:
        # Execute the query and fetch the data into a DataFrame
        fixtures_data = pd.read_sql(query, engine)
        return len(fixtures_data) > 0
    except exc.ProgrammingError as e:
        # Catch UndefinedTable error and return False
        return False

def get_fixtures_data(table_name,engine):
    # Write your SQL query to retrieve data
    query = f"SELECT * FROM {table_name}"

    try:
        # Execute the query and fetch the data into a DataFrame
        fixtures = pd.read_sql(query, engine)
        return fixtures
    except exc.ProgrammingError as e:
        # Catch UndefinedTable error and return None
        print(f"Error: {e}")
        return None


def xpoints_table_pre90_stats(xPoints_table):
    xPoints_table['points'] = xPoints_table['points'].astype(int)
    xPoints_table['pts'] = xPoints_table['pts'].astype(int)

    xPoints_table['ppg'] = xPoints_table['pts'] / xPoints_table['played']
    xPoints_table['gf_per90'] = xPoints_table['GF'] / xPoints_table['played']
    xPoints_table['ga_per90'] = xPoints_table['GA'] / xPoints_table['played']

    return xPoints_table
