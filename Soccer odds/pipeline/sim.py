
import pandas as pd

from scipy.stats import poisson

import sqlite3



import pickle

# Load the saved model_params
with open("data/model.pkl", "rb") as f:
    loaded_model_params = pickle.load(f)




con = sqlite3.connect("../Database/EPL23_24.db")
cur = con.cursor()



data = pd.read_sql("Select * from shots_data ", con)

# Convert matchId column to integer type
data['matchId'] = data['matchId'].astype(int)


# Group by match and team and sum expected goals for each team in each match
team_goals = data.groupby(['matchId', 'TeamName'])['expectedGoals'].sum().reset_index()

# Create a new DataFrame with expected goals against for each team in each match
opponent_goals = team_goals.groupby('matchId')['expectedGoals'].sum().reset_index()
opponent_goals = opponent_goals.rename(columns={'TeamName': 'OpponentTeam', 'expectedGoals': 'ExpectedGoalsAgainst'})

team_goals = team_goals.merge(opponent_goals, on='matchId', how='left')
team_goals['ExpectedGoalsAgainst'] = team_goals['ExpectedGoalsAgainst'] - team_goals['expectedGoals']



# Group by team and sum expected goals for and expected goals against for each team
xg_table = team_goals.groupby('TeamName')[['expectedGoals', 'ExpectedGoalsAgainst']].sum()


data['expectedGoals'] = data['expectedGoals'].fillna(0)

def simulate_match_on_shots(match_id, shot_df):
    '''
    This function takes a match ID and simulates an outcome based on the shots
    taken by each team.
    '''

    shots = shot_df[shot_df['matchId'] == match_id]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']

    home_goals = 0
    if shots_home['expectedGoals'].shape[0] > 0:
        for shot in shots_home['expectedGoals']:
            # Sample a number from the Poisson distribution using the xG value as the lambda parameter
            goals = poisson.rvs(mu=shot)
            home_goals += goals

    away_goals = 0
    if shots_away['expectedGoals'].shape[0] > 0:
        for shot in shots_away['expectedGoals']:
            # Sample a number from the Poisson distribution using the xG value as the lambda parameter
            goals = poisson.rvs(mu=shot)
            away_goals += goals

    return {'home_goals':home_goals, 'away_goals':away_goals}


def iterate_k_simulations_on_match_id(match_id, shot_df, k=int):
    '''
    Performs k simulations on a match, and returns the probabilites of a win, loss, draw.
    '''
    # Count the number of occurances
    home = 0
    draw = 0
    away = 0
    # Get the teams

    home_team_id = shot_df[shot_df['Venue'] == 'Home']
    away_team_id =  shot_df[shot_df['Venue'] == 'Away']


    for i in range(k):
        simulation = simulate_match_on_shots(match_id, shot_df)
        if simulation['home_goals'] > simulation['away_goals']:
            home += 1
        elif simulation['home_goals'] < simulation['away_goals']:
            away += 1
        else:
            draw += 1
    home_prob = home/k
    draw_prob = draw/k
    away_prob = away/k
    return {'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob, 'match_id': match_id}



match_probs = []
for index, match in enumerate(data['matchId']):
    outcome_probs = iterate_k_simulations_on_match_id(match, data,100)
    match_probs.append(outcome_probs)
    if index % 10 == 0:
        print(f'{index/data.shape[0]:.1%} done.')





match_probs = pd.DataFrame(match_probs)

match_probs.to_csv('data/match_probs.csv')
