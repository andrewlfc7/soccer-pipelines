#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import urllib3
from scipy.stats import poisson
from matplotlib.colors import ListedColormap
import seaborn as sns
from random import randint
from statistics import mode
from PIL import Image
import sqlite3
from tabulate import tabulate


#%%
con = sqlite3.connect("../Database/EPL23_24.db")
cur = con.cursor()

shots_data = pd.read_sql("Select * from shots_data ", con)

#%%
# Group by match and team and sum expected goals for each team in each match
team_goals = shots_data.groupby(['matchId', 'TeamName','Venue','teamId'])['expectedGoals'].sum().reset_index()

# Create a new DataFrame with expected goals against for each team in each match
opponent_goals = team_goals.groupby('matchId')['expectedGoals'].sum().reset_index()
opponent_goals = opponent_goals.rename(columns={'TeamName': 'OpponentTeam', 'expectedGoals': 'ExpectedGoalsAgainst'})

team_goals = team_goals.merge(opponent_goals, on='matchId', how='left')
team_goals['ExpectedGoalsAgainst'] = team_goals['ExpectedGoalsAgainst'] - team_goals['expectedGoals']

team_goals
#%%
# Group by team and sum expected goals for and expected goals against for each team
xg_table = team_goals.groupby([ 'TeamName' ,'teamId'])[['expectedGoals', 'ExpectedGoalsAgainst']].sum()

#%%
import requests
import json
response = requests.get('https://www.fotmob.com/api/leagues?id=47&ccode3=USA_MA')
data = json.loads(response.content)

#%%
table = pd.DataFrame(data['table'][0]['data']['table']['all'])
#%%
table[['GF', 'GA']] = table['scoresStr'].str.split('-', expand=True)


#%%
table['GF'] = pd.to_numeric(table['GF'])
table['GA'] = pd.to_numeric(table['GA'])

#%%
match_probs = pd.read_csv('data/match_probs.csv')
match_probs
#%%
match_probs = match_probs.rename(columns={'match_id':'matchId'})
#%%
last_match_probs = match_probs.groupby('matchId').tail(1)
last_match_probs
#%%
team_goals['matchId'] = team_goals['matchId'].astype(int)

#%%
merged_df = last_match_probs.merge(team_goals, on='matchId')
merged_df = merged_df.rename(columns={'expectedGoals':'home_ExpectedGoals','ExpectedGoalsAgainst':'away_ExpectedGoals'})
#%%
df = merged_df
# Create an empty dictionary to store each team's total expected points
expected_points = {}

# Loop over each unique matchId in the dataframe
for match_id in df['matchId'].unique():

    # Filter the dataframe by the current matchId and venue
    home_df = df.loc[(df['matchId'] == match_id) & (df['Venue'] == 'Home')]
    away_df = df.loc[(df['matchId'] == match_id) & (df['Venue'] == 'Away')]

    # Get the home and away team names and their expected goals
    home_team = home_df.iloc[0]['teamId']
    away_team = away_df.iloc[0]['teamId']
    home_expected_goals = home_df.iloc[0]['home_ExpectedGoals']
    away_expected_goals = away_df.iloc[0]['away_ExpectedGoals']

    # Get the match probabilities for a home win, away win, and draw
    home_prob = home_df.iloc[0]['home_prob']
    away_prob = away_df.iloc[0]['away_prob']
    draw_prob = home_df.iloc[0]['draw_prob']

    # Calculate the expected points for each team based on the match probabilities
    home_points = home_prob * 3 + draw_prob * 1
    away_points = away_prob * 3 + draw_prob * 1

    # Add the expected points to the total for each team
    if home_team in expected_points:
        expected_points[home_team] += home_points
    else:
        expected_points[home_team] = home_points

    if away_team in expected_points:
        expected_points[away_team] += away_points
    else:
        expected_points[away_team] = away_points

# Create a new dataframe with each team's name and total expected points
ranking_df = pd.DataFrame({ 'teamId': list(expected_points.keys()), 'ExpectedPoints': list(expected_points.values()) })

# Sort the dataframe by expected points in descending order
ranking_df = ranking_df.sort_values('ExpectedPoints', ascending=False)

# Reset the index of the dataframe to start at 1 instead of 0
ranking_df.index = range(1, len(ranking_df) + 1)

# Print the final ranking dataframe
print(ranking_df)

#%%
xPoints_table = table.merge(ranking_df, on='teamId')
