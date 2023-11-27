import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import mplsoccer to demo creating a pitch on gridsearch
from mplsoccer import Pitch
from mplsoccer import VerticalPitch
import sqlite3
from highlight_text import fig_text, ax_text
from ast import literal_eval
from unidecode import unidecode

import requests
import bs4
import json
from PIL import Image
import urllib
from Football_Analysis_Tools import fotmob_visuals as fotmobvis
from Football_Analysis_Tools import  whoscored_visuals as whovis
#%%
from Football_Analysis_Tools import whoscored_data_engineering as who_eng
import datetime


from sqlalchemy import create_engine
engine = create_engine('')


conn = engine.connect()

today = datetime.date.today()
today = today.strftime('%Y-%m-%d')



shots_query =f"""
SELECT * FROM fotmob_shots_data WHERE match_date = '{today}' AND ("teamId" = 8650 OR "match_id" IN (SELECT "match_id" FROM fotmob_shots_data WHERE "teamId" = 8650))

"""

query =f"""
SELECT * FROM opta_event_data WHERE match_date = '{today}' AND ("teamId" = 26 OR "match_id" IN (SELECT "match_id" FROM opta_event_data WHERE "teamId" = 26))
"""


shots_data = pd.read_sql(shots_query, conn)

data = pd.read_sql(query, conn)

data['playerName'] = data['playerName'].replace('Unknown', pd.NA)
data = data.dropna(subset=['playerId', 'playerName'])

Fotmob_matchID = shots_data['match_id'].iloc[0]



# boolean columns
bool_cols = ['isTouch',
             'is_open_play',
             'is_progressive',
             'is_pass_into_box',
             'won_possession',
             'key_pass',
             'assist',
             'FinalThirdPasses',
             'pre_assist',
             'switch']

# convert boolean columns to boolean values
for col in bool_cols:
    data[col] = data[col].astype(bool)



#%%

data['qualifiers'] = [literal_eval(x) for x in data['qualifiers']]
data['satisfiedEventsTypes'] = [literal_eval(x) for x in data['satisfiedEventsTypes']]
CrossPasses_set = set(['passCrossAccurate', 'passCrossInaccurate'])
data = data.copy()
data['is_cross'] = False
for index, row in enumerate(data['satisfiedEventsTypes']):
    set_element = set(row)
    if len(CrossPasses_set.intersection(set_element)) > 0:
        data.at[index, 'is_cross'] = True


#%%
# data[(~data['is_open_play']) & (data['assist']) & (data['key_pass'])]

#%%
# filter for rows where openplay is True
openplay_data = data.loc[data['is_open_play'] == True]

# filter for rows where assist and keypass are True but openplay is False
assist_keypass_data = data.loc[(data['assist'] == True) & (data['key_pass'] == True) & (data['is_open_play'] == False)]

# combine the two filtered dataframes
combined_data = pd.concat([openplay_data, assist_keypass_data])

# sort the combined dataframe by index
combined_data = combined_data.sort_index()

#%%
data=combined_data
#%%



def get_passes_df(events_dict):
    df = pd.DataFrame(events_dict)
    # create receiver column based on the next event
    # this will be correct only for successfull passes
    df["pass_recipient"] = df["playerName"].shift(-1)
    # filter only passes

    passes_ids = df.index[df['event_type'] == 'Pass']
    df_passes = df.loc[
        passes_ids,  ["id","minute", "x", "y", "endX", "endY", "teamId", "playerId","playerName", "event_type", "outcomeType","pass_recipient",'switch','pre_assist','assist','FinalThirdPasses','key_pass','is_open_play','is_progressive','is_cross']]

    return df_passes



passes_df = data[data['teamId']==26]

#%%
passes_df = get_passes_df(passes_df)
#%%
def find_offensive_actions(events_df):
    """ Return dataframe of in-play offensive actions from event data.
    Function to find all in-play offensive actions within a whoscored-style events dataframe (single or multiple
    matches), and return as a new dataframe.
    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
    Returns:
        pandas.DataFrame: whoscored-style dataframe of offensive actions.
    """

    # Define and filter offensive events
    offensive_actions = ['TakeOn', 'MissedShots', 'SavedShot', 'Goal', 'Carry','ShotOnPost']
    offensive_action_df = events_df[events_df['event_type'].isin(offensive_actions)].reset_index(drop=True)

    # Filter for passes with assists or pre-assists
    pass_df = events_df[(events_df['event_type'] == 'Pass') & ((events_df['assist'] == True) | (events_df['pre_assist'] == True))]

    # Concatenate offensive actions and passes with assists or pre-assists
    offensive_actions_df = pd.concat([offensive_action_df, pass_df]).reset_index(drop=True)

    return offensive_actions_df

offensive_actions = find_offensive_actions(data)
defensive_actions = who_eng.find_defensive_actions(data)

from unidecode import unidecode



centerbacks = data[(data['teamId'] == 26) & (data['position'].isin(['DC']))]['playerName'].dropna().unique()
fullbacks = data[(data['teamId'] == 26) & (data['position'].isin(['DR','DL']))]['playerName'].dropna().unique()
midfielders = data[(data['teamId'] == 26) & (data['position'].isin(['MC','DMR','DML','DM']))]['playerName'].dropna().unique()
forwards = data[(data['teamId'] == 26) & (data['position'].isin(['FWL', 'FW', 'FWR']))]['playerName'].dropna().unique()
subs = data[(data['teamId'] == 26) & (data['position'].isin(['Sub']))]['playerName'].dropna().unique()




response = requests.get('https://www.fotmob.com/api/teams?id=8650&ccode3=USA_MA')
player_id_data = json.loads(response.content)
all_players = []
player_data = player_id_data['squad']

for category, players in player_data:
    if category != 'coach':  # Exclude 'coach' category
        all_players.extend(players)

# Create a DataFrame from the combined player data
df_players = pd.DataFrame(all_players)[['name', 'id']]

new_player_data = {'name': 'Ben Doak', 'id': 1324871}
df_new_player = pd.DataFrame([new_player_data])
df_players = pd.concat([df_players, df_new_player], ignore_index=True)

player_id_dict = dict(zip(df_players['name'], df_players['id']))

# ...

# Function to get player ID, handling accents
# def get_player_id(name):
#     # First, try to find player ID with accents
#     player_id = player_id_dict.get(name)
#
#     if player_id is None:
#         # If not found, try a partial string match
#         matching_names = df_players[df_players['name'].str.contains(name, case=False, na=False)]
#
#         if len(matching_names) == 1:
#             player_id = matching_names.iloc[0]['id']
#
#         # If still not found, remove accents and try again
#         if player_id is None:
#             normalized_name = unidecode(name)
#             player_id = player_id_dict.get(normalized_name)
#
#     return player_id

def get_player_id(name):
    # First, try to find player ID with accents
    player_id = player_id_dict.get(name)

    if player_id is None:
        # If not found, split the name into first and last names
        names = name.split()
        if len(names) == 2:
            first_name, last_name = names

            # Try searching with both first and last names
            matching_first_names = df_players[df_players['name'].str.contains(first_name, case=False, na=False)]
            matching_last_names = df_players[df_players['name'].str.contains(last_name, case=False, na=False)]

            if len(matching_first_names) == 1:
                player_id = matching_first_names.iloc[0]['id']
            elif len(matching_last_names) == 1:
                player_id = matching_last_names.iloc[0]['id']

        # If still not found, remove accents and try again
        if player_id is None:
            normalized_name = unidecode(name)
            player_id = player_id_dict.get(normalized_name)

            if player_id is None:
                # If still not found, try to find shorter names without accents
                for player_name in player_id_dict.keys():
                    if (len(player_name) >= len(name) and
                            unidecode(player_name).startswith(unidecode(name))):
                        player_id = player_id_dict.get(player_name)
                        break

    return player_id



for player_name in centerbacks:
    player_id = get_player_id(player_name)
    try:
        player_logo_path = f'Data/player_image/{player_id}.png'
        club_icon = Image.open(player_logo_path).convert('RGBA')
    except FileNotFoundError as e:
        print(f"Could not find image for player: {player_name}")
        continue
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
    fig.set_facecolor("#201D1D")

    # create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
    ax6 = fig.add_subplot(gs[1, 2])  # add new subplot

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # apply modifications to all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor("#201D1D")
        ax.axis('off')

    ax1.set_title('Offensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax4.set_title('Ball Carries', color='#e1dbd6', fontsize=16, pad=10)
    ax3.set_title('Defensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax2.set_title('Territory map', color='#e1dbd6', fontsize=16, pad=10)
    ax5.set_title('Pass Map', color='#e1dbd6', fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color='#e1dbd6', fontsize=16, pad=10)


    whovis.plot_players_offensive_actions_opta(ax1, player_name, offensive_actions, color='#1f8e98')
    whovis.plot_carry_player_opta(ax4, player_name, data)

    whovis.plot_players_defensive_actions_opta(ax3, player_name, defensive_actions, color='#1f8e98')

    whovis.plot_player_heatmap(ax2, data, player_name, color='#1f8e98', sd=0)

    whovis.plot_player_passmap_opta(ax5, player_name, data)

    whovis.plot_player_passes_rec_opta(ax6, player_name, passes_df)


    player_logo_path = f'Data/player_image/{player_id}.png'
    club_icon = Image.open(player_logo_path).convert('RGBA')

    logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
    logo_ax.imshow(club_icon)
    logo_ax.set_xticks([])
    logo_ax.set_yticks([])

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    plt.savefig(
        f"figures/playerdashboard{player_name}.png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )




for player_name in fullbacks:
    player_id = get_player_id(player_name)
    try:
        player_logo_path = f'Data/player_image/{player_id}.png'
        club_icon = Image.open(player_logo_path).convert('RGBA')
    except FileNotFoundError as e:
        print(f"Could not find image for player: {player_name}")
        continue
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
    fig.set_facecolor("#201D1D")

    # create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
    ax6 = fig.add_subplot(gs[1, 2])  # add new subplot

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # apply modifications to all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor("#201D1D")
        ax.axis('off')

    ax1.set_title('Offensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax4.set_title('Ball Carries', color='#e1dbd6', fontsize=16, pad=10)
    ax3.set_title('Defensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax2.set_title('Territory map', color='#e1dbd6', fontsize=16, pad=10)
    ax5.set_title('Pass Map', color='#e1dbd6', fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color='#e1dbd6', fontsize=16, pad=10)


    whovis.plot_players_offensive_actions_opta(ax1, player_name, offensive_actions, color='#1f8e98')
    whovis.plot_carry_player_opta(ax4, player_name, data)

    whovis.plot_players_defensive_actions_opta(ax3, player_name, defensive_actions, color='#1f8e98')

    whovis.plot_player_heatmap(ax2, data, player_name, color='#1f8e98', sd=0)

    whovis.plot_player_passmap_opta(ax5, player_name, data)

    whovis.plot_player_passes_rec_opta(ax6, player_name, passes_df)


    player_logo_path = f'Data/player_image/{player_id}.png'
    club_icon = Image.open(player_logo_path).convert('RGBA')

    logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
    logo_ax.imshow(club_icon)
    logo_ax.set_xticks([])
    logo_ax.set_yticks([])

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    plt.savefig(
        f"figures/playerdashboard{player_name}.png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )




for player_name in midfielders:
    player_id = get_player_id(player_name)
    try:
        player_logo_path = f'Data/player_image/{player_id}.png'
        club_icon = Image.open(player_logo_path).convert('RGBA')
    except FileNotFoundError as e:
        print(f"Could not find image for player: {player_name}")
        continue
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
    fig.set_facecolor("#201D1D")

    # create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
    ax6 = fig.add_subplot(gs[1, 2])  # add new subplot

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # apply modifications to all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor("#201D1D")
        ax.axis('off')


    ax1.set_title('Offensive Actions', color ='#e1dbd6'  , fontsize=16, pad=10)
    ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax3.set_title('Defensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)




    whovis.plot_players_offensive_actions_opta(ax1,player_name,offensive_actions,color='#1f8e98')
    whovis.plot_carry_player_opta(ax4,player_name,data)
    whovis.plot_players_defensive_actions_opta(ax3,player_name,defensive_actions,color='#1f8e98')
    whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)
    whovis.plot_player_passmap_opta(ax5,player_name,data)
    whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)



    player_logo_path = f'Data/player_image/{player_id}.png'
    club_icon = Image.open(player_logo_path).convert('RGBA')

    logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
    logo_ax.imshow(club_icon)
    logo_ax.set_xticks([])
    logo_ax.set_yticks([])

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    plt.savefig(
        f"figures/playerdashboard{player_name}.png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )






for player_name in forwards:
    player_id = get_player_id(player_name)
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
    fig.set_facecolor("#201D1D")

    # create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
    ax6 = fig.add_subplot(gs[1, 2])  # add new subplot

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # apply modifications to all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor("#201D1D")
        ax.axis('off')


    ax1.set_title('Shot Map', color ='#e1dbd6'  , fontsize=16, pad=10)
    ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax3.set_title('Offensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)



    fotmobvis.plot_player_shotmap(ax1,Fotmob_matchID,player_name)

    whovis.plot_carry_player_opta(ax4,player_name,data)

    whovis.plot_players_defensive_actions_opta(ax3,player_name,offensive_actions,color='#1f8e98')

    whovis.plot_player_passmap_opta(ax5,player_name,data)

    whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)
    whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)


    player_logo_path = f'Data/player_image/{player_id}.png'
    club_icon = Image.open(player_logo_path).convert('RGBA')

    logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
    logo_ax.imshow(club_icon)
    logo_ax.set_xticks([])
    logo_ax.set_yticks([])

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    plt.savefig(
        f"figures/playerdashboard{player_name}.png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )

for player_name in subs:
    player_id = get_player_id(player_name)
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
    fig.set_facecolor("#201D1D")

    # create subplots using gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
    ax6 = fig.add_subplot(gs[1, 2])  # add new subplot

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # apply modifications to all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor("#201D1D")
        ax.axis('off')


    ax1.set_title('Offensive Actions', color ='#e1dbd6'  , fontsize=16, pad=10)
    ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax3.set_title('Defensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)



    fotmobvis.plot_player_shotmap(ax1,Fotmob_matchID,player_name)

    # whovis.plot_players_offensive_actions_opta(ax1,player_name,offensive_actions,color='#1f8e98')
    whovis.plot_carry_player_opta(ax4,player_name,data)
    whovis.plot_players_defensive_actions_opta(ax3,player_name,defensive_actions,color='#1f8e98')
    whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)
    whovis.plot_player_passmap_opta(ax5,player_name,data)
    whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)



    player_logo_path = f'Data/player_image/{player_id}.png'
    club_icon = Image.open(player_logo_path).convert('RGBA')

    logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
    logo_ax.imshow(club_icon)
    logo_ax.set_xticks([])
    logo_ax.set_yticks([])

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    plt.savefig(
        f"figures/playerdashboard{player_name}.png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )


#%%
