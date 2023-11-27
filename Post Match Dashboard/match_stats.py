#%%

import datetime
import json

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from PIL import Image
from highlight_text import fig_text

#%%
today = datetime.date.today()
today = today.strftime('%Y-%m-%d')

#%%
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:Liverpool19@localhost:5432/soccer')


current_competition = 'Premier League'

conn = engine.connect()

#%%

#%%

#%%
if current_competition == 'Premier League':
    shots_query = """SELECT * FROM fotmob_shots_data WHERE ("teamId" = 8650 OR "match_id" IN (SELECT "match_id" FROM fotmob_shots_data WHERE "teamId" = 8650)) AND competition = 'Premier League'"""
    comp_name = 'Premier League'
else:
    shots_query = """SELECT * FROM fotmob_shots_data WHERE  ("teamId" = 8650 OR "match_id" IN (SELECT "match_id" FROM fotmob_shots_data WHERE "teamId" = 8650))"""
    comp_name = 'All Competition'

# Query the database and load data into a DataFrame
data = pd.read_sql(shots_query, conn)

if current_competition == 'Premier League':
    query = """SELECT * FROM opta_event_data WHERE ("teamId" = 26 OR "match_id" IN (SELECT "match_id" FROM opta_event_data WHERE "teamId" = 26)) AND competition = 'Premier League'"""
    comp_name = 'Premier League'
else:
    query = """SELECT * FROM opta_event_data WHERE ("teamId" = 26 OR "match_id" IN (SELECT "match_id" FROM opta_event_data WHERE "teamId" = 26))"""
    comp_name = 'All Competition'


# Query the database and load data into a DataFrame
event_data = pd.read_sql(query, conn)

#%%

event_data = event_data.rename(columns={"match_id":"matchId"})

data = data.rename(columns={"match_id":"matchId"})

match_date = today

Fotmob_matchID = data[data['match_date'] == match_date]['matchId'].iloc[0]

opta_matchID = event_data[event_data['match_date'] == match_date]['matchId'].iloc[0]

#%%


def calculate_match_shots_stats(data,teamId):

    data['situation'] = data['situation'].replace({
        'RegularPlay': 'RegularPlay',
        'FromCorner': 'SetPiece',
        'SetPiece': 'SetPiece',
        'FastBreak': 'RegularPlay',
        'FreeKick': 'SetPiece',
        'ThrowInSetPiece': 'SetPiece',
        'Penalty': 'Penalty'
    })


    liv_data = data[data['teamId']==teamId]


    team_matches = data[data['teamId']==teamId]['matchId'].unique()

    opponents_data = data[(data['matchId'].isin(team_matches)) & (data['teamId']!=teamId)]


    #--- Liverpool
    xG = liv_data.groupby(['matchId','teamId'])['expectedGoalsOnTarget'].sum().reset_index()
    xG = xG.rename(columns={'expectedGoalsOnTarget': 'xGOT_liv'})

    npxG = liv_data[liv_data['situation']!='Penalty'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()

    openplay_xG = liv_data[liv_data['situation']=='RegularPlay'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    setpiece_xG = liv_data[liv_data['situation']=='SetPiece'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    openplay_shots = liv_data[liv_data['situation']=='RegularPlay'].groupby(['matchId', 'teamId']).size().reset_index(name='shotCount')
    openplay_xG = openplay_xG.merge(openplay_shots, on=['matchId', 'teamId'])
    openplay_xG['xG_per_shot'] = openplay_xG['expectedGoals'] / openplay_xG['shotCount']
    openplay_xG = openplay_xG.rename(columns={'expectedGoals': 'openplay_xG_liv','xG_per_shot':'xG_per_shot_Liv'})
    setpiece_xG = setpiece_xG.rename(columns={'expectedGoals': 'setpiece_xG_liv'})
    npxG = npxG.rename(columns={'expectedGoals': 'npxG_liv'})


    #--- Opponents
    npxG_Opp = opponents_data[opponents_data['situation']!='Penalty'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()

    openplay_xG_Opp = opponents_data[opponents_data['situation']=='RegularPlay'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    setpiece_xG_Opp = opponents_data[opponents_data['situation']=='SetPiece'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    openplay_shots_Opp = opponents_data[opponents_data['situation']=='RegularPlay'].groupby(['matchId', 'teamId']).size().reset_index(name='shotCount')
    openplay_xG_Opp = openplay_xG_Opp.merge(openplay_shots_Opp, on=['matchId', 'teamId'])
    openplay_xG_Opp['xG_per_shot'] = openplay_xG_Opp['expectedGoals'] / openplay_xG_Opp['shotCount']

    openplay_xG_Opp = openplay_xG_Opp.rename(columns={'expectedGoals': 'openplay_xG_Opp','xG_per_shot':'OP_xG/shots_Opp'})
    setpiece_xG_Opp = setpiece_xG_Opp.rename(columns={'expectedGoals': 'setpiece_xG_Opp'})
    npxG_Opp = npxG_Opp.rename(columns={'expectedGoals': 'npxG_Opp'})

    # Merge dataframes for Liverpool
    liv_merged = npxG.merge(setpiece_xG, on=['matchId', 'teamId']).merge(openplay_xG, on=['matchId', 'teamId']).merge(xG, on=['matchId', 'teamId'])


    opp_merged_df = npxG_Opp.merge(setpiece_xG_Opp, on=['matchId', 'teamId']).merge(openplay_xG_Opp, on=['matchId', 'teamId'])

    # Add match_date to the dataframes
    liv_merged = liv_merged.merge(data[['matchId', 'match_date']], on=['matchId'])
    opp_merged_df = opp_merged_df.merge(data[['matchId', 'match_date']], on=['matchId'])


    return liv_merged,opp_merged_df
#%%

#%%

#%%
stats =calculate_match_shots_stats(data,8650)[0]

stats_opp = calculate_match_shots_stats(data,8650)[1]
#%%




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
    event_data[col] = event_data[col].astype(bool)


#%%

#%%
def calculate_match_stats(data, teamId):
    liv_data = data[data['teamId'] == teamId]
    team_matches = liv_data['matchId'].unique()
    opp_data = data[(data['matchId'].isin(team_matches)) & (data['teamId'] != teamId)]


    final_third = data[data['x'] >= 60]
    defensive_actions = final_third[final_third['event_type'].isin(
        ['BallRecovery', 'BlockedPass', 'ChallengeWon', 'Clearance', 'Foul', 'Interception', 'TackleWon'])]
    defensive_actions_count = defensive_actions.groupby(['matchId', 'teamId'])['eventId'].count().reset_index(
        name='defensive_actions_count')
    opponent_passes = final_third[final_third['event_type'] == 'Pass']
    opponent_passes_count = opponent_passes.groupby(['matchId', 'teamId'])['eventId'].count().reset_index(
        name='opponent_passes_count')
    ppda_data = pd.merge(defensive_actions_count, opponent_passes_count, on=['matchId', 'teamId'])
    ppda_data['PPDA'] = ppda_data['opponent_passes_count'] / ppda_data['defensive_actions_count']


    # Liverpool stats
    liv_xthreat = liv_data.groupby(['matchId', 'team_name', 'teamId'])['xThreat_gen'].sum().reset_index()
    liv_passes = liv_data[liv_data['event_type'] == 'Pass']
    liv_successful_passes = liv_passes[liv_passes['outcomeType'] == 'Successful'].groupby('matchId').count()
    liv_total_passes = liv_passes.groupby('matchId').count()
    liv_pass_success_rate = (liv_successful_passes['id'] / liv_total_passes['id']).reset_index()
    liv_pass_success_rate.columns = ['matchId', 'pass_success_rate']


    defensive_actions = data[data['event_type'].isin(
        ['BallRecovery', 'BlockedPass', 'ChallengeWon', 'Clearance', 'Foul', 'Interception', 'TackleWon'])]
    defensive_line_height = 100 - defensive_actions.groupby(['matchId'])['endY'].mean()
    liv_defensive_line_height = defensive_line_height.reset_index(name='defensive_line_height')

    liv_merged_df = liv_xthreat.merge(liv_pass_success_rate, on='matchId').merge(liv_defensive_line_height, on='matchId')


    # Opponent stats
    opp_xthreat = opp_data.groupby(['matchId', 'team_name', 'teamId'])['xThreat_gen'].sum().reset_index()
    opp_passes = opp_data[opp_data['event_type'] == 'Pass']
    opp_successful_passes = opp_passes[opp_passes['outcomeType'] == 'Successful'].groupby('matchId').count()
    opp_total_passes = opp_passes.groupby('matchId').count()
    opp_pass_success_rate = (opp_successful_passes['id'] / opp_total_passes['id']).reset_index()
    opp_pass_success_rate.columns = ['matchId', 'pass_success_rate']
    opp_merged_df = opp_xthreat.merge(opp_pass_success_rate, on='matchId')

    # Add match_date to the dataframes
    liv_merged_df = liv_merged_df.merge(data[['matchId', 'match_date']], on=['matchId'])
    opp_merged_df = opp_merged_df.merge(data[['matchId', 'match_date']], on=['matchId'])


    return liv_merged_df,opp_merged_df,ppda_data
#%%
touches = event_data[event_data['isTouch'] == True]
possession_metric = touches.groupby(['matchId', 'teamId']).size() / touches.groupby('matchId').size()
possession_metric = possession_metric.reset_index(name='possession_metric')

#%%
match_stats = calculate_match_stats(data=event_data,teamId=26)[0]
match_stats_opp =calculate_match_stats(data=event_data,teamId=26)[1]
ppda = calculate_match_stats(data=event_data,teamId=26)[2]


#%%

#%%
def get_match_name(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    general = data['general']
    Hteam = general['homeTeam']
    Ateam = general['awayTeam']
    Hteam = Hteam['name']
    Ateam = Ateam['name']
    return Hteam + " " + "vs" + " " + Ateam


match_name = get_match_name(Fotmob_matchID)


def get_match_score(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    match_score = data['header']['status']['scoreStr']
    return match_score


match_score = get_match_score(Fotmob_matchID)
#%%
fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(3, 4), dpi=900)
fig.set_facecolor("#201D1D")
fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.8, wspace=0.2, hspace=0.5)



team_logo_path = f'Data/team_logo/{8650}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


logo_ax = fig.add_axes([0.0, .82, 0.08, 0.08], frameon=False)

logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])

fig_text(
    0.4,
    0.88,
    match_score,
    fontsize=5,
    color="#FCE6E6",
    ha="center",
    va="center",
    # transform=ax.transAxes
)

fig_text(
    0.4,
    0.86,
    match_name,
    fontsize=4,
    color="#FCE6E6",
    ha="center",
    va="center",
    # transform=ax.transAxes
)

fig_text(
    0.4,
    0.84,
    f'Compared to Liverpool\'s {comp_name} Average since the start of the 2023/24 season',
    fontsize=3,
    color="#FCE6E6",
    ha="center",
    va="center",
    # transform=ax.transAxes
)




axs = axs.flatten()
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_facecolor("#212529")
    # ax.set_xlim(x_lower_bound, x_upper_bound)

# Create the scatter plot
df_scatter = pd.DataFrame()
for index, match in enumerate(stats['matchId']):
    df_aux = stats[stats['matchId'] == match]
    # df_aux = df_aux.assign(index)
    df_aux = df_aux.assign(index=index)
    df_scatter = pd.concat([df_scatter, df_aux])
    df_scatter.reset_index(drop=True, inplace=True)

axs[0].set_ylabel('npxG', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[1].set_ylabel('Opp npxG', fontsize=3, color='white', rotation='horizontal', labelpad=16)

axs[2].set_ylabel('SetPiece xG', fontsize=3, color='white', rotation=0, labelpad=16)

axs[3].set_ylabel('Opp SetPiece xG', fontsize=3, color='white', rotation=0, labelpad=16)

axs[4].set_ylabel('Open-play xG', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[5].set_ylabel('Opp Open-play xG', fontsize=3, color='white', rotation='horizontal', labelpad=16)

axs[6].set_ylabel('Open-play xG /Shot', fontsize=3, color='white', rotation=0, labelpad=16)

axs[7].set_ylabel('Opp Open-play xG /Shot', fontsize=3, color='white', rotation=0, labelpad=18)

axs[8].set_ylabel('xThreat', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[9].set_ylabel('Opp xThreat', fontsize=3, color='white', rotation=0, labelpad=16)
axs[10].set_ylabel('PPDA', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[11].set_ylabel('Defensive line height ', fontsize=3, color='white', rotation=0, labelpad=16)
# axs[12].set_ylabel('Field Tilt', fontsize=3, color='white', rotation=0, labelpad=16)
axs[13].set_ylabel('Possession', fontsize=3, color='white', rotation=0, labelpad=16)
axs[14].set_ylabel('Pass Completion %', fontsize=3, color='white', rotation='horizontal', labelpad=20)
axs[15].set_ylabel('Opp Pass Completion %', fontsize=3, color='white', rotation=0, labelpad=20)

# axs[16].set_ylabel('Counter Attacks xG', fontsize=3, color='white', rotation='horizontal', labelpad=20)
# axs[17].set_ylabel('Opp Counter Attacks xG ', fontsize=3, color='white', rotation=0, labelpad=20)

axs[12].set_ylabel('xGOT', fontsize=3, color='white', rotation=0, labelpad=14)

sns.scatterplot(data=stats_opp, x='npxG_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[1])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='npxG_Opp', y=index, c='#660708', edgecolor='k', s=20,
                marker='o', alpha=.88, ax=axs[1])


sns.scatterplot(data=stats, x='openplay_xG_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[4])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='openplay_xG_liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[4])


sns.scatterplot(data=stats_opp, x='openplay_xG_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[5])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='openplay_xG_Opp', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[5])

sns.scatterplot(data=stats, x='setpiece_xG_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[2])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='setpiece_xG_liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[2])


sns.scatterplot(data=stats_opp, x='setpiece_xG_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[3])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='setpiece_xG_Opp', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[3])



sns.scatterplot(data=stats, x='npxG_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[0])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='npxG_liv', y=index, c='#660708', edgecolor='k', s=20,
                marker='o', alpha=.88, ax=axs[0])


sns.scatterplot(data=stats, x='xG_per_shot_Liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[6])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='xG_per_shot_Liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[6])

sns.scatterplot(data=stats_opp, x='OP_xG/shots_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[7])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='OP_xG/shots_Opp', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[7])


sns.scatterplot(data=match_stats_opp[match_stats_opp['teamId']!=26], x='xThreat_gen', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[9])
sns.scatterplot(data=match_stats[match_stats['teamId']==26], x='xThreat_gen', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[8])

sns.scatterplot(data=match_stats_opp[(match_stats_opp['teamId']!=26) & (match_stats_opp['matchId'] == opta_matchID)], x='xThreat_gen', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[9])
sns.scatterplot(data=match_stats[(match_stats['teamId']==26) & (match_stats['matchId'] == opta_matchID)], x='xThreat_gen', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[8])


sns.scatterplot(data=ppda[ppda['teamId']==26], x='PPDA', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[10])

sns.scatterplot(data=ppda[(ppda['teamId']==26) & (ppda['matchId'] == opta_matchID)], x='PPDA', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[10])



sns.scatterplot(data=match_stats[(match_stats['teamId']==26) & (match_stats['matchId'] == opta_matchID)], x='defensive_line_height', y=index, c='#660708', edgecolor='k', s=20, marker='o',zorder=4, alpha=.88,
                ax=axs[11])

sns.scatterplot(data=match_stats[match_stats['teamId']==26], x='defensive_line_height', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[11])

sns.scatterplot(data=possession_metric[(possession_metric['teamId']==26) & (possession_metric['matchId'] == opta_matchID)], x='possession_metric', y=index, c='#660708', edgecolor='k', s=20, marker='o',zorder=4, alpha=.88,
                ax=axs[13])

sns.scatterplot(data=possession_metric[possession_metric['teamId']==26], x='possession_metric', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[13])

sns.scatterplot(data=match_stats[match_stats['teamId']==26], x='pass_success_rate', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[14])

sns.scatterplot(data=match_stats[(match_stats['teamId']==26) & (match_stats['matchId'] == opta_matchID)], x='pass_success_rate', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[14])


sns.scatterplot(data=match_stats_opp[match_stats_opp['teamId']!=26], x='pass_success_rate', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[15])
sns.scatterplot(data=match_stats_opp[(match_stats_opp['teamId']!=26) & (match_stats_opp['matchId'] == opta_matchID)], x='pass_success_rate', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[15])


sns.scatterplot(data=stats, x='xGOT_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[12])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='xGOT_liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[12])


# shots_data.loc[(shots_data['match_date'] == match_date) & (shots_data['teamId'] == 8650), 'match_id'].iloc[0]


# sns.scatterplot(data=stats, x='counterxG_liv', y=index, c='#2a9d8f', edgecolor='#2a9d8f', s=20, marker='o', alpha=.2,
#                 ax=axs[16])
# sns.scatterplot(data=stats[stats['matchId'] == '3901269'], x='counterxG_liv', y=index, c='#ad2831', edgecolor='k',
#                 s=20, marker='o', alpha=.88, ax=axs[16])
#
#
#
# sns.scatterplot(data=stats_opp, x='counterxG_Opp', y=index, c='#2a9d8f', edgecolor='#2a9d8f', s=20, marker='o', alpha=.2,
#                 ax=axs[17])
# sns.scatterplot(data=stats_opp[stats_opp['matchId'] == '3901269'], x='counterxG_Opp', y=index, c='#ad2831', edgecolor='k',
#                 s=20, marker='o', alpha=.88, ax=axs[17])

#
# #-- Mean
#
# axs[0].axvline(stats_mean['npxG_liv']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[1].axvline(stats_opp_mean['npxG_Opp']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#
#
#
# # axs[2].axvline(stats_mean['setpiece_xG_liv']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[3].axvline(stats_opp_mean['setpiece_xG_Opp']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#
#
# axs[4].axvline(stats_mean['openplay_xG_liv']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[5].axvline(stats_opp_mean['openplay_xG_Opp']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#
#
# #
# axs[6].axvline(stats_mean['xG_per_shot_Liv']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[7].axvline(stats_opp_mean['OP_xG/shots_Opp']['mean'], color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#

# axs[8].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[9].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#
# axs[10].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[11].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#
# axs[12].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[13].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#
# axs[14].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
# axs[15].axvline(possession_mean, color='#70798C',linewidth=.6, alpha=.88,linestyle='--')
#


#
# #-- SD
#
# axs[0].set_xlim(mean_value[2] - 2*std_value[1], mean_value[2] + 2*std_value[1])
#
# axs[4].set_xlim(mean_value[4] - 2*std_value[3], mean_value[4] + 2*std_value[3])
#
# # axs[2].set_xlim(mean_value[3] - 1.5*std_value[2], mean_value[3] + 1.5*std_value[2])
# axs[6].set_xlim(mean_value[6] - 2*std_value[5], mean_value[6] + 2*std_value[5])
#
# # axs[2].set_xlim(lower_lim, upper_lim)

fig.savefig(f"figures/match_avgDashboard{today}.png", dpi=900, bbox_inches="tight")
