#%%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib import rcParams
from highlight_text import fig_text, ax_text
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import requests
from io import BytesIO

import pandas as pd

from PIL import Image
import urllib
import os
import numpy as np
import seaborn as sns

import requests
from PIL import Image
from io import BytesIO

from pipeline.sim import predict


def matchround_forecast(df,league,fotmob_leagueid,cmap):

    fig = plt.figure(figsize=(6, 6), dpi=600)

    ax = plt.subplot(111, facecolor = "#201D1D")
    fig.set_facecolor('#201D1D')

    ncols = 4
    nrows = df.shape[0]

    matchweekround =df.matchRound.iloc[0]


    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')



    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # --- Define URL and helper functions to add logos --------------------------------

    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # Native data to normalized data coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    # -- Add team logos

    for y in range(nrows):
        # - Home logo
        x = 0.25
        team_id = df['home_team_id'].iloc[y]
        ax_coords = DC_to_NFC([x, y + .25])
        logo_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.1, 0.04], anchor="W")
        response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
        club_icon = Image.open(BytesIO(response.content))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")
        # - Away logo
        x = .3
        team_id = df['away_team_id'].iloc[y]
        ax_coords = DC_to_NFC([x + .5, y + .25])
        logo_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.1, 0.04], anchor="E")
        response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
        club_icon = Image.open(BytesIO(response.content))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")

        home_odds = df['home_prob'].iloc[y]
        away_odds = df['away_prob'].iloc[y]
        draw_odds = df['draw_prob'].iloc[y]

        label_h_ = f'{home_odds:.0%}'
        label_a_ = f'{away_odds:.0%}'
        label_d_ = f'{draw_odds:.0%}'

        # -- Home odds
        x = 1.75
        text_ = ax.annotate(
            xy=(x,y+0.5),
            text=label_h_,
            ha='center',
            va='center',
            size=10,
            weight='bold'
        )
        text_.set_path_effects(
            [path_effects.Stroke(linewidth=1.75, foreground="white"), path_effects.Normal()]
        )
        # -- Draw Odds
        x = 2.75
        text_ = ax.annotate(
            xy=(x,y+0.5),
            text=label_d_,
            ha='center',
            va='center',
            size=10,
            weight='bold'
        )
        text_.set_path_effects(
            [path_effects.Stroke(linewidth=1.75, foreground="white"), path_effects.Normal()]
        )

        # -- Away odds
        x = 3.75
        text_ = ax.annotate(
            xy=(x,y+0.5),
            text=label_a_,
            ha='center',
            va='center',
            size=10,
            weight='bold'

        )
        text_.set_path_effects(
            [path_effects.Stroke(linewidth=1.75, foreground="white"), path_effects.Normal()]
        )

        cmap = cm.get_cmap(cmap)
        # -----------------------------------------
        # -- Adding the colors
        x = 2
        ax.fill_between(
            x=[(x - 0.5), (x + .35)],
            y1=y,
            y2=y + 1,
            color=cmap(home_odds/away_odds - draw_odds),
            zorder=2,
            ec="None",
            alpha=0.75
        )
        x = 2.75
        ax.fill_between(
            x=[(x - .4), (x + .4)],
            y1=y,
            y2=y + 1,
            color=cmap(draw_odds/away_odds - home_odds),
            zorder=2,
            ec="None",
            alpha=0.75
        )
        x = 3.75
        ax.fill_between(
            x=[(x - .6), (x + .6)],
            y1=y,
            y2=y + 1,
            color=cmap(away_odds/home_odds - draw_odds),
            zorder=2,
            ec="None",
            alpha=0.75
        )



    # ----------------------------------------------------------------
    # - Column titles

    ax.text(.75, nrows + .1, "MATCH", weight="bold", ha="center",color = '#E1D3D3', size=8)

    ax.text(1.85, nrows + .1, "Home Win (%)", weight="bold", ha="center", color = '#E1D3D3',size=8)
    ax.text(2.75, nrows + .1, "Draw (%)", weight="bold", ha="center", color = '#E1D3D3',size=8)
    ax.text(3.65, nrows + .1, "Away Win (%)", weight="bold", ha="center", color = '#E1D3D3',size=8)

    # Table borders
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
    for x in range(nrows):
        if x == 0:
            continue
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3, marker='')


    fig_text(
        x = .22, y = .96,
        s = f"{league} Round {matchweekround} Match Predictions",
        va = "bottom", ha = "left",
        fontsize = 12, color = "#E1D3D3", weight = "bold"
    )

    fig_text(
        x = .22, y = .92,
        s = "Viz & Model by @andrewlfc_lfc18  | Season 2023/2024",
        va = "bottom", ha = "left",
        fontsize = 8, color = "#E1D3D3")

    #------

    league_logo_path = f'logos/{fotmob_leagueid}.png'

    # Create the league icon axes object
    league_icon = fig.add_axes([0.12, .92, .1, .08], frameon=False)

    # Load the league logo image and add it to the league icon axes object
    league_logo = Image.open(league_logo_path).convert('RGBA')
    league_icon.imshow(league_logo)

    # Turn off the axis of the league icon axes object
    league_icon.axis('off')

    return plt




def simulate_match_matrix(ax, matchId, upcoming_match_data, model_params, max_goals=10):
    # Filter upcoming match data to get only home and away team ids for the given match
    upcoming_match_data = upcoming_match_data[upcoming_match_data['matchId'] == matchId]

    home_team_name = upcoming_match_data['home_team_name'].iloc[0]
    away_team_name = upcoming_match_data['away_team_name'].iloc[0]

    matrix = predict(model_params, home_team_name, away_team_name, max_goals)

    mask = matrix < 0.05
    # Create the plot
    sns.heatmap(matrix, cmap='Pastel2', ax=ax, square=True, annot=True, fmt='.1f', linewidths=.25, linestyle='--', linecolor='#343a40', mask=mask, cbar=False)
    ax.set_xlabel(away_team_name, fontsize=8, color='#E1D3D3')
    ax.set_ylabel(home_team_name, fontsize=8, color='#E1D3D3')
    ax.tick_params(axis='x', colors='#E1D3D3')
    ax.tick_params(axis='y', colors='#E1D3D3')

    ax.set_ylim(0, max_goals+1)

    return ax




def plot_match_probabilities(ax,model_params,matchId, upcoming_match_data, k=10,max_goals=10):
    '''
    Performs k simulations on a match, and returns the probabilites of a win, loss, draw.
    '''
    # Count the number of occurrences
    home = 0
    draw = 0
    away = 0

    # Get the teams
    upcoming_match_data = upcoming_match_data[upcoming_match_data['matchId']==matchId]
    # home_team_id = upcoming_match_data['home_team_id'].iloc[0]
    # away_team_id = upcoming_match_data['away_team_id'].iloc[0]
    Fotmob_home_name = upcoming_match_data['home_team_name'].iloc[0]
    Fotmob_away_name = upcoming_match_data['away_team_name'].iloc[0]


    for i in range(k):
        matrix = predict(model_params, Fotmob_home_name, Fotmob_away_name,max_goals)
        home_goals_probs, away_goals_probs = np.sum(matrix, axis=1), np.sum(matrix, axis=0)
        # Normalize probabilities
        home_goals_probs /= np.sum(home_goals_probs)
        away_goals_probs /= np.sum(away_goals_probs)
        home_goals = np.argmax(np.random.multinomial(1, home_goals_probs))
        away_goals = np.argmax(np.random.multinomial(1, away_goals_probs))
        if home_goals > away_goals:
            home += 1
        elif home_goals < away_goals:
            away += 1
        else:
            draw += 1

    home_prob = home/k
    draw_prob = draw/k
    away_prob = away/k

    team_probs = {Fotmob_home_name: home_prob, 'Draw': draw_prob, Fotmob_away_name: away_prob}
    labels = [Fotmob_home_name, 'Draw', Fotmob_away_name]

    colors = ['#fcbf49', '#ef476f', '#2a9d8f']

    # Create the plot
    ax.barh(labels, [home_prob, draw_prob, away_prob], color=colors)

    # fig.set_facecolor("#201D1D")

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

    # Add the probability percentages to the bars
    for i, v in enumerate([home_prob, draw_prob, away_prob]):
        ax.text(v + 0.01, i, str(round(v*100, 2))+'%', color='#E1D3D3', fontweight='bold')

    # Set the plot title and axis labels
    ax.set_title(f'{Fotmob_home_name} Vs {Fotmob_away_name} Outcome Probabilities',fontsize=8,
                 color="#E1D3D3")

    ax.set_xlabel('Probability',fontsize=6,
                  color="#E1D3D3")

    # Create legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles, team_probs.keys(), loc='best')

    return ax




def get_team_logo(team_id):
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
    return Image.open(BytesIO(response.content)).convert('RGBA')

def eos_distribution_v1(simulated_tables, league_name):
    n_teams = simulated_tables['team_id'].unique()

    # Create subplots with specified number of rows and columns
    fig, axs = plt.subplots(nrows=len(n_teams), ncols=1, figsize=(6, 10), dpi=600)

    # Set background color and adjust subplot spacing
    fig.set_facecolor("#201D1D")
    fig.subplots_adjust(left=0.2, right=1, bottom=.14, top=.9, wspace=.4, hspace=.1)

    # Flatten axs array
    axs = axs.flatten()

    overall_point_min = simulated_tables['points'].min()
    overall_point_max = simulated_tables['points'].max()

    # Iterate through subplots
    for i, (ax, team_id) in enumerate(zip(axs, n_teams)):
        team_min = simulated_tables[simulated_tables['team_id'] == team_id]['points'].min()
        team_max = simulated_tables[simulated_tables['team_id'] == team_id]['points'].max()

        # Remove ticks, labels, and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)

        # Remove spines, except for the left one
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)

        # Set background color
        ax.set_facecolor("#212529")

        # Plot KDE for each team
        sns.kdeplot(
            data=simulated_tables[simulated_tables['team_id'] == team_id],
            x='points',
            common_norm=False,
            fill=True,
            color='#2a9d8f',
            alpha=0.6,
            gridsize=1000,
            ax=ax
        )
        ax.set_xlim(team_min, team_max)


    # Add team logos
        logo_ax = fig.add_axes([0.14, ax.get_position().y0, 0.04, 0.04], frame_on=False)
        logo_ax.imshow(get_team_logo(team_id))
        logo_ax.axis("off")


    # Set common x-axis limit for all teams
    for ax in axs:
        ax.set_xlim(overall_point_min, overall_point_max)


    # Add figure text
    fig.text(
        x=.22, y=.94,
        s=f"{league_name} End of Season Forecast Points Distribution",
        va="bottom", ha="left",
        fontsize=10, color="#E1D3D3", weight="bold"
    )

    fig.text(
        x=.22, y=.92,
        s="Viz & Model by @andrewlfc_lfc18  | Season 2023/2024",
        va="bottom", ha="left",
        fontsize=8, color="#E1D3D3")

    # Set the league ID and logo path
    Fotmob_leagueID = 47
    league_logo_path = f'logos/{Fotmob_leagueID}.png'

    # Create the league icon axes object
    league_icon = fig.add_axes([0.08, .90, .1, .08], frameon=False)

    # Load the league logo image and add it to the league icon axes object
    league_logo = Image.open(league_logo_path).convert('RGBA')
    league_icon.imshow(league_logo)

    # Turn off the axis of the league icon axes object
    league_icon.axis('off')

    return plt


def xpt_table(xPoints_table,league_name):
    fig = plt.figure(figsize=(3.4, 4.4), dpi=400)

    ax = plt.subplot(111, facecolor="#201D1D")
    fig.set_facecolor('#201D1D')

    ncols = 11
    nrows = xPoints_table.shape[0]

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # --- Define URL and helper functions to add logos --------------------------------

    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # Native data to normalized data coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    # -- Add team logos and points
    for y in reversed(range(nrows)):
        # - Home logo
        x = 0.20
        team_id = xPoints_table['id'].iloc[y]
        ax_coords = DC_to_NFC([x , nrows - y -1.25])
        logo_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.036, 0.06], anchor="C")
        response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
        club_icon = Image.open(BytesIO(response.content))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")



        matches = xPoints_table['played'].iloc[y]
        x_text = 1
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(matches),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )
        points = xPoints_table['pts'].iloc[y]

        x_text = 1.8
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(points),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        wins = xPoints_table['wins'].iloc[y]
        x_text = 2.6
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(wins),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )


        draws = xPoints_table['draws'].iloc[y]
        x_text = 3.4
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(draws),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )



        loss = xPoints_table['losses'].iloc[y]
        x_text = 4.2
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(loss),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        gf = xPoints_table['GF'].iloc[y]
        x_text = 4.8
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(gf),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )


        ga = xPoints_table['GA'].iloc[y]
        x_text = 5.4
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(ga),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )
        ppr = round(xPoints_table['ppg'].iloc[y],2)
        x_text = 6.2
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(ppr),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        gf_90 = round(xPoints_table['gf_per90'].iloc[y],2)
        x_text = 7.2
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(gf_90),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        ga_90 = round(xPoints_table['ga_per90'].iloc[y],2)
        x_text = 8
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(ga_90),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        xpoints = round(xPoints_table['points'].iloc[y],1)
        x_text = 8.8
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(xpoints),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        xG = round(xPoints_table['expectedGoals'].iloc[y],1)
        x_text = 9.4
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(xG),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )

        xGA = round(xPoints_table['ExpectedGoalsAgainst'].iloc[y],1)
        x_text = 10.2
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(xGA),
            ha='left',
            va='center',
            size=4,
            weight='bold',
            c='white'
        )


        # cmap = cm.get_cmap('xpts_preformance')
        # # -----------------------------------------
        # # -- Adding the colors
        # x = 6.8
        # ax.fill_between(
        #     x=[(x - 0.5), (x + .4)],
        #     y1=y,
        #     y2=y + 1,
        #     color=cmap((points/ xpoints - xpoints/ points)/1),
        #     zorder=2,
        #     ec="None",
        #     alpha=0.75
        # )
        # x = 8.65
        # ax.fill_between(
        #     x=[(x - .5), (x + .4)],
        #     y1=y,
        #     y2=y + 1,
        #     color=cmap((gf-xG +1)/2),
        #     zorder=2,
        #     ec="None",
        #     alpha=0.75
        # )
        # x = 9.65
        # ax.fill_between(
        #     x=[(x - .5), (x + .4)],
        #     y1=y,
        #     y2=y + 1,
        #     color=cmap((ga - xGA+ 1)/2),
        #     zorder=2,
        #     ec="None",
        #     alpha=0.75
        # )




    # ----------------------------------------------------------------
    # - Column titles
    ax.text(1.2, nrows + .1, "Played", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(2, nrows + .1, "Points", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(2.7, nrows + .1, "Wins", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(3.4, nrows + .1, "Draws", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(4.2, nrows + .1, "Losses", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(4.9, nrows + .1, "GF", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(5.5, nrows + .1, "GA", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(6.4, nrows + .1, "Points/90", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(7.4, nrows + .1, "GF/90", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(8.2, nrows + .1, "GA/90", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(8.9, nrows + .1, "xPoint", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(9.6, nrows + .1, "xG", weight="bold", ha="center", color = '#E1D3D3',size=3)
    ax.text(10.4, nrows + .1, "xGA", weight="bold", ha="center", color = '#E1D3D3',size=3)




    # Table borders
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)

    for x in range(nrows):
        if x == 0:
            continue
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3, marker='')

    fig_text(
        x = .20, y = .94,
        s = f"{league_name} Expected Points Table",
        va = "bottom", ha = "left",
        fontsize = 5, color = "#E1D3D3",
    )

    fig_text(
        x = .20, y = .91,
        s = "Viz & Model by @andrewlfc_lfc18  | Season 2023/2024",
        va = "bottom", ha = "left",
        fontsize = 4, color = "#E1D3D3")

    #------

    # Set the league ID and logo path
    Fotmob_leagueID = 47
    league_logo_path = f'logos/{Fotmob_leagueID}.png'

    # Create the league icon axes object
    league_icon = fig.add_axes([0.12, .92, .05, .04], frameon=False)

    # Load the league logo image and add it to the league icon axes object
    league_logo = Image.open(league_logo_path).convert('RGBA')
    league_icon.imshow(league_logo)

    # Turn off the axis of the league icon axes object
    league_icon.axis('off')


    return plt

def plot_eos_table_v1(simulated_tables,league_name):
    fig = plt.figure(figsize=(4, 5), dpi=600)

    ax = plt.subplot(111, facecolor="#201D1D")
    fig.set_facecolor('#201D1D')

    ncols = 7
    nrows = simulated_tables.shape[0]

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    current_datetime = datetime.now()

    # Extract the date part
    current_date = current_datetime.date()
    date = current_date.strftime("%Y-%m-%d")

    # --- Define URL and helper functions to add logos --------------------------------

    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # Native data to normalized data coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    # -- Add team logos and points
    for y in reversed(range(nrows)):
        # - Home logo
        x = 0.1
        team_id = simulated_tables['team_id'].iloc[y]
        ax_coords = DC_to_NFC([x , nrows - y -1.25])
        logo_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.034, 0.06], anchor="C")
        response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
        club_icon = Image.open(BytesIO(response.content))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")


        matches = simulated_tables['played'].iloc[y]
        x_text = .6
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(matches),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )
        points = simulated_tables['points'].iloc[y]
        x_text = 1.2
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(points),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )
        wins = simulated_tables['w'].iloc[y]
        x_text = 1.75
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(wins),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )


        draws = simulated_tables['d'].iloc[y]
        x_text = 2.30
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(draws),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )



        loss = simulated_tables['l'].iloc[y]
        x_text = 2.8
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(loss),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )

        gf = simulated_tables['gf'].iloc[y]
        x_text = 3.3
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(gf),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )


        ga = simulated_tables['ga'].iloc[y]
        x_text = 3.85
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(ga),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )

        gd = round(gf-ga)
        x_text = 4.3
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(gd),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )

        ppg = round(simulated_tables['ppg'].iloc[y],2)
        x_text = 4.75
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(ppg),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )

        gf_per90 = round(simulated_tables['gf_per90'].iloc[y],2)
        x_text = 5.55
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(gf_per90),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )

        ga_per90 = round(simulated_tables['ga_per90'].iloc[y],2)
        x_text = 6.4
        y_text = nrows - y - 0.5
        text_ = ax.annotate(
            xy=(x_text, y_text),
            text=str(ga_per90),
            ha='left',
            va='center',
            size=6.25,
            weight='normal',
            c='white'
        )


    # ----------------------------------------------------------------
    # - Column titles
    ax.text(.70, nrows + .1, "Matches", weight="bold", ha="center", color = '#E1D3D3',size=4)

    ax.text(1.35, nrows + .1, "Points", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(1.90, nrows + .1, "Wins", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(2.4, nrows + .1, "Draws", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(2.9, nrows + .1, "Losses", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(3.4, nrows + .1, "GF", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(3.9, nrows + .1, "GA", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(4.4, nrows + .1, "GD", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(5.0, nrows + .1, "Pts/90", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(5.8, nrows + .1, "GF/90", weight="bold", ha="center", color = '#E1D3D3',size=4)
    ax.text(6.6, nrows + .1, "GA/90", weight="bold", ha="center", color = '#E1D3D3',size=4)

    # Table borders
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)

    for x in range(nrows):
        if x == 0:
            continue
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3, marker='')

    fig_text(
        x = .18, y = .94,
        s = f"{league_name} EOS Simulation",
        va = "bottom", ha = "left",
        fontsize = 7, color = "#E1D3D3",
    )

    fig_text(
        x = .18, y = .92,
        s = f"Viz by @andrewlfc_lfc18  | Season 2023/2024  | Simulated on {date} | 1000 Simulations" ,
        va = "bottom", ha = "left",
        fontsize = 4.25, color = "#E1D3D3")

    #------

    # Set the league ID and logo path
    Fotmob_leagueID = 47
    league_logo_path = f'logos/{Fotmob_leagueID}.png'

    # Create the league icon axes object
    league_icon = fig.add_axes([0.12, .92, .06, .04], frameon=False)

    # Load the league logo image and add it to the league icon axes object
    league_logo = Image.open(league_logo_path).convert('RGBA')
    league_icon.imshow(league_logo)

    # Turn off the axis of the league icon axes object
    league_icon.axis('off')

    return plt

def plot_finishing_position_distribution(position_prob, league_name):
    fig = plt.figure(figsize=(10, 8), dpi=900)
    ax = plt.subplot(111, facecolor="#201D1D")
    fig.set_facecolor('#201D1D')

    ncols = 21
    nrows = position_prob.shape[0]

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # --- Define URL and helper functions to add logos --------------------------------

    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # Native data to normalized data coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    for y in reversed(range(nrows)):
        # - Home logo
        x = 0.06
        team_id = position_prob['team_id'].iloc[y]
        ax_coords = DC_to_NFC([x, nrows - y - 1.25])
        logo_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.02, 0.06], anchor="C")
        response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
        club_icon = Image.open(BytesIO(response.content))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")

    # -- Loop to add columns 1 to 20
    for col in range(1, 21):
        for y in reversed(range(nrows)):
            x_text = 0.2 + col
            y_text = nrows - y - 0.5
            text_ = ax.annotate(
                xy=(x_text, y_text),
                text="{:.1f}".format(position_prob[col].iloc[y]),
                ha='left',
                va='center',
                size=7.25,
                weight='normal',
                c='white'
            )
            # Add vertical lines between columns
            if col not in [ 1, 3,5,6,7,8,9,11,12,13,14,20]:
                ax.vlines(x_text - 0.25, 0, nrows, colors='gray', linestyles=':')

    # -- Loop to add column titles dynamically
    for col in range(1, 21):
        ordinal_title = 'th' if 11 <= col <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(col % 10, 'th')
        ax.text(0.4 + col, nrows + 0.1, f"{col}{ordinal_title}", weight="bold", ha="center", color='#E1D3D3', size=6.25)

    # Table borders
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)

    for x in range(nrows):
        if x == 0:
            continue
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3, marker='')

    fig_text(
        x=.16, y=.92,
        s=f"{league_name} Simulated EOS Finishing Position Distribution | Season 2023/24 | 1000 Simulation \nData From :Fotmob",
        va="bottom", ha="left",
        fontsize=8, color="#E1D3D3",
    )

    # Set the league ID and logo path
    Fotmob_leagueID = 47
    league_logo_path = f'logos/{Fotmob_leagueID}.png'

    # Create the league icon axes object
    league_icon = fig.add_axes([0.10, .90, .05, .08], frameon=False)

    # Load the league logo image and add it to the league icon axes object
    league_logo = Image.open(league_logo_path).convert('RGBA')
    league_icon.imshow(league_logo)

    # Turn off the axis of the league icon axes object
    league_icon.axis('off')

    return plt
