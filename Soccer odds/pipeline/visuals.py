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

import pandas as pd

from PIL import Image
import urllib
import os

import requests
from PIL import Image
from io import BytesIO

colors = ['#d8f3dc', '#b7e4c7', '#95d5b2', '#74c69d', '#52b788', '#40916c', '#2d6a4f', '#1b4332', '#081c15'
          ]

odd_cm = LinearSegmentedColormap.from_list('ODD', colors, N=25)
cm.register_cmap(name='ODD', cmap=odd_cm)
odd_cm
#%%
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
    team_id = merged_df['home_team_id'].iloc[y]
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

    home_odds = merged_df['home_prob'].iloc[y]
    away_odds = merged_df['away_prob'].iloc[y]
    draw_odds = merged_df['draw_prob'].iloc[y]

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

    cmap = cm.get_cmap('ODD')
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
    s = f"EPL Round {matchweekround} Match Predictions",
    va = "bottom", ha = "left",
    fontsize = 12, color = "#E1D3D3", weight = "bold"
)

fig_text(
    x = .22, y = .92,
    s = "Viz & Model by @andrewlfc_lfc18  | Season 2023/2024",
    va = "bottom", ha = "left",
    fontsize = 8, color = "#E1D3D3")

#------

# Set the league ID and logo path
Fotmob_leagueID = 47
league_logo_path = f'logos/{Fotmob_leagueID}.png'

# Create the league icon axes object
league_icon = fig.add_axes([0.12, .92, .1, .08], frameon=False)

# Load the league logo image and add it to the league icon axes object
league_logo = Image.open(league_logo_path).convert('RGBA')
league_icon.imshow(league_logo)

# Turn off the axis of the league icon axes object
league_icon.axis('off')



plt.savefig(
    f"figures/EPL{matchweekround}forecast.png",
    dpi = 900,
    bbox_inches="tight",
    edgecolor="none",
    transparent = False
)

