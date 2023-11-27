import requests
from PIL import Image
from io import BytesIO
import os

def check_logo_existence(away_id, home_id):
    logo_folder = 'Data/team_logo/'

    away_logo_path = f'{logo_folder}Fotmob_{away_id}.png'
    home_logo_path = f'{logo_folder}Fotmob_{home_id}.png'

    return os.path.exists(away_logo_path) and os.path.exists(home_logo_path)

def get_and_save_logo(team_id):
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
    club_icon = Image.open(BytesIO(response.content))

    logo_path = f'Data/team_logo/{team_id}.png'
    club_icon.save(logo_path, 'PNG', quality=95, dpi=(300, 300))

def ax_logo(team_id, ax):
    '''
    Plots the logo of the team at a specific axes.
    Args:
        team_id (int): the id of the team according to Fotmob. You can find it in the url of the team page.
        ax (object): the matplotlib axes where we'll draw the image.
    '''
    logo_folder = 'Data/team_logo/'
    logo_path = f'{logo_folder}{team_id}.png'

    if not os.path.exists(logo_path):
        get_and_save_logo(team_id)

    club_icon = Image.open(logo_path)
    ax.imshow(club_icon)
    ax.axis('off')
    return ax


