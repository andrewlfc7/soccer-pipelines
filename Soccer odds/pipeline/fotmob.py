import requests
import json
import pandas as pd

def get_league_fixtures(league_id):
    try:
        if not isinstance(league_id, int):
            raise ValueError("League ID must be an integer")

        url = f'https://www.fotmob.com/api/leagues?id={league_id}&ccode3=USA_MA'

        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        fixtures_data = response.json()

        return fixtures_data

    except (ValueError, requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"An error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def get_matchround_fixtures(season_length,match_round,current_round):
    match_ids = []
    for i in range(season_length):
        if match_round[i]['round'] == current_round:
            match_ids.append(match_round[i]['id'])

    # Create an empty list to hold the dictionaries
    data_list = []

    # Loop over match ids and get match details
    for match_id in match_ids:
        response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
        data = response.content
        data = json.loads(data)

        home_team = data['general']['homeTeam']
        home_team_name = home_team['name']
        home_team_id = home_team['id']

        away_team = data['general']['awayTeam']
        away_team_name = away_team['name']
        away_team_id = away_team['id']

        game_match_id = data['general']['matchId']
        matchRound = data['general']['matchRound']

        # Append the data as a dictionary to the list
        data_list.append({'matchRound':matchRound,'game_match_id': game_match_id, 'home_team_id': home_team_id, 'home_team_name': home_team_name, 'away_team_id': away_team_id, 'away_team_name': away_team_name})

    # Create the DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    return df


def get_latest_comp_shotsdata(season_length=int,league_id=int):
    response = requests.get(f'https://www.fotmob.com/api/leagues?id={league_id}&ccode3=USA_MA&season=2020%2F2021')
    data = json.loads(response.content)
    leagues = data['overview']['leagueOverviewMatches']
    match_ids = []
    not_started_count = 0
    for i in range(season_length):
        if leagues[i]['notStarted']:
            not_started_count += 1
            if not_started_count >= 10:
                break
        else:
            not_started_count = 0
            match_ids.append(leagues[i]['id'])

    match_data = []

    for match_id in match_ids:
        response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')

        data = response.content
        data = json.loads(data)

        matchId = data['general']['matchId']
        matchTimeUTCDate = data['general']['matchTimeUTCDate'][:10]
        teamcolors = data['general']['teamColors']

        home_color = teamcolors['darkMode']['home']
        away_color = teamcolors['darkMode']['away']
        competitions = data['general']['parentLeagueName']
        league_id=data['details']['id']

        homeTeam = data['general']['homeTeam']
        awayTeam = data['general']['awayTeam']

        home_team_id = homeTeam['id']
        away_team_id = awayTeam['id']

        homeTeamName = homeTeam['name']
        awayTeamName = awayTeam['name']

        homeTeam = pd.DataFrame(homeTeam, index=[0])
        awayTeam = pd.DataFrame(awayTeam, index=[0])

        shot_data = data['content']['shotmap']['shots']

        df_shot = pd.DataFrame(shot_data)

        df_shot['matchId'] = matchId
        df_shot['match_date'] = matchTimeUTCDate
        df_shot['competition'] = competitions
        df_shot['league_id'] = league_id




        df_shot['Venue'] = ''
        df_shot['TeamColor'] = ''

        for index, row in df_shot.iterrows():
            if row['teamId'] == home_team_id:
                df_shot.loc[index, 'Venue'] = 'Home'
                df_shot.loc[index, 'TeamName'] = homeTeamName
                df_shot.loc[index, 'TeamColor'] = home_color
            elif row['teamId'] == away_team_id:
                df_shot.loc[index, 'Venue'] = 'Away'
                df_shot.loc[index, 'TeamName'] = awayTeamName
                df_shot.loc[index, 'TeamColor'] = away_color


        def extract_value(d, key):
            return d[key]

        df_shot['onGoalShot_X'] = df_shot['onGoalShot'].apply(extract_value, args=('x',))
        df_shot['onGoalShot_Y'] = df_shot['onGoalShot'].apply(extract_value, args=('y',))
        df_shot['onGoalShot_ZR'] = df_shot['onGoalShot'].apply(extract_value, args=('zoomRatio',))
        df_shot.drop(['onGoalShot'], axis=1, inplace=True)

        match_data.append(df_shot)
        print(f"{match_id} processed")
        print(f"{len(match_data)} matches processed.")


    df_shot = pd.concat(match_data, ignore_index=True)

    return df_shot


def get_season_shots_data(season,league_id):
    response = requests.get(f'https://www.fotmob.com/api/leagues?id={league_id}&ccode3=USA_MA&season={season}')
    data = json.loads(response.content)
    leagues = data['overview']['leagueOverviewMatches']
    season_length = len(data['matches']['allMatches'])

    match_ids = []
    not_started_count = 0

    for i in range(season_length):
        if leagues[i]['notStarted']:
            not_started_count += 1
            if not_started_count >= 10:
                break
        else:
            not_started_count = 0
            match_ids.append(leagues[i]['id'])

    match_data = []

    for match_id in match_ids:
        response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
        data = response.content
        data = json.loads(data)

        matchId = data['general']['matchId']
        matchTimeUTCDate = data['general']['matchTimeUTCDate'][:10]
        teamcolors = data['general']['teamColors']

        home_color = teamcolors['darkMode']['home']
        away_color = teamcolors['darkMode']['away']
        competitions = data['general']['parentLeagueName']

        homeTeam = data['general']['homeTeam']
        awayTeam = data['general']['awayTeam']

        home_team_id = homeTeam['id']
        away_team_id = awayTeam['id']

        homeTeamName = homeTeam['name']
        awayTeamName = awayTeam['name']

        homeTeam = pd.DataFrame(homeTeam, index=[0])
        awayTeam = pd.DataFrame(awayTeam, index=[0])

        shot_data = data['content']['shotmap']['shots']

        df_shot = pd.DataFrame(shot_data)

        df_shot['matchId'] = matchId
        df_shot['match_date'] = matchTimeUTCDate
        df_shot['competition'] = competitions

        df_shot['Venue'] = ''
        df_shot['TeamColor'] = ''

        for index, row in df_shot.iterrows():
            if row['teamId'] == home_team_id:
                df_shot.loc[index, 'Venue'] = 'Home'
                df_shot.loc[index, 'TeamName'] = homeTeamName
                df_shot.loc[index, 'TeamColor'] = home_color
            elif row['teamId'] == away_team_id:
                df_shot.loc[index, 'Venue'] = 'Away'
                df_shot.loc[index, 'TeamName'] = awayTeamName
                df_shot.loc[index, 'TeamColor'] = away_color

        def extract_value(d, key):
            return d[key]

        df_shot['onGoalShot_X'] = df_shot['onGoalShot'].apply(extract_value, args=('x',))
        df_shot['onGoalShot_Y'] = df_shot['onGoalShot'].apply(extract_value, args=('y',))
        df_shot['onGoalShot_ZR'] = df_shot['onGoalShot'].apply(extract_value, args=('zoomRatio',))
        df_shot.drop(['onGoalShot'], axis=1, inplace=True)

        match_data.append(df_shot)
        print(f"{match_id} processed")
        print(f"{len(match_data)} matches processed.")

    df_shot = pd.concat(match_data, ignore_index=True)
    return df_shot

#
# def get_season_matchgoals_data(season, league_id):
#     response = requests.get(f'https://www.fotmob.com/api/leagues?id={league_id}&ccode3=USA_MA&season={season}')
#
#     if response.status_code != 200:
#         print(f"Error fetching league data. Status code: {response.status_code}")
#         return None
#
#     try:
#         data = json.loads(response.content)
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return None
#
#     if 'overview' not in data or 'leagueOverviewMatches' not in data['overview']:
#         print("Error: Unexpected data format")
#         return None
#
#     leagues = data['overview']['leagueOverviewMatches']
#     season_length = len(data['matches']['allMatches'])
#     match_ids = []
#     not_started_count = 0
#
#     for i in range(season_length):
#         if leagues[i]['notStarted']:
#             not_started_count += 1
#             if not_started_count >= 10:
#                 break
#         else:
#             not_started_count = 0
#             match_ids.append(leagues[i]['id'])
#
#     match_data = []
#
#     for match_id in match_ids:
#         response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
#
#         if response.status_code != 200:
#             print(f"Error fetching match details. Match ID: {match_id}, Status code: {response.status_code}")
#             continue
#
#         try:
#             data = json.loads(response.content)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding match details JSON: {e}")
#             continue
#
#         if 'general' not in data:
#             print(f"Error: Unexpected match details format for Match ID: {match_id}")
#             continue
#
#         matchId = data['general']['matchId']
#         matchTimeUTCDate = data['general']['matchTimeUTCDate'][:10]
#         competitions = data['general']['parentLeagueName']
#
#         homeTeam = data['general']['homeTeam']
#         awayTeam = data['general']['awayTeam']
#
#         homeTeamName = homeTeam['name']
#         awayTeamName = awayTeam['name']
#
#         homegoal = data['header']['teams'][0]['score']
#         awaygoal = data['header']['teams'][1]['score']
#
#         LeagueSeason = data['content']['table']['parentLeagueSeason']
#
#         # Create a new DataFrame for each match
#         df_shot = pd.DataFrame({
#             'matchId': [matchId],
#             'match_date': [matchTimeUTCDate],
#             'season':[LeagueSeason],
#             'competition': [competitions],
#             'hometeam': [homeTeamName],
#             'awayteam': [awayTeamName],
#             'homegoal': [homegoal],
#             'awaygoal': [awaygoal]
#         })
#
#         match_data.append(df_shot)
#         print(f"{match_id} processed")
#         print(f"{len(match_data)} matches processed.")
#
#     # Concatenate all DataFrames in the list
#     if match_data:
#         df_result = pd.concat(match_data, ignore_index=True)
#         return df_result
#     else:
#         return None


import requests
import json
import pandas as pd

def get_season_matchgoals_data(season, league_id):
    response = requests.get(f'https://www.fotmob.com/api/leagues?id={league_id}&ccode3=USA_MA&season={season}')

    if response.status_code != 200:
        print(f"Error fetching league data. Status code: {response.status_code}")
        return None

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

    if 'overview' not in data or 'leagueOverviewMatches' not in data['overview']:
        print("Error: Unexpected data format")
        return None

    leagues = data['overview']['leagueOverviewMatches']
    season_length = len(data['matches']['allMatches'])
    match_ids = []
    not_started_count = 0

    for i in range(season_length):
        if leagues[i]['notStarted']:
            not_started_count += 1
            if not_started_count >= 10:
                break
        else:
            not_started_count = 0
            match_ids.append(leagues[i]['id'])

    match_data = []

    # Batch requests for match details
    match_details = []
    for match_id in match_ids:
        response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')

        if response.status_code != 200:
            print(f"Error fetching match details. Match ID: {match_id}, Status code: {response.status_code}")
            continue

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"Error decoding match details JSON: {e}")
            continue

        if 'general' not in data:
            print(f"Error: Unexpected match details format for Match ID: {match_id}")
            continue

        match_details.append({
            'matchId': data['general']['matchId'],
            'matchTimeUTCDate': data['general']['matchTimeUTCDate'][:10],
            'competitions': data['general']['parentLeagueName'],
            'homeTeamName': data['general']['homeTeam']['name'],
            'awayTeamName': data['general']['awayTeam']['name'],
            'homegoal': data['header']['teams'][0]['score'],
            'awaygoal': data['header']['teams'][1]['score'],
            'LeagueSeason': data['content']['table']['parentLeagueSeason']
        })

        print(f"{match_id} processed")
        print(f"{len(match_details)} matches processed.")

    # Create DataFrame from match_details
    if match_details:
        df_result = pd.DataFrame(match_details)
        return df_result
    else:
        return None
