import requests
import json
import os

def are_all_matches_finished(season_length, match_round, current_round):
    match_ids = []

    # Collect match IDs for the round before the current round
    for i in range(season_length):
        if match_round[i]['round'] == current_round - 1:
            match_ids.append(match_round[i]['id'])

    # Check if any matches exist for the round before the current round
    if not match_ids:
        return False  # No matches for the round before the current round

    # Check if all matches in the round before the current round are finished
    for match_id in match_ids:
        response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
        data = response.content
        data = json.loads(data)

        # Check if the match data is valid
        if 'general' not in data:
            return False  # Invalid match data

        # Check if the match is finished
        if data['general'].get('finished') == 'true':
            return False  # At least one match is not finished

    return True


def is_model_built(league_name):
    model_path = f'model/model{league_name}.pkl'
    return os.path.exists(model_path)
