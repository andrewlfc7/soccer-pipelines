import numpy as np
from scipy.stats import poisson


def rho_correction(goals_home, goals_away, home_exp, away_exp, rho):
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0

def predict(params, home_team, away_team, max_goals=6):
    home_attack = params["attack_" + home_team]
    home_defence = params["defence_" + home_team]
    away_attack = params["attack_" + away_team]
    away_defence = params["defence_" + away_team]
    home_advantage = params["home_adv"]
    rho = params["rho"]

    team_avgs = [
        np.exp(home_attack + away_defence + home_advantage),
        np.exp(away_attack + home_defence)
    ]

    team_pred = [
        [poisson.pmf(i, team_avg) for i in range(max_goals + 1)] for team_avg in team_avgs
    ]

    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

    correction_matrix = np.array([
        [rho_correction(home_goals, away_goals, team_avgs[0], team_avgs[1], rho) for away_goals in range(2)]
        for home_goals in range(2)
    ])

    output_matrix[:2, :2] = output_matrix[:2, :2] * correction_matrix

    # Normalize to percentages
    total_probability = np.sum(output_matrix)
    output_matrix_percentage = (output_matrix / total_probability) * 100

    return output_matrix_percentage


def simulate_match(matchId, model_params,upcoming_match_data, k=int):
    """
    Performs k simulations on a match, and returns the average goals and probability of different scorelines.
    """

    # Extract match data
    upcoming_match_data = upcoming_match_data[upcoming_match_data['matchId'] == matchId]
    home_team_name = upcoming_match_data['home_team_name'].iloc[0]
    away_team_name = upcoming_match_data['away_team_name'].iloc[0]

    # Initialize counters for goals
    home_goals = 0
    away_goals = 0

    # Initialize probability variable
    total_probability = 0

    # Perform k simulations
    for _ in range(k):

        # Simulate a match
        h, a, prob = simulate_single_match(model_params, home_team_name, away_team_name)

        # Update goal counters
        home_goals += h
        away_goals += a

        # Update total probability
        total_probability += prob

    # Calculate average goals
    avg_home_goals = round(home_goals / k)
    avg_away_goals = round(away_goals / k)


    # Calculate average probability
    avg_probability = total_probability / k

    return {'home_goals': avg_home_goals,'home_team_name':home_team_name ,'away_goals': avg_away_goals,'away_team_name':away_team_name ,'probability': avg_probability, 'matchId': matchId}

def simulate_single_match(params, home_team, away_team):
    """
    Simulates a single match and returns the home and away goals and probability.
    """
    goal = 10  # Specify the maximum number of goals for the simulation

    home_attack = params["attack_" + home_team]
    home_defence = params["defence_" + home_team]
    away_attack = params["attack_" + away_team]
    away_defence = params["defence_" + away_team]
    home_advantage = params["home_adv"]
    rho = params["rho"]

    home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
    away_goal_expectation = np.exp(away_attack + home_defence)

    home_poisson = poisson(home_goal_expectation)
    away_poisson = poisson(away_goal_expectation)

    # Sample from the Poisson distributions
    home_goal = int(home_poisson.rvs())  # Convert to integer
    away_goal = int(away_poisson.rvs())  # Convert to integer

    # Ensure goals are non-negative
    home_goal = max(0, home_goal)
    away_goal = max(0, away_goal)

    home_probs = home_poisson.pmf(range(goal))
    away_probs = away_poisson.pmf(range(goal))

    m = np.outer(home_probs, away_probs)

    m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
    m[0, 1] *= 1 + home_goal_expectation * rho
    m[1, 0] *= 1 + away_goal_expectation * rho
    m[1, 1] *= 1 - rho

    prob = np.sum(np.tril(m, -1))  # Sum of probabilities of different scorelines

    return home_goal, away_goal, prob


def iterate_k_simulations_on_match_id_goals(matchId, shot_df,model_params, k=10):
    '''
    Performs k simulations on a match, and returns the probabilities of a win, loss, draw.
    '''
    shots = shot_df[shot_df['matchId'] == matchId]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Count the number of occurrences
    home_win = 0
    draw = 0
    away_win = 0

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
        away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])

        # Update outcome counters
        if home_goals == away_goals:
            draw += 1
        elif home_goals > away_goals:
            home_win += 1
        else:
            away_win += 1

    home_prob = home_win / k
    draw_prob = draw / k
    away_prob = away_win / k

    return {'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob, 'matchId': matchId}


import pandas as pd
from tqdm import tqdm

def simulate_goals(expected_goals, attack_param):
    """
    Simulate goals based on the expected goal values using Poisson distribution.
    """
    goals = 0
    if expected_goals.shape[0] > 0:
        for shot in expected_goals:
            goals += poisson.rvs(mu=attack_param * shot)
    return max(0, goals)


def simulate_match_on_shots_xg(matchId, shot_df, params, k):
    """
    This function takes a match ID and simulates an outcome based on the shots
    taken by each team, using attack parameters from the model.

    Args:
    - matchId (int): The ID of the match.
    - shot_df (DataFrame): DataFrame containing shot data for the match.
    - params (dict): Dictionary containing model parameters.
    - k (int): Number of simulations to perform.

    Returns:
    - list: List of dictionaries containing simulated results and probabilities for each simulation.
    """

    shots = shot_df[shot_df['matchId'] == matchId]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Store results and probabilities for each simulation
    simulation_results = []

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), params["attack_" + home_team_name])
        away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), params["attack_" + away_team_name])

        # Update outcome counters
        if home_goals == away_goals:
            outcome = 'draw'
        elif home_goals > away_goals:
            outcome = 'home_win'
        else:
            outcome = 'away_win'

        # Store simulation results and probabilities
        simulation_results.append({'home_goals': home_goals, 'away_goals': away_goals, 'outcome': outcome})

    # Calculate probabilities
    home_prob = sum(1 for result in simulation_results if result['outcome'] == 'home_win') / k
    draw_prob = sum(1 for result in simulation_results if result['outcome'] == 'draw') / k
    away_prob = sum(1 for result in simulation_results if result['outcome'] == 'away_win') / k

    # Simulate final goals for home and away teams (using the first simulation)
    home_goals_final = simulate_goals(shots_home['expectedGoals'], params["attack_" + home_team_name])
    away_goals_final = simulate_goals(shots_away['expectedGoals'], params["attack_" + away_team_name])

    return {'home_goals': home_goals_final, 'home_team_name':home_team_name, 'away_goals': away_goals_final,'away_team_name':away_team_name , 'home_prob': home_prob,
            'draw_prob': draw_prob, 'away_prob': away_prob, 'matchId': matchId}, simulation_results


def calculate_table(df):
    home_teams = []
    away_teams = []
    home_goals_for = []
    home_goals_against = []
    home_points = []
    away_goals_for = []
    away_goals_against = []
    away_points = []

    for matchId in df['matchId'].unique():
        match_df = df[df['matchId'] == matchId]

        home_team_name = match_df['home_team_name'].iloc[0]
        away_team_name = match_df['away_team_name'].iloc[0]

        home_goals = match_df['home_goals'].iloc[0]
        away_goals = match_df['away_goals'].iloc[0]

        home_team_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_team_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0

        home_teams.append(home_team_name)
        away_teams.append(away_team_name)
        home_goals_for.append(home_goals)
        home_goals_against.append(away_goals)
        home_points.append(home_team_points)
        away_goals_for.append(away_goals)
        away_goals_against.append(home_goals)
        away_points.append(away_team_points)

    # Create DataFrames for home and away teams
    home_df = pd.DataFrame({'team': home_teams, 'gf': home_goals_for, 'ga': home_goals_against, 'points': home_points})
    away_df = pd.DataFrame({'team': away_teams, 'gf': away_goals_for, 'ga': away_goals_against, 'points': away_points})

    # Calculate other statistics for home and away teams
    home_stats = home_df.groupby('team').agg(
        w=('points', lambda x: (x == 3).sum()),
        d=('points', lambda x: (x == 1).sum()),
        l=('points', lambda x: (x == 0).sum()),
        gf=('gf', 'sum'),
        ga=('ga', 'sum')
    ).reset_index()

    away_stats = away_df.groupby('team').agg(
        w=('points', lambda x: (x == 3).sum()),
        d=('points', lambda x: (x == 1).sum()),
        l=('points', lambda x: (x == 0).sum()),
        gf=('gf', 'sum'),
        ga=('ga', 'sum')
    ).reset_index()

    # Combine home and away statistics
    stats_df = pd.DataFrame({
        'team': home_stats['team'],
        'w': home_stats['w'] + away_stats['w'],
        'd': home_stats['d'] + away_stats['d'],
        'l': home_stats['l'] + away_stats['l'],
        'gf': home_stats['gf'] + away_stats['gf'],
        'ga': home_stats['ga'] + away_stats['ga']
    })

    # Calculate goal difference and points
    stats_df['gd'] = stats_df['gf'] - stats_df['ga']
    stats_df['points'] = 3 * stats_df['w'] + stats_df['d']

    # Sort the DataFrame
    stats_df = stats_df.sort_values(by=['points', 'gd', 'gf'], ascending=[False, False, False])

    # Add position column
    stats_df['position'] = range(1, len(stats_df) + 1)

    return stats_df


#
# n_simulations = 4
# simulated_tables = []
#
# for simulation_id in tqdm(range(1, n_simulations + 1)):
#     simulated_matches = []  # Initialize a list to store DataFrames for each match in the simulation
#
#     for index, matchId in enumerate(all_fixtures['matchId']):
#         simulated_data = simulate_match(matchId, upcoming_match_data=all_fixtures, k=1)
#         simulated_data['simulation_id'] = simulation_id
#         nested_data = pd.DataFrame([simulated_data])  # Convert dictionary to DataFrame
#         simulated_matches.append(nested_data)
#
#     # Concatenate the DataFrames for all matches in the current simulation
#     simulated_tables.extend(simulated_matches)
#
# # Concatenate all DataFrames for all simulations
# simulated_tables = pd.concat(simulated_tables, ignore_index=True)
#
# # Calculate the league table based on simulation_id
# simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()

def iterate_k_simulations_on_match_id_v1(matchId, shot_df, model_params, k=1):
    '''
    Performs k simulations on a match, and returns the probabilities of a win, loss, draw, along with average goals and team names.
    '''
    shots = shot_df[shot_df['matchId'] == matchId]

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']
    home_team_name = shots_home['TeamName'].values[0]
    away_team_name = shots_away['TeamName'].values[0]

    # Count the number of occurrences
    home_win = 0
    draw = 0
    away_win = 0

    # Initialize counters for goals
    home_goals_total = 0
    away_goals_total = 0

    # Perform k simulations
    for _ in range(k):
        # Simulate a match
        home_goals = simulate_goals(shots_home['expectedGoals'].to_numpy(), model_params["attack_" + home_team_name])
        away_goals = simulate_goals(shots_away['expectedGoals'].to_numpy(), model_params["attack_" + away_team_name])

        # Update outcome counters
        if home_goals == away_goals:
            draw += 1
        elif home_goals > away_goals:
            home_win += 1
        else:
            away_win += 1

        # Update goal counters
        home_goals_total += home_goals
        away_goals_total += away_goals

    # Calculate probabilities after the loop
    home_prob_final = home_win / k
    away_prob_final = away_win / k
    draw_prob_final = draw / k

    # Calculate average goals
    avg_home_goals = round(home_goals_total / k)
    avg_away_goals = round(away_goals_total / k)

    return {
        'home_prob': home_prob_final,
        'away_prob': away_prob_final,
        'draw_prob': draw_prob_final,
        'matchId': matchId,
        'home_goals': avg_home_goals,
        'away_goals': avg_away_goals,
        'home_team_name': home_team_name,
        'away_team_name': away_team_name
    }

def run_simulations(n_simulations, shot_df, remaining_games,  model_params):
    simulated_tables = []

    for simulation_id in tqdm(range(1, n_simulations + 1)):
        played_matches = []  # Initialize a list to store DataFrames for each played match in the simulation
        unplayed_matches = []  # Initialize a list to store DataFrames for each unplayed match in the simulation

        # Simulate played matches
        for index, matchId in enumerate(shot_df['matchId']):
            simulated_data = iterate_k_simulations_on_match_id_v1(matchId,shot_df, model_params, 1)
            simulated_data['simulation_id'] = simulation_id
            nested_data = pd.DataFrame([simulated_data])
            nested_data = nested_data.drop(columns={'home_prob', 'draw_prob', 'away_prob'})
            played_matches.append(nested_data)

        # Simulate unplayed matches
        for index, matchId in enumerate(remaining_games['matchId']):
            simulated_data = simulate_match(matchId, model_params, remaining_games, 1)
            simulated_data['simulation_id'] = simulation_id
            nested_data = pd.DataFrame([simulated_data])
            nested_data = nested_data.drop(columns={'probability'})
            unplayed_matches.append(nested_data)

        # Concatenate the DataFrames for all played matches and unplayed matches in the current simulation
        simulated_tables.extend(played_matches)
        simulated_tables.extend(unplayed_matches)

    # Concatenate all DataFrames for all simulations
    simulated_tables = pd.concat(simulated_tables, ignore_index=True)

    # Calculate the league table based on simulation_id
    simulated_tables = simulated_tables.groupby('simulation_id').apply(lambda x: calculate_table(x)).reset_index()

    return simulated_tables



def calculate_position_probabilities(league_table):
    total_simulations = league_table['simulation_id'].nunique()
    position_probabilities = league_table.groupby('team')['position'].value_counts(normalize=True).unstack(fill_value=0)
    position_probabilities *= 100
    position_probabilities = position_probabilities.reset_index()
    return position_probabilities


def calculate_xpts(df):
    expected_points = {}

    for match_id, match_df in df.groupby('matchId'):
        home_team_name = match_df['home_team_name'].iloc[0]
        away_team_name = match_df['away_team_name'].iloc[0]

        # Assuming you have match probabilities somewhere in your dataframe, adjust as needed
        home_prob = match_df['home_prob'].iloc[0]
        away_prob = match_df['away_prob'].iloc[0]
        draw_prob = match_df['draw_prob'].iloc[0]

        home_points = home_prob * 3 + draw_prob * 1
        away_points = away_prob * 3 + draw_prob * 1

        # Update the expected points for home team
        if home_team_name in expected_points:
            expected_points[home_team_name] += home_points
        else:
            expected_points[home_team_name] = home_points

        # Update the expected points for away team
        if away_team_name in expected_points:
            expected_points[away_team_name] += away_points
        else:
            expected_points[away_team_name] = away_points

    # Create a DataFrame for both home and away teams
    combined_df = pd.DataFrame({'TeamName': list(expected_points.keys()), 'points': list(expected_points.values())})

    return combined_df
