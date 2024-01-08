from pipeline.fotmob import *
from pipeline.model import *
from pipeline.visuals import *
from pipeline.sim import *
from pipeline.utils import *
import os

import joblib
import yaml
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine,exc
import joblib
import sqlalchemy
from google.cloud import storage
from io import BytesIO
import datetime

with open('pipeline/database_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

user = config['PGUSER']
passwd = config['PGPASSWORD']
host = config['PGHOST']
port = config['PGPORT']
db = config['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'

engine = create_engine(db_url)


os.environ['GOOGLE_APPLICATION_CREDENTIALS']='careful-aleph-398521-3eb70ef00c68.json'

storage_client = storage.Client()
bucket_name = "soccer-forecasting"
bucket = storage_client.get_bucket(bucket_name)

#%%
colors = ['#036666', '#14746F', '#248277', '#358F80','#ffffff' ,'#E5383B', '#BA181B', '#A4161A', '#660708']

xpts_cm = LinearSegmentedColormap.from_list('xpts_preformance', colors, N=250)
cm.register_cmap(name='xpts_preformance', cmap=xpts_cm)

#%%
colors_odds = ['#d8f3dc', '#b7e4c7', '#95d5b2', '#74c69d', '#52b788', '#40916c', '#2d6a4f', '#1b4332', '#081c15'
               ]

odd_cm = LinearSegmentedColormap.from_list('ODD', colors_odds, N=250)
cm.register_cmap(name='ODD', cmap=odd_cm)


league_name='English Premier League'

league_id = 47
fixtures_data = get_league_fixtures(league_id)

match_rounds = fixtures_data['matches']['allMatches']
current_round = fixtures_data['matches']['firstUnplayedMatch']['firstRoundWithUnplayedMatch']
season_length = len(fixtures_data['matches']['allMatches'])



today = datetime.date.today().strftime("%Y%m%d")


if are_all_matches_finished(season_length, match_rounds, current_round):
    print(f"All matches in round {current_round - 1} are finished.")
    if not does_fixtures_exist('epl_fixtures', engine):
        all_fixtures = get_league_fixtures(league_id)
    else:
        all_fixtures = get_fixtures_data('epl_fixtures', engine)
    all_fixtures['matchRound'] = all_fixtures['matchRound'].astype(int)
    all_fixtures['matchId'] = all_fixtures['matchId'].astype(int)

    all_fixtures = all_fixtures.replace({'Tottenham': 'Tottenham Hotspur'})


    upcoming_games = all_fixtures[all_fixtures['matchRound']==current_round]

    upcoming_games['matchId'] = upcoming_games['matchId'].astype(int)

    if not is_model_built('Premier_League'):
        df_goals = get_season_matchgoals_data('2023/2024', 47)
        df_goals = df_goals.replace({'Tottenham': 'Tottenham Hotspur'})


        dc_model = Dixon_Coles_Model()
        model_params = dc_model.fit_poisson_model(df=df_goals, xi=0.0001)
        model_path = f'model/modelPremier_League.pkl'
        joblib.dump(model_params, model_path)
    else:
        model_path =  f'model/modelPremier_League.pkl'
        model_params = joblib.load(model_path)

    match_probs = []
    for index, match in enumerate(upcoming_games['matchId']):
        outcome_probs = iterate_k_simulations_upcoming_matches(match,upcoming_games, model_params,100)
        match_probs.append(outcome_probs)
        if index % 10 == 0:
            print(f'{index / upcoming_games.shape[0]:.1%} done.')

    match_probs = pd.DataFrame(match_probs)
    merged_df = pd.merge(upcoming_games, match_probs, on='matchId')
    figure_buffer_matchround_forecast = BytesIO()
    matchround_forecast(df=merged_df, league='EPL', fotmob_leagueid=47,cmap='ODD').savefig(
        figure_buffer_matchround_forecast,
        format="png",  # Use the appropriate format for your figure
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )

    figure_buffer_matchround_forecast.seek(0)
    blob_path_matchround_forecast = f"figures/{today}/matchround_forecast_{league_name}_{today}.png"
    blob_matchround_forecast = bucket.blob(blob_path_matchround_forecast)
    blob_matchround_forecast.upload_from_file(figure_buffer_matchround_forecast, content_type="image/png")
    # Close the BytesIO buffers
    figure_buffer_matchround_forecast.close()



    db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'
    engine = create_engine(db_url)

    if not does_matches_exist('english_premier_league_matches',engine):
        shots_data = get_latest_comp_shotsdata(season_length, league_id=47)
        shots_data.to_sql(f'{league_name}_matches', con=engine, if_exists='append', index=False)
        shots_data['expectedGoals'] = shots_data['expectedGoals'].fillna(0)
        shots_data['matchId'] = shots_data['matchId'].astype(int)

        played_matches_probs = []
        for index, match in enumerate(shots_data['matchId']):
            outcome_probs = simulate_match_on_shots_xg(matchId=match, shot_df=played_matches_probs, model_params=model_params, k=100)
            played_matches_probs.append(outcome_probs)
            if index % 10 == 0:
                print(f'{index / shots_data.shape[0]:.1%} done.')

        played_matches_probs = pd.DataFrame(played_matches_probs)
        played_matches_probs.to_sql(f'{league_name}_matches_sim', con=engine, if_exists='append', index=False)
    else:
        shots_data = get_matches_data('english_premier_league_matches',engine)
        shots_data['expectedGoals'] = shots_data['expectedGoals'].fillna(0)
        shots_data['matchId'] = shots_data['matchId'].astype(int)
        shots_data = shots_data.replace({'Tottenham': 'Tottenham Hotspur'})




    shots_match_probs = []
    for index, match in enumerate(shots_data['matchId']):
        outcome_probs = simulate_match_on_shots_xg(match, shots_data, model_params, 100)
        shots_match_probs.append(outcome_probs)
        if index % 10 == 0:
            print(f'{index / shots_data.shape[0]:.1%} done.')

    shots_match_probs = pd.DataFrame(shots_match_probs)


    xpts = calculate_xpts(shots_match_probs)
    xg_table = calculate_xg_table(shots_data)
    xpts = xg_table.merge(xpts, on='TeamName')

    table = get_league_table(league_id=47)
    xPoints_table = xpts.merge(table, on='TeamName')
    xPoints_table = xpoints_table_pre90_stats(xPoints_table)


    xPoints_table = xPoints_table.sort_values(by='pts', ascending=False)

    # xpt_table(xPoints_table, league_name='English Premier League',cmap='xpts_preformance').savefig(f'xPoints_table{current_round -1}.png')

    all_fixtures['matchRound'] = all_fixtures['matchId'].astype(int)
    all_fixtures = all_fixtures.replace({'Tottenham': 'Tottenham Hotspur'})


    remaining_games = all_fixtures[all_fixtures['matchRound'] >= current_round]

    # remaining_fixtures = get_remaining_fixtures(season_length, match_rounds, current_round)
    remaining_fixtures_match_probs = []
    for index, match in enumerate(remaining_games['matchId']):
        outcome_probs = simulate_match(matchId=match, upcoming_match_data=remaining_games, model_params=model_params, k=100)
        remaining_fixtures_match_probs.append(outcome_probs)
        if index % 10 == 0:
            print(f'{index / remaining_games.shape[0]:.1%} done.')

    remaining_fixtures_probs = pd.DataFrame(remaining_fixtures_match_probs)

    shots_match_probs = shots_match_probs.drop(columns={'home_prob', 'draw_prob', 'away_prob'}).groupby('matchId').tail(1)
    remaining_fixtures_probs = remaining_fixtures_probs.drop(columns={'probability'})
    shots_match_probs['matchId'] = shots_match_probs['matchId'].astype(int)
    remaining_fixtures_probs['matchId'] = remaining_fixtures_probs['matchId'].astype(int)

    results = pd.merge(shots_match_probs, remaining_fixtures_probs, how='outer',
                       left_on=['home_team_name', 'away_team_name', 'home_goals', 'away_goals', 'matchId'],
                       right_on=['home_team_name', 'away_team_name', 'home_goals', 'away_goals', 'matchId'])

    sim_table = calculate_table(results)
    sim_table['team_id'] = sim_table['team'].map(get_team_id_mapping(league_id))
    updated_sim_table = calculate_per_90_metrics(sim_table)

    eos_sim_results = run_simulations(100, shots_data,remaining_games,model_params)
    eos_sim_results['team_id'] = eos_sim_results['team'].map(get_team_id_mapping(league_id))

    position_prob = calculate_position_probabilities(eos_sim_results)
    position_prob['team_id'] = position_prob['team'].map(get_team_id_mapping(league_id))

    # eos_distribution_v1(eos_sim_results, league_name)
    # plot_eos_table_v1(updated_sim_table, league_name)
    # plot_finishing_position_distribution(position_prob, league_name)
    # Save eos_distribution plot to a BytesIO buffer
    figure_buffer_xpt_table = BytesIO()
    xpt_table(xPoints_table, league_name='English Premier League').savefig(
        figure_buffer_xpt_table,
        format="png",  # Use the appropriate format for your figure
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_xpt_table.seek(0)


    # Save eos_distribution plot to a BytesIO buffer
    figure_buffer_eos_distribution = BytesIO()
    eos_distribution_v1(eos_sim_results, league_name).savefig(
        figure_buffer_eos_distribution,
        format="png",  # Use the appropriate format for your figure
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_eos_distribution.seek(0)

    # Save plot_eos_table plot to a BytesIO buffer
    figure_buffer_eos_table = BytesIO()
    plot_eos_table_v1(updated_sim_table, league_name).savefig(
        figure_buffer_eos_table,
        format="png",  # Use the appropriate format for your figure
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_eos_table.seek(0)

    # Save plot_finishing_position_distribution plot to a BytesIO buffer
    figure_buffer_finishing_position = BytesIO()
    plot_finishing_position_distribution(position_prob, league_name).savefig(
        figure_buffer_finishing_position,
        format="png",  # Use the appropriate format for your figure
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer_finishing_position.seek(0)


    # Specify the blob paths within the bucket using the plot names and the current date
    blob_path_eos_distribution = f"figures/{today}/eos_distribution_{league_name}_{today}.png"
    blob_path_eos_table = f"figures/{today}/eos_table_{league_name}_{today}.png"
    blob_path_finishing_position = f"figures/{today}/finishing_position_odds_{league_name}_{today}.png"
    blob_path_xpt_table = f"figures/{today}/xpt_table_{league_name}_{today}.png"


    # Create Blobs and upload the figures
    blob_eos_distribution = bucket.blob(blob_path_eos_distribution)
    blob_eos_distribution.upload_from_file(figure_buffer_eos_distribution, content_type="image/png")

    blob_eos_table = bucket.blob(blob_path_eos_table)
    blob_eos_table.upload_from_file(figure_buffer_eos_table, content_type="image/png")

    blob_finishing_position = bucket.blob(blob_path_finishing_position)
    blob_finishing_position.upload_from_file(figure_buffer_finishing_position, content_type="image/png")

    blob_xpt_table = bucket.blob(blob_path_xpt_table)
    blob_xpt_table.upload_from_file(figure_buffer_xpt_table, content_type="image/png")

    # Close the BytesIO buffers
    figure_buffer_eos_distribution.close()
    figure_buffer_eos_table.close()
    figure_buffer_finishing_position.close()
    figure_buffer_xpt_table.close()




