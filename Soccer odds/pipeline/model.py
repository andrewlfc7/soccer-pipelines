#%%
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
def log_likelihood(
        goals_home_observed,
        goals_away_observed,
        home_attack,
        home_defence,
        away_attack,
        away_defence,
        home_advantage,
        rho,
        weight
):
    goal_expectation_home = np.exp(home_attack + away_defence + home_advantage)
    goal_expectation_away = np.exp(away_attack + home_defence)

    home_llk = poisson.pmf(goals_home_observed, goal_expectation_home)
    away_llk = poisson.pmf(goals_away_observed, goal_expectation_away)
    adj_llk = rho_correction(
        goals_home_observed,
        goals_away_observed,
        goal_expectation_home,
        goal_expectation_away,
        rho,
    )

    if goal_expectation_home < 0 or goal_expectation_away < 0 or adj_llk < 0:
        return 10000

    log_llk = weight * (np.log(home_llk) + np.log(away_llk) + np.log(adj_llk))

    return -log_llk



from pprint import pprint
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def dc_decay(xi, t):
    return np.exp(-xi * t)


def fit_poisson_model(df, xi=0.0001):
    teams = np.sort(np.unique(np.concatenate([df["homeTeamName"], df["awayTeamName"]])))
    n_teams = len(teams)
    df["matchTimeUTCDate"] = pd.to_datetime(df["matchTimeUTCDate"])

    df["days_since"] = (df["matchTimeUTCDate"].max() - df["matchTimeUTCDate"]).dt.days
    df["weight"] = dc_decay(xi, df["days_since"])

    params = np.concatenate(
        (
            np.random.uniform(0.5, 1.5, (n_teams)),  # attack strength
            np.random.uniform(0, -1, (n_teams)),  # defence strength
            [0.25],  # home advantage
            [-0.1],  # rho
        )
    )

    def _fit(params, df, teams):
        attack_params = dict(zip(teams, params[:n_teams]))
        defence_params = dict(zip(teams, params[n_teams : (2 * n_teams)]))
        home_advantage = params[-2]
        rho = params[-1]

        llk = list()
        for idx, row in df.iterrows():
            tmp = log_likelihood(
                row["homegoal"],
                row["awaygoal"],
                attack_params[row["homeTeamName"]],
                defence_params[row["homeTeamName"]],
                attack_params[row["awayTeamName"]],
                defence_params[row["awayTeamName"]],
                home_advantage,
                rho,
                row["weight"],
            )
            llk.append(tmp)

        return np.sum(llk)

    options = {
        "maxiter": 100,
        "disp": False,
    }

    constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]

    res = minimize(
        _fit,
        params,
        args=(df, teams),
        constraints=constraints,
        options=options,
    )

    model_params = dict(
        zip(
            ["attack_" + team for team in teams]
            + ["defence_" + team for team in teams]
            + ["home_adv", "rho"],
            res["x"],
            )
    )

    return model_params




# def fit_poisson_model(df_goals):
#     teams = np.sort(np.unique(np.concatenate([df_goals["hometeam"], df_goals["awayteam"]])))
#     n_teams = len(teams)
#
#     params = np.concatenate(
#         (
#             np.random.uniform(0.5, 1.5, (n_teams)),  # attack strength
#             np.random.uniform(0, -1, (n_teams)),  # defence strength
#             [0.25],  # home advantage
#             [-0.1], # rho
#         )
#     )
#
#     def _fit(params, df, teams):
#         attack_params = dict(zip(teams, params[:n_teams]))
#         defence_params = dict(zip(teams, params[n_teams : (2 * n_teams)]))
#         home_advantage = params[-2]
#         rho = params[-1]
#
#         llk = list()
#         for idx, row in df.iterrows():
#             tmp = log_likelihood(
#                 row["homegoal"],
#                 row["awaygoal"],
#                 attack_params[row["hometeam"]],
#                 defence_params[row["hometeam"]],
#                 attack_params[row["awayteam"]],
#                 defence_params[row["awayteam"]],
#                 home_advantage,
#                 rho
#             )
#             llk.append(tmp)
#
#         return np.sum(llk)
#
#     options = {
#         "maxiter": 100,
#         "disp": False,
#     }
#
#     constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]
#
#     res = minimize(
#         _fit,
#         params,
#         args=(df_goals, teams),
#         constraints=constraints,
#         options=options,
#     )
#
#     model_params = dict(
#         zip(
#             ["attack_" + team for team in teams]
#             + ["defence_" + team for team in teams]
#             + ["home_adv", "rho"],
#             res["x"],
#             )
#     )
#
#     print("Log Likelihood: ", res["fun"])
#
#     return model_params


from sqlalchemy import create_engine
import pandas as pd

from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://postgres:Liverpool19@localhost:5432/soccer_forecasting')


conn = engine.connect()


query =f"""
SELECT * FROM epl_results;
"""

data = pd.read_sql(query, conn)


#%%
model_params = fit_poisson_model(df=data)
pprint(model_params)

import pickle

# Save the model_params to a file
with open("../data/model.pkl", "wb") as f:
    pickle.dump(model_params, f)
