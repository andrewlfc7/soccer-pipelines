# pipeline.model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import os
class Dixon_Coles_Model:
    def rho_correction(self, goals_home, goals_away, home_exp, away_exp, rho):
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
            self,
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
        adj_llk = self.rho_correction(
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

    def dc_decay(self, xi, t):
        return np.exp(-xi * t)

    def fit_poisson_model(self, df, xi=0.0001):
        teams = np.sort(np.unique(np.concatenate([df["homeTeamName"], df["awayTeamName"]])))
        n_teams = len(teams)
        df["matchTimeUTCDate"] = pd.to_datetime(df["matchTimeUTCDate"])

        df["days_since"] = (df["matchTimeUTCDate"].max() - df["matchTimeUTCDate"]).dt.days
        df["weight"] = self.dc_decay(xi, df["days_since"])

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
                tmp = self.log_likelihood(
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

def is_model_built(league_name):
    model_path = f'../model/model{league_name}.pkl'
    return os.path.exists(model_path)
