from fotmob import get_season_matchgoals_data
from sqlalchemy import create_engine


engine = create_engine('postgresql://postgres:Liverpool19@localhost:5432/soccer_forecasting')


# Create a connection
with engine.connect() as conn:
    seasons = [
        '2023%2F2024',
        '2022%2F2023',
        '2021%2F2022',
        '2020%2F2021',
        '2019%2F2020'
    ]

    for season in seasons:
        data = get_season_matchgoals_data(season, 47)  # Assuming get_season_matchgoals_data is defined
        data.to_sql('epl_results', conn, if_exists='append', index=False)

    conn.commit()



