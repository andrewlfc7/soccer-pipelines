from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import pandas as pd
import requests

def get_shots_data(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')

    data = response.content
    data = json.loads(data)

    matchId = data['general']['matchId']
    matchTimeUTCDate = data['general']['matchTimeUTCDate'][:10]

    competitions = data['general']['parentLeagueName']
    teamcolors = data['general']['teamColors']


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

    df_shot['match_id'] = matchId
    df_shot['match_date'] = matchTimeUTCDate
    df_shot['competition'] = competitions


    df_shot['Venue'] = ''
    for index, row in df_shot.iterrows():
        if row['teamId'] == home_team_id:
            df_shot.loc[index, 'Venue'] = 'Home'
            df_shot.loc[index, 'TeamName'] = homeTeamName
        elif row['teamId'] == away_team_id:
            df_shot.loc[index, 'Venue'] = 'Away'
            df_shot.loc[index, 'TeamName'] = awayTeamName

    def extract_value(d, key):
        return d[key]

    df_shot['onGoalShot_X'] = df_shot['onGoalShot'].apply(extract_value, args=('x',))
    df_shot['onGoalShot_Y'] = df_shot['onGoalShot'].apply(extract_value, args=('y',))
    df_shot['onGoalShot_ZR'] = df_shot['onGoalShot'].apply(extract_value, args=('zoomRatio',))
    df_shot.drop(['onGoalShot'], axis=1, inplace=True)
    if 'shortName' in df_shot.columns:
        df_shot.drop(['shortName'], axis=1, inplace=True)


    return df_shot

from retrying import retry

@retry(stop_max_attempt_number=2)
def get_match_event_data(date):
    chrome_driver_path = "chromedriver-mac-x64/chromedriver"
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")

    driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)

    driver.get('https://www.whoscored.com/Teams/26/Fixtures/England-Liverpool')

    date_to_search = date

    # Find the div element containing fixture data
    fixture_div_element = driver.find_element(By.XPATH, '//*[@id="team-fixtures"]')

    # Find all rows within the fixture div element
    fixture_rows = fixture_div_element.find_elements(By.CLASS_NAME, 'divtable-row')

    match_link = ""
    # Iterate through the fixture rows and search for the desired date
    for row in fixture_rows:
        date_element = row.find_element(By.CLASS_NAME, 'date')
        if date_to_search in date_element.text:
            match_link_element = row.find_element(By.CLASS_NAME, 'box')
            match_link = match_link_element.get_attribute('href')
            print(f"Match link for date {date_to_search}: {match_link}")
            break

    driver.get(match_link)

    driver.implicitly_wait(10)

    chalkboard = driver.find_element(By.XPATH, '//*[@id="sub-navigation"]/ul/li[4]/a')
    chalkboard.click()

    wait = WebDriverWait(driver, 15)
    script_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="layout-wrapper"]/script[1]')))

    script_text = script_element.get_attribute("textContent")

    # Close the webdriver
    driver.quit()
    return script_text


def data_preprocessing(script_text):
    edited_content = script_text.replace('require.config.params["args"] = ', '')
    # edited_content = edited_content.replace('{\n', '')
    edited_content = edited_content.replace('\n', '')
    edited_content = edited_content.replace(';', '')

    edited_content = edited_content.replace('matchId', '"matchId"')
    edited_content = edited_content.replace('matchCentreData', '"matchCentreData"')
    edited_content = edited_content.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
    edited_content = edited_content.replace('formationIdNameMappings', '"formationIdNameMappings"')

    # Parse JSON
    parsed_json = json.loads(edited_content)

    return parsed_json




