import requests
from requests_oauthlib import OAuth1
import os


consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")
access_token = os.environ.get("ACCESS_TOKEN")
access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")


def connect_to_oauth(consumer_key, consumer_secret, acccess_token, access_token_secret):
    url = "https://api.twitter.com/2/tweets"
    auth = OAuth1(consumer_key, consumer_secret, acccess_token, access_token_secret)
    return url, auth



bear = 'AAAAAAAAAAAAAAAAAAAAAIoWlgEAAAAAshQ1IqU3uBssRP7bXt7unPSKP%2B0%3DpTqrvyMN60rFDBQJpaA86fxeuX7YzMunxoknjUXOodUdZdXcRL'
