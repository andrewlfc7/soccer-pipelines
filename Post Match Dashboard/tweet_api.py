import os
import json
from requests_oauthlib import OAuth1Session

# Add your consumer key and secret here
consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")

# Create a new OAuth session
oauth = OAuth1Session(consumer_key, client_secret=consumer_secret)

# Get request token
request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"

try:
    fetch_response = oauth.fetch_request_token(request_token_url)
except ValueError:
    print("There may have been an issue with the consumer_key or consumer_secret you entered.")

resource_owner_key = fetch_response.get("oauth_token")
resource_owner_secret = fetch_response.get("oauth_token_secret")
print("Got OAuth token: %s" % resource_owner_key)

# Get authorization
base_authorization_url = "https://api.twitter.com/oauth/authorize"
authorization_url = oauth.authorization_url(base_authorization_url)
print("Please go here and authorize: %s" % authorization_url)
verifier = input("Paste the PIN here: ")

# Get the access token
access_token_url = "https://api.twitter.com/oauth/access_token"
oauth = OAuth1Session(
    consumer_key,
    client_secret=consumer_secret,
    resource_owner_key=resource_owner_key,
    resource_owner_secret=resource_owner_secret,
    verifier=verifier,
)
oauth_tokens = oauth.fetch_access_token(access_token_url)

access_token = oauth_tokens["oauth_token"]
access_token_secret = oauth_tokens["oauth_token_secret"]

# Prepare the media files for the first tweet
media_files = []
figures_folder = "figures"
for filename in os.listdir(figures_folder):
    if filename.endswith(".png"):  # Adjust the file extension as needed
        with open(os.path.join(figures_folder, filename), "rb") as f:
            media = f.read()
            media_files.append(("media", (filename, media, "image/png")))

# Upload media for the first tweet
media_ids = []
for i, media in enumerate(media_files, start=1):
    response = oauth.post(
        "https://upload.twitter.com/1.1/media/upload.json",
        files={media[0]: media[1]},
    )

    if response.status_code != 200:
        raise Exception(
            "Error uploading media: {} {}".format(response.status_code, response.text)
        )

    media_id = response.json()["media_id"]
    media_ids.append(media_id)

# Attach media to the first tweet payload
payload = {
    "status": "Match Name: [Your Match Name]\nDashboard: [Your Dashboard]",
    "media_ids": ",".join(str(id) for id in media_ids)
}

# Making the request to post the first tweet with media
response = oauth.post("https://api.twitter.com/2/tweets", json=payload)

if response.status_code != 201:
    raise Exception(
        "Request returned an error: {} {}".format(response.status_code, response.text)
    )

print("First tweet posted successfully!")

# For subsequent tweets, you can upload and attach media as needed in a similar loop.
# You can provide the match name and dashboard information when you're ready.
