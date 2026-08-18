"""Run once in a trusted environment to create a YouTube refresh token.
Never commit client secrets or the printed refresh token.
"""
import argparse
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]

parser = argparse.ArgumentParser()
parser.add_argument("client_secrets_json")
args = parser.parse_args()
flow = InstalledAppFlow.from_client_secrets_file(args.client_secrets_json, SCOPES)
creds = flow.run_local_server(port=0)
print("YOUTUBE_REFRESH_TOKEN=" + (creds.refresh_token or ""))
