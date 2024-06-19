import os

import requests
from dotenv import load_dotenv

load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")

if github_token is None:
    raise ValueError("GitHub token not found in environment variables.")

github_headers = {
    "Authorization": f"token {github_token}"
}


def download_model_from_github(github_url, model_path):
    response = requests.get(github_url, headers=github_headers)

    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download model from GitHub")
