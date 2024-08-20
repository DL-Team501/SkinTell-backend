import json
from pathlib import Path

USERS_FILE = Path("users.json")


def get_users():
    if USERS_FILE.exists():
        with USERS_FILE.open("r") as f:
            users = json.load(f)
    else:
        users = []

    return users


def write_users(users):
    with USERS_FILE.open("w") as f:
        json.dump(users, f)
