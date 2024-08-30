import json
from pathlib import Path
from typing import TypedDict, List, Optional

USERS_FILE = Path("users.json")


class User(TypedDict):
    username: str
    password: str
    classification: Optional[str]


def get_users() -> List[User]:
    if USERS_FILE.exists():
        with USERS_FILE.open("r") as f:
            users = json.load(f)
    else:
        users = []

    return users


def get_user_by_name(username: str):
    users = get_users()

    for user in users:
        if user["username"] == username:
            return user

    return None


def update_user_classification(username: str, classification: str) -> None:
    users = get_users()

    for curr_user in users:
        if curr_user["username"] == username:
            curr_user["classification"] = classification

    write_users(users)


def write_users(users: List[User]):
    with USERS_FILE.open("w") as f:
        json.dump(users, f)
