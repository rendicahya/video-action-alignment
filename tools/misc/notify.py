import sys

sys.path.append(".")

import click
import requests
from config import settings as conf


@click.command()
@click.argument(
    "message",
    nargs=1,
    required=True,
    type=str,
)
def notify(message):
    url = f"https://api.telegram.org/bot{conf.telegram.token}"
    params = {"chat_id": conf.telegram.id, "text": message}

    try:
        requests.get(url + "/sendMessage", params=params)
    except:
        print("Failed to send Telegram message")


if __name__ == "__main__":
    notify()
