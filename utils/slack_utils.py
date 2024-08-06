from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os

slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

def send_slack_message(channel, message, thread_ts=None):
    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            text=message,
            thread_ts=thread_ts
        )
    except SlackApiError as e:
        print(f"Error sending message: {e}")