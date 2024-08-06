import os
import threading
from slack_bolt.adapter.socket_mode import SocketModeHandler
from config import slack_app
from assistant import SalesforceAssistant
from utils.slack_utils import send_slack_message

assistant = SalesforceAssistant()

@slack_app.event("message")
def message_handler(message, say, ack):
    ack()
    thread_ts = message.get("thread_ts", message["ts"])
    say("Your request is being processed, please hold on...", thread_ts=thread_ts)

    def process_and_respond():
        response = assistant.handle_slack_event(message)
        send_slack_message(message["channel"], response, thread_ts=thread_ts)

    threading.Thread(target=process_and_respond).start()

if __name__ == "__main__":
    handler = SocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"])
    handler.start()