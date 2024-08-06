# Project Structure

```
.env
.env.example
.env.local
.github
└── workflows
  └── deploy.yml
.gitignore
Dockerfile
__pycache__
assistant.py
assistant_instructions.txt
commit_changes.sh
config.py
context.py
context_output.md
globals.py
init_assistant.py
init_rio.sh
project_tree.py
prompts
└── code_debugger.txt
└── flow_management
  └── create_flow.txt
  └── execute_flow.txt
  └── modify_flow.txt
└── generate_code.txt
└── generate_code_old.txt
└── generate_code_refine.txt
└── get_columns.txt
└── get_filters.txt
└── get_filters_new.txt
└── get_tables.txt
└── identify_task.txt
└── plan_data_loader.md
└── user_input.txt
public
└── message_flow.png
└── multi-agent.png
└── services.png
readme.md
requirements.txt
server.log
server.py
slackbot.json
start.sh
stop.sh
system_prompt.md
test
test.sh
threads_db.db
timeit.log
tools
└── __pycache__
└── data_loader.py
└── flow_management
  └── __init__.py
  └── flow_executor.py
  └── flow_manager.py
  └── flow_retriever.py
└── generate_chart.py
└── metadata.py
└── soql.py
└── test_soql.py
utils
└── __pycache__
└── auth.py
└── connection.py
└── slack_utils.py
└── time.py
validation_rules.md
```

# File: ./init_assistant.py

# run this once to fetch assistant_id and then
# plug it in server.py file

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_assistant():
    function_signatures = [
        {
            "name": "get_all_sf_tables",
            "description": "Fetches all the available Salesforce table names.",
        },
        {
            "name": "get_all_cols_of_sf_table",
            "description": "Fetches all the columns available for the given Salesforce table name.",
            "parameters": {
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "required": ["table_name"],
            },
        },
        {
            "name": "execute_soql_query",
            "description": "Executes a Salesforce SOQL query. This function runs the provided SOQL query against the Salesforce database and returns the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sf_conn": {
                        "type": "object",
                        "description": "Salesforce connection instance",
                    },
                    "soql_query": {"type": "string"},
                },
                "required": ["soql_query"],
            },
        },
        {
            "name": "generate_chartjs_config_and_chart",
            "description": "Generates Chart.js configuration and chart, and returns the configuration in JSON format. This function helps in creating visual representations of data using Chart.js.",
            "parameters": {
                "type": "object",
                "properties": {
                    "config": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["config", "title"],
            },
        },
        {
            "name": "initiate_data_loading",
            "description": "Initiates the data loading process for Salesforce. This function handles the loading of data into Salesforce, supporting operations like insert, update, delete, and upsert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["insert", "update", "delete", "upsert"],
                    },
                    "object_name": {"type": "string"},
                    "file_url": {"type": "string"},
                },
                "required": ["operation", "object_name", "file_url"],
            },
        },
        {
            "name": "deploy_validation_rule",
            "description": "Deploy a validation rule to Salesforce. You have to first generate validation_rule object as described below : ",
            "description": '''Deploy validation rule(s) to Salesforce by first iteratively generating metadata object and then validation rule object as shown below : 
                    validation_rule = {
                        "FullName": "add appropriate validation rule name",
                        "Metadata": {
                            "errorConditionFormula": "add appropriate condition formula",
                            "errorMessage": "add appropriate error message",
                            "description": "add appropriate description",
                            "active": True
                        },
                        "EntityDefinition": {"fullName": "sobject_name"}
                        } ''',
            "parameters": {
                "type": "object",
                "properties": {
                    "validation_rule": {
                        "type": "object",
                        "description": "Validation rule object",
                    }
                },
                "required": ["validation_rule"],
            },
        },
        {
            "name": "execute_apex_code",
            "description": "Executes a Salesforce Apex code. This function runs the provided Apex code within the Salesforce environment, allowing for custom business logic execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "apex_code": {"type": "string"},
                },
                "required": ["apex_code"],
            },
        },
    ]

    # Read instructions from a text file
    instructions_file_path = os.path.join(os.path.dirname(__file__), 'assistant_instructions.txt')
    with open(instructions_file_path, 'r') as file:
        instructions = file.read().strip()

    assistant = client.beta.assistants.create(
        name="Salesforce Assistant (v0.2)",
        instructions=instructions,
        tools=[
            {"type": "function", "function": signature}
            for signature in function_signatures
        ],
        model="gpt-4o",
    )
    return assistant


# Example usage
assistant_object = create_assistant()
print(assistant_object.id)


# File: ./server.py

from config import slack_app, openai_client
import os
import threading
import shelve
from slack_bolt.adapter.socket_mode import SocketModeHandler
from assistant import process_thread_with_assistant
from globals import OS_THREAD_ID_SLACK_THREAD_MAPPER
import datetime

# Slack message handler
@slack_app.event("message")
def message_handler(message, say, ack):
    ack()  # Acknowledge the event immediately
    thread_ts = message.get("thread_ts", message["ts"])
    say("Your request is being processed, please hold on...", thread_ts=thread_ts)

    curr_datetime = datetime.datetime.utcnow().strftime("%d %B %Y %H:%M")
    user_query = f"Current UTC date time for reference {curr_datetime}.\n\n"
    user_query += message["text"]
    assistant_id = os.getenv("OPEN_AI_ASSISTANT_ID")
    from_user = message["user"]

    active_threads.setdefault(message["channel"], {})
    active_threads[message["channel"]].setdefault(
        thread_ts, {"openai_thread_id": None, "last_msg_sent_at": 0, "tool_outputs": {}}
    )

    # Uploaded CSV file for dataloader assistant
    if "files" in message and message["files"][0]["filetype"] == "csv":
        user_query += (
            f"\n\nHere is the uploaded CSV file: {message['files'][0]['url_private']}"
        )

    def process_and_respond():
        os_thread_id = threading.get_ident()
        if type(os_thread_id) == tuple:
            os_thread_id = os_thread_id[0]

        OS_THREAD_ID_SLACK_THREAD_MAPPER[os_thread_id] = {
            "channel": message["channel"],
            "thread_ts": thread_ts,
            "event_ts": message["ts"],
        }

        process_thread_with_assistant(
            user_query=user_query,
            assistant_id=assistant_id,
            from_user=from_user,
            channel=message["channel"],
            thread_ts=thread_ts,
            event_ts=message["ts"],
            openai_client=openai_client,
            slack_app=slack_app,
            active_threads=active_threads,
        )

        OS_THREAD_ID_SLACK_THREAD_MAPPER.pop(os_thread_id, None)

    threading.Thread(target=process_and_respond).start()


# Start the Slack app
if __name__ == "__main__":
    # active threads DS <ACTUAL VALUE WITHIN BRACKETS>
    # acitve_threads = {
    #     "<CHANNEL_ID>": {
    #         "<SLACK_THREAD_TS>": {
    #             "openai_thread_id": "<OPENAI_THREAD_ID>",
    #             "last_msg_sent_at": last message created_at EPOCH_TIME returned from OpenAI assistant
    #             "tool_ouptuts": {}
    #         }
    #     }
    # }
    active_threads = shelve.open("threads_db", writeback=True)
    try:
        SocketModeHandler(slack_app, os.getenv("SLACK_APP_TOKEN")).start()
    finally:
        active_threads.close()


# File: ./config.py

import os
from dotenv import load_dotenv
from slack_bolt import App
from openai import OpenAI
import logging
import threading
from globals import OS_THREAD_ID_SLACK_THREAD_MAPPER

load_dotenv()


def custom_log_factory(*args, **kwargs):
    record = base_log_factory(*args, **kwargs)

    os_thread_id = threading.get_ident()
    if type(os_thread_id) == tuple:
        os_thread_id = os_thread_id[0]

    m = OS_THREAD_ID_SLACK_THREAD_MAPPER.get(
        os_thread_id,
        {
            "channel": None,
            "thread_ts": None,
            "event_ts": None,
        },
    )
    record.slack_thread_ts = m["thread_ts"]
    record.slack_event_ts = m["event_ts"]
    record.slack_channel = m["channel"]

    return record


def create_logger(name, filename, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(slack_channel)s - %(slack_thread_ts)s - %(slack_event_ts)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if os.getenv("ENV") == "DEV":
        file_handler = logging.FileHandler(filename=filename, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


## Initialize loggers
base_log_factory = logging.getLogRecordFactory()
logging.setLogRecordFactory(custom_log_factory)

general_logger = create_logger("general", "server.log")
timeit_logger = create_logger("timeit", "timeit.log")

# Initialize clients
slack_app = App(token=os.getenv("SLACK_BOT_TOKEN"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# File: ./globals.py

OS_THREAD_ID_SLACK_THREAD_MAPPER = dict()


# File: ./assistant.py

import json
import logging
from typing import Dict, Any
import time
import shelve

from openai import OpenAI, OpenAIError
from slack_bolt import App

from tools.soql import (
    execute_soql_query,
    get_all_sf_tables,
    get_all_cols_of_sf_table,
    execute_apex_code,
)
from tools.generate_chart import generate_chartjs_config, get_chart_url_from_config
from tools.data_loader import initiate_data_loading
from tools.metadata import deploy_validation_rule
from utils import slack_utils
from utils.connection import get_salesforce_connection
from slack_sdk.errors import SlackApiError
from simple_salesforce import Salesforce
from utils.time import timeit

general_logger = logging.getLogger("general")


@timeit()
def process_thread_with_assistant(
    user_query: str,
    assistant_id: str,
    from_user: str,
    channel: str,
    thread_ts: str,
    event_ts: str,
    openai_client: OpenAI,
    slack_app: App,
    active_threads: shelve.Shelf[Any],
):
    try:
        general_logger.info(f"User query -> {user_query}")
        openai_thread_id = get_or_create_thread(
            active_threads, thread_ts, channel, openai_client
        )
        run = create_and_run_thread(
            openai_client, assistant_id, openai_thread_id, user_query
        )

        sf_conn = get_salesforce_connection()

        while True:
            run_status = openai_client.beta.threads.runs.retrieve(
                thread_id=openai_thread_id,
                run_id=run.id,
            )

            general_logger.info(f"Run Status {run_status.status}")

            if run_status.status == "completed":
                break
            elif run_status.status == "requires_action":
                run = process_required_actions(
                    sf_conn,
                    run_status,
                    openai_client,
                    openai_thread_id,
                    slack_app,
                    channel,
                    thread_ts,
                    active_threads,
                )
            else:
                process_completed_run(
                    openai_client,
                    openai_thread_id,
                    active_threads,
                    slack_app,
                    channel,
                    thread_ts,
                )
            time.sleep(1)

        if run_status.status == "completed":
            process_completed_run(
                openai_client,
                openai_thread_id,
                active_threads,
                slack_app,
                channel,
                thread_ts,
            )

    except Exception as e:
        handle_process_error(e, slack_app, channel, thread_ts)


@timeit()
def get_or_create_thread(
    active_threads: shelve.Shelf[Any],
    thread_ts: str,
    channel: str,
    openai_client: OpenAI,
) -> str:
    openai_thread_id = active_threads[channel][thread_ts]["openai_thread_id"]
    if not openai_thread_id:
        openai_thread_id = openai_client.beta.threads.create().id
        active_threads[channel][thread_ts]["openai_thread_id"] = openai_thread_id
    return openai_thread_id


@timeit()
def create_and_run_thread(
    openai_client: OpenAI, assistant_id: str, thread_id: str, user_query: str
):
    openai_client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_query
    )
    return openai_client.beta.threads.runs.create(
        assistant_id=assistant_id,
        thread_id=thread_id,
    )


@timeit()
def process_required_actions(
    sf_conn,
    run_status,
    openai_client,
    thread_id,
    slack_app,
    channel,
    thread_ts,
    active_threads,
):
    tool_outputs = []
    for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        general_logger.info(
            f"Attempting to execute function: {function_name} with arguments: {arguments}"
        )

        output = dict()

        try:
            if function_name == "initiate_data_loading":
                output = handle_data_loading(
                    sf_conn, arguments, slack_app, channel, thread_ts
                )
            elif function_name == "execute_soql_query":
                output = handle_soql_query(
                    sf_conn, arguments, active_threads, thread_ts, channel
                )
            elif function_name == "generate_chartjs_config_and_chart":
                output = handle_chart_generation(
                    arguments, active_threads, thread_ts, slack_app, channel
                )
            elif function_name == "get_all_sf_tables":
                output = get_all_sf_tables(sf_conn)
            elif function_name == "get_all_cols_of_sf_table":
                output = get_all_cols_of_sf_table(sf_conn, **arguments)
            elif function_name == "deploy_validation_rule":
                output = deploy_validation_rule(sf_conn, **arguments)
            elif function_name == "execute_apex_code":
                apex_code = arguments.get("apex_code")
                if apex_code and isinstance(apex_code, str):
                    output = execute_apex_code(sf_conn, apex_code)
                else:
                    output = {"error": "Invalid or missing Apex code"}
        except Exception as e:
            output = json.dumps({"error": str(e)})
     
        tool_outputs.append(
            {"tool_call_id": tool_call.id, "output": json.dumps(output)}
        )

    return openai_client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id, run_id=run_status.id, tool_outputs=tool_outputs
    )


@timeit()
def handle_data_loading(
    sf_conn: Salesforce, arguments: dict, slack_app: App, channel: str, thread_ts: str
) -> dict:
    return initiate_data_loading(sf_conn, **arguments)


@timeit()
def handle_soql_query(
    sf_conn: Salesforce,
    arguments: dict,
    active_threads: shelve.Shelf[Any],
    thread_ts: str,
    channel: str,
) -> dict:
    output = execute_soql_query(sf_conn, **arguments)
    if "csv_data" in output:
        active_threads[channel][thread_ts]["tool_outputs"].update(
            {"csv_data": output["csv_data"]}
        )
    return output


@timeit()
def handle_chart_generation(
    arguments: dict,
    active_threads: shelve.Shelf[Any],
    thread_ts: str,
    slack_app,
    channel: str,
) -> dict:
    return generate_and_add_chart_to_active_thread(
        arguments, active_threads, slack_app, channel, thread_ts
    )


def generate_and_add_chart_to_active_thread(
    arguments: dict,
    active_threads: shelve.Shelf[Any],
    slack_app,
    channel: str,
    thread_ts: str,
) -> dict:
    chart_js_config = generate_chartjs_config(arguments["config"])
    chart_url = get_chart_url_from_config(chart_js_config["config"])

    active_threads[channel][thread_ts]["tool_outputs"].update(
        {
            "chart_url": chart_url,
            "chart_title": arguments["title"],
        }
    )

    return {"response": "Chart generated"}


@timeit()
def process_completed_run(
    openai_client: OpenAI,
    thread_id: str,
    active_threads: shelve.Shelf[Any],
    slack_app,
    channel: str,
    thread_ts: str,
):
    new_last_msg_sent_at = 0
    messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
    for message in messages.data:
        last_msg_sent_at = active_threads[channel][thread_ts]["last_msg_sent_at"]
        if message.role == "assistant" and (
            not last_msg_sent_at or message.created_at > last_msg_sent_at
        ) and message.content:
            for content in message.content:
                if content.type == "text":
                    slack_app.client.chat_postMessage(
                        channel=channel, text=content.text.value, thread_ts=thread_ts
                    )

            new_last_msg_sent_at = max(new_last_msg_sent_at, message.created_at)

    if new_last_msg_sent_at:
        active_threads[channel][thread_ts]["last_msg_sent_at"] = new_last_msg_sent_at

    if active_threads[channel][thread_ts]["tool_outputs"].get("chart_url", None):
        slack_app.client.chat_postMessage(
            channel=channel,
            blocks=slack_utils.img_block(
                active_threads[channel][thread_ts]["tool_outputs"]["chart_title"],
                active_threads[channel][thread_ts]["tool_outputs"]["chart_url"],
                "chart",
                "Chart showing latest data",
            ),
            thread_ts=thread_ts,
        )

        active_threads[channel][thread_ts]["tool_outputs"]['chart_url'] = None
        active_threads[channel][thread_ts]["tool_outputs"]['chart_title'] = None


@timeit()
def handle_process_error(e: Exception, slack_app, channel: str, thread_ts: str):
    error_message = str()
    general_logger.error(f"Error in process_thread_with_assistant: {e}")

    if isinstance(e, SlackApiError):
        error_message = (
            "There was an issue posting a message to Slack. "
            "If the problem persists, contact support."
        )
    elif isinstance(e, OpenAIError):
        error_message = (
            "The AI agent encountered an error. "
            "Please try your request again. "
            "If the issue continues, please reach out to support for assistance."
        )
    else:
        error_message = (
            "An unexpected internal error occurred. "
            "Please try again. "
            "If the problem persists, please contact support."
        )

    slack_app.client.chat_postMessage(
        channel=channel, text=error_message, thread_ts=thread_ts
    )

# Import flow management classes
from tools.flow_management import FlowManager, FlowExecutor, FlowRetriever

# Initialize flow management objects
flow_manager = FlowManager(sf_conn)
flow_executor = FlowExecutor(sf_conn)
flow_retriever = FlowRetriever(sf_conn)

# Add flow management to your assistant's capabilities
# You'll need to implement the logic to use these in your existing assistant structure


# File: ./context.py

import os
import fnmatch
import sys
from project_tree import generate_project_tree

def build_context(whitelist, blacklist):
    context_output = []
    
    def is_blacklisted(filepath):
        for pattern in blacklist:
            if fnmatch.fnmatch(filepath, pattern) or pattern in filepath:
                return True
        return False
    
    # Generate project tree
    project_tree = generate_project_tree(".", max_depth=3)
    context_output.append("# Project Structure\n\n```\n" + project_tree + "\n```\n\n")
    
    for item in whitelist:
        # Check if the item is a directory
        if os.path.isdir(item):
            # If it's a directory, include all files in that directory
            for dir_root, dir_dirs, dir_files in os.walk(item):
                for dir_file in dir_files:
                    filepath = os.path.join(dir_root, dir_file)
                    if not is_blacklisted(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as file:
                                content = file.read()
                            context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                        except UnicodeDecodeError:
                            try:
                                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                                    content = file.read()
                                context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                            except Exception as e:
                                print(f"Error reading file {filepath}: {e}")
        else:
            # If it's a file pattern, use fnmatch to filter files
            for root, dirs, files in os.walk('.'):
                for filename in fnmatch.filter(files, item):
                    filepath = os.path.join(root, filename)
                    if not is_blacklisted(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as file:
                                content = file.read()
                            context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                        except UnicodeDecodeError:
                            try:
                                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                                    content = file.read()
                                context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                            except Exception as e:
                                print(f"Error reading file {filepath}: {e}")
    
    with open('context_output.md', 'w') as output_file:
        output_file.write(''.join(context_output))
    
    return 'context_output.md'

# Usage example:
if __name__ == "__main__":
    # Hardcoded whitelist
    whitelist = ['*.py', 'README.md', 'tools/',
                 'prompts/', 'utils/']  # Example whitelist patterns
    blacklist = ['*.log', '*.db', 'node_modules/', '__pycache__/']  # Example blacklist patterns
    
    if not whitelist:
        print("Please provide file patterns or directories as arguments, e.g., '*.py' '*.js' 'README.md' 'src/'")
        sys.exit(1)
    
    output_file = build_context(whitelist, blacklist)
    print(f"Context written to {output_file}")

# File: ./project_tree.py

import os

def generate_project_tree(root_dir, max_depth=2):
    tree = []
    
    # Add this list of extensions to ignore
    ignore_extensions = ['.pyc', '.pyo', '.pyd', '.class', '.dll', '.exe', '.so', '.cache']
    ignore_dirs = ['.git']  # Add this line
    
    def walk(directory, depth):
        if depth > max_depth:
            return
        
        items = sorted(os.listdir(directory))
        for item in items:
            path = os.path.join(directory, item)
            
            # Skip .git directory and its contents
            if os.path.isdir(path) and item in ignore_dirs:
                continue
            
            # Skip files with ignored extensions
            if any(item.endswith(ext) for ext in ignore_extensions):
                continue
            
            relative_path = os.path.relpath(path, root_dir)
            indent = "  " * (depth - 1)
            tree.append(f"{indent}{'└── ' if depth > 0 else ''}{item}")
            
            if os.path.isdir(path) and depth < max_depth:
                walk(path, depth + 1)
    
    walk(root_dir, 0)
    return "\n".join(tree)

if __name__ == "__main__":
    root_directory = "."  # Current directory, or specify a different path
    tree = generate_project_tree(root_directory)
    print(tree)

# File: ./tools/generate_chart.py

import requests
import urllib.parse


def generate_chartjs_config(config: str) -> dict:
    """
    Generate a Chart.js configuration.
    """
    return {"config": config}


def get_chart_url_from_config(config: str) -> str:
    """
    Generate a Chart.js chart from a configuration.
    """
    urlencode_config = urllib.parse.quote_plus(config)
    chart_url = f"https://quickchart.io/chart?bkg=white&c={urlencode_config}"

    chart_response = requests.get(chart_url)
    if chart_response.status_code != 200 or not urlencode_config:
        raise Exception("Invalid chartjs config, failed to generate chart")

    return chart_url


# File: ./tools/metadata.py

import logging
from utils.time import timeit

general_logger = logging.getLogger("general")


@timeit()
def deploy_validation_rule(sf_conn, validation_rule):
    result = sf_conn.toolingexecute(
        method="POST", action="sobjects/ValidationRule", json=validation_rule
    )

    if result.get("success"):
        general_logger.info(f"Validation rule created successfully. ID: {result['id']}")
    else:
        general_logger.info(f"Failed to create validation rule. Error: {result}")

    return result


# File: ./tools/data_loader.py

import os
import csv
import requests
from typing import List, Dict
from simple_salesforce import Salesforce


def initiate_data_loading(sf_conn: Salesforce, operation: str, object_name: str, file_url: str):
    """
    Initiate the data loading process for Salesforce.
    This function doesn't actually load the data, but sets up the process
    and returns instructions for the user.
    """
    valid_operations = ["insert", "update", "delete", "upsert"]
    if operation not in valid_operations:
        return {
            "error": f"Invalid operation. Please choose from {', '.join(valid_operations)}"
        }

    if not is_valid_salesforce_object(sf_conn, object_name):
        return {"error": f"Invalid Salesforce object: {object_name}"}

    return process_file(sf_conn, file_url, operation, object_name)


def process_file(sf_conn: Salesforce, file_url: str, operation: str, object_name: str) -> Dict[str, str]:
    """
    Process the uploaded file and perform the Salesforce operation.
    """
    try:
        records = read_csv_from_url(file_url)
        result = perform_salesforce_operation(sf_conn, operation, object_name, records)
        return result
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}


def read_csv_from_url(file_url: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}
    response = requests.get(file_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to download CSV file")
    content = response.content.decode("utf-8")
    csv_reader = csv.DictReader(content.splitlines())
    return list(csv_reader)


def perform_salesforce_operation(
    sf_conn: Salesforce, operation: str, object_name: str, records: List[Dict]
) -> Dict[str, str]:
    try:
        if operation == "insert":
            results = sf_conn.bulk.__getattr__(object_name).insert(records)
        elif operation == "update":
            results = sf_conn.bulk.__getattr__(object_name).update(records)
        elif operation == "delete":
            results = sf_conn.bulk.__getattr__(object_name).delete(records)
        elif operation == "upsert":
            results = sf_conn.bulk.__getattr__(object_name).upsert(records, "Id")
        else:
            return {"error": "Invalid operation"}
        if results and not results[0]["success"]:
            return {"error": results}

        success_count = sum(1 for result in results if result["success"])
        return {
            "message": f"Operation completed. {success_count} out of {len(records)} records processed successfully."
        }
    except Exception as e:
        return {"error": f"Salesforce operation failed: {str(e)}"}


def is_valid_salesforce_object(sf, object_name: str) -> bool:
    sf_objects = sf.describe()["sobjects"]
    return any(obj["name"].lower() == object_name.lower() for obj in sf_objects)


# File: ./tools/soql.py

import logging
from typing import Dict, Any
from simple_salesforce import Salesforce
import cachetools.func
from utils.time import timeit
import json
from simple_salesforce.exceptions import SalesforceError
from urllib.parse import quote

general_logger = logging.getLogger("general")


@timeit()
@cachetools.func.ttl_cache(maxsize=10, ttl=600)
def get_all_sf_tables(sf_conn: Salesforce) -> list:
    general_logger.info("Connecting to Salesforce to fetch all tables...")
    return [sobject["name"] for sobject in sf_conn.describe()["sobjects"]]


@timeit()
@cachetools.func.ttl_cache(maxsize=10, ttl=600)
def get_all_cols_of_sf_table(sf_conn: Salesforce, table_name: str) -> list:
    general_logger.info(
        f"Connecting to Salesforce to fetch cols for table: {table_name}"
    )
    describe = getattr(sf_conn, table_name).describe()
    return [field["name"] for field in describe["fields"]]


def execute_soql_query(sf_conn: Salesforce, soql_query: str) -> Dict[str, Any]:
    general_logger.info(f"Executing SOQL query {soql_query}")
    return sf_conn.query_all(soql_query)


@timeit()
def execute_apex_code(sf_conn: Salesforce, apex_code: str):
    general_logger.info(f"Executing Apex code... {apex_code}")
    
    url_encoded_apex = quote(apex_code)
    
    try:
        result = sf_conn.restful(
            f"tooling/executeAnonymous?anonymousBody={url_encoded_apex}",
            method="GET"
        )
        general_logger.info(f"Apex code execution response: {result}")
        
        if result.get('compiled') and result.get('success'):
            return {"success": True, "message": "Apex code executed successfully", "details": result}
        else:
            return {"success": False, "message": "Apex code execution failed", "details": result}
    
    except SalesforceError as e:
        general_logger.error(f"Salesforce error during Apex execution: {str(e)}")
        return {"error": f"Salesforce error: {str(e)}"}
    except Exception as e:
        general_logger.error(f"Unexpected error during Apex execution: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

# File: ./tools/test_soql.py

import unittest
from unittest.mock import Mock, patch
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceError
from tools.soql import execute_apex_code

class TestExecuteApexCode(unittest.TestCase):

    @patch('tools.soql.general_logger')
    def test_successful_execution(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.return_value = {
            'compiled': True,
            'success': True,
            'compileProblem': None,
            'exceptionMessage': None,
            'line': -1,
            'column': -1,
            'exceptionStackTrace': None,
            'logs': ''
        }

        result = execute_apex_code(mock_sf_conn, "System.debug('Hello, World!');")

        self.assertTrue(result['success'])
        self.assertEqual(result['message'], "Apex code executed successfully")
        mock_sf_conn.restful.assert_called_once()
        mock_logger.info.assert_called()

    @patch('tools.soql.general_logger')
    def test_failed_execution(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.return_value = {
            'compiled': True,
            'success': False,
            'compileProblem': None,
            'exceptionMessage': 'Error message',
            'line': 1,
            'column': 1,
            'exceptionStackTrace': 'Stack trace',
            'logs': ''
        }

        result = execute_apex_code(mock_sf_conn, "Invalid Apex code;")

        self.assertFalse(result['success'])
        self.assertEqual(result['message'], "Apex code execution failed")
        mock_sf_conn.restful.assert_called_once()
        mock_logger.info.assert_called()

    @patch('tools.soql.general_logger')
    def test_salesforce_error(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.side_effect = SalesforceError(
            'Salesforce API Error',
            status=400,
            resource_name='tooling/executeAnonymous',
            content={'error': 'Invalid request'}
        )

        result = execute_apex_code(mock_sf_conn, "System.debug('Hello, World!');")

        self.assertIn('error', result)
        self.assertIn('Salesforce error', result['error'])
        mock_sf_conn.restful.assert_called_once()
        mock_logger.error.assert_called()

    @patch('tools.soql.general_logger')
    def test_unexpected_error(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.side_effect = Exception("Unexpected error")

        result = execute_apex_code(mock_sf_conn, "System.debug('Hello, World!');")

        self.assertIn('error', result)
        self.assertIn('Unexpected error', result['error'])
        mock_sf_conn.restful.assert_called_once()
        mock_logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()

# File: ./tools/flow_management/__init__.py

from .flow_manager import FlowManager
from .flow_executor import FlowExecutor
from .flow_retriever import FlowRetriever

__all__ = ['FlowManager', 'FlowExecutor', 'FlowRetriever']

# File: ./tools/flow_management/flow_retriever.py

import logging
import json

class FlowRetriever:
    def __init__(self, sf_conn):
        self.sf_conn = sf_conn
        self.logger = logging.getLogger(__name__)

    def get_flow_details(self, flow_api_name):
        try:
            result = self.sf_conn.metadata.read('Flow', [flow_api_name])
            self.logger.info(f"Retrieved details for flow '{flow_api_name}'")
            return json.dumps(result, indent=2)
        except Exception as e:
            self.logger.error(f"Error retrieving flow details: {str(e)}")
            raise

    def list_flows(self):
        try:
            query = "SELECT Id, ApiName, Label, ProcessType, Status FROM FlowDefinitionView"
            result = self.sf_conn.query(query)
            self.logger.info("Retrieved list of flows")
            return result['records']
        except Exception as e:
            self.logger.error(f"Error listing flows: {str(e)}")
            raise

# File: ./tools/flow_management/flow_executor.py

import logging
import json

class FlowExecutor:
    def __init__(self, sf_conn):
        self.sf_conn = sf_conn
        self.logger = logging.getLogger(__name__)

    def execute_flow(self, flow_api_name, input_variables):
        try:
            url = f"{self.sf_conn.base_url}services/data/v53.0/actions/custom/flow/{flow_api_name}"
            headers = self.sf_conn.headers
            headers['Content-Type'] = 'application/json'

            payload = {
                "inputs": input_variables
            }

            response = self.sf_conn.session.post(url, headers=headers, data=json.dumps(payload))
            result = response.json()
            self.logger.info(f"Flow '{flow_api_name}' executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing flow: {str(e)}")
            raise

    def process_flow_result(self, result):
        # Process and format the flow execution result
        # This method can be expanded based on specific requirements
        return {
            "success": result.get('isSuccess', False),
            "outputs": result.get('outputValues', {})
        }

# File: ./tools/flow_management/flow_manager.py

import logging
from simple_salesforce import Salesforce
from ..metadata import deploy_validation_rule

class FlowManager:
    def __init__(self, sf_conn: Salesforce):
        self.sf_conn = sf_conn
        self.logger = logging.getLogger(__name__)

    def create_flow(self, flow_metadata):
        try:
            result = self.sf_conn.metadata.create('Flow', flow_metadata)
            if result[0]['success']:
                self.logger.info(f"Flow '{flow_metadata['fullName']}' created successfully")
            else:
                self.logger.error(f"Failed to create flow: {result[0]['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error creating flow: {str(e)}")
            raise

    def update_flow(self, flow_metadata):
        try:
            result = self.sf_conn.metadata.update('Flow', flow_metadata)
            if result[0]['success']:
                self.logger.info(f"Flow '{flow_metadata['fullName']}' updated successfully")
            else:
                self.logger.error(f"Failed to update flow: {result[0]['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error updating flow: {str(e)}")
            raise

    def delete_flow(self, flow_name):
        try:
            result = self.sf_conn.metadata.delete('Flow', flow_name)
            if result[0]['success']:
                self.logger.info(f"Flow '{flow_name}' deleted successfully")
            else:
                self.logger.error(f"Failed to delete flow: {result[0]['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error deleting flow: {str(e)}")
            raise

    def deploy_flow(self, flow_metadata):
        try:
            result = deploy_validation_rule(self.sf_conn, flow_metadata)
            if result['success']:
                self.logger.info(f"Flow '{flow_metadata['fullName']}' deployed successfully")
            else:
                self.logger.error(f"Failed to deploy flow: {result['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error deploying flow: {str(e)}")
            raise

# File: ./utils/auth.py

import requests
from utils.time import timeit

@timeit()
def get_clientell_token():
    url = "https://rev-prod-k8s.clientellone.com/clientell/api/user/login"
    body = {"email": "ruthuparna@getclientell.com", "password": "Clientell@123"}
    response = requests.post(url, json=body)
    return response.json()["access_token"]

@timeit()
def get_salesforce_token(clientell_token):
    url = "https://rev-prod-k8s.clientellone.com/api/salesforce/getAccessToken"
    headers = {"Authorization": f"Token {clientell_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["access_token"]


# File: ./utils/time.py

import time
import functools
import logging

timeit_logger = logging.getLogger("timeit")


def timeit(name=None):
    def args_wrapper(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end = time.perf_counter() - start
                func_name = name or func.__name__
                timeit_logger.info(
                    f"Function: {func_name}, Time: {end:.2f} s"
                )

        return _timeit

    return args_wrapper


# File: ./utils/slack_utils.py

import base64
from io import BytesIO
from typing import Dict


def upload_chart_to_slack(
    slack_app,
    channel: str,
    ts: str,
    base64_image: str,
    filename: str,
    data_summary: Dict[str, any],
):
    """
    Upload a chart image to Slack and post it in the thread with a summary.

    Args:
        slack_app: The Slack app instance
        channel: The Slack channel ID
        ts: The thread timestamp
        base64_image: Base64 encoded image data
        filename: The filename for the chart image
        data_summary: Summary of the data used to generate the chart
    """
    image_data = base64.b64decode(base64_image)
    slack_app.client.files_upload_v2(
        channels=channel,
        thread_ts=ts,
        file=BytesIO(image_data),
        filename=filename,
        title="Generated Chart",
    )
    summary_text = "Chart generated successfully. Data summary:\n" + "\n".join(
        [f"- {key}: {value}" for key, value in data_summary.items()]
    )
    slack_app.client.chat_postMessage(channel=channel, thread_ts=ts, text=summary_text)


def upload_file_to_slack(slack_app, channel: str, ts: str, file_content, filename: str):
    """
    Upload a file to Slack and post it in the thread.

    Args:
        slack_app: The Slack app instance
        channel: The Slack channel ID
        ts: The thread timestamp
        file_content: The content of the file to be uploaded
        filename: The filename for the uploaded file
    """
    slack_app.client.files_upload_v2(
        channels=channel, thread_ts=ts, file=BytesIO(file_content), filename=filename
    )


def img_block(title: str, url: str, block_id: str, alt_text: str) -> list:
    return [
        {
            "type": "image",
            "title": {"type": "plain_text", "text": title},
            "block_id": block_id,
            "image_url": url,
            "alt_text": alt_text,
        }
    ]


# File: ./utils/connection.py

from simple_salesforce import Salesforce
from utils.auth import get_clientell_token, get_salesforce_token
import cachetools.func
from utils.time import timeit

@timeit()
@cachetools.func.ttl_cache(maxsize=1, ttl=600)
def get_salesforce_connection():
    clientell_token = get_clientell_token()
    salesforce_token = get_salesforce_token(clientell_token)
    sf = Salesforce(
        instance_url="https://clientell4-dev-ed.my.salesforce.com",
        session_id=salesforce_token,
    )
    return sf


# File: tools/generate_chart.py

import requests
import urllib.parse


def generate_chartjs_config(config: str) -> dict:
    """
    Generate a Chart.js configuration.
    """
    return {"config": config}


def get_chart_url_from_config(config: str) -> str:
    """
    Generate a Chart.js chart from a configuration.
    """
    urlencode_config = urllib.parse.quote_plus(config)
    chart_url = f"https://quickchart.io/chart?bkg=white&c={urlencode_config}"

    chart_response = requests.get(chart_url)
    if chart_response.status_code != 200 or not urlencode_config:
        raise Exception("Invalid chartjs config, failed to generate chart")

    return chart_url


# File: tools/metadata.py

import logging
from utils.time import timeit

general_logger = logging.getLogger("general")


@timeit()
def deploy_validation_rule(sf_conn, validation_rule):
    result = sf_conn.toolingexecute(
        method="POST", action="sobjects/ValidationRule", json=validation_rule
    )

    if result.get("success"):
        general_logger.info(f"Validation rule created successfully. ID: {result['id']}")
    else:
        general_logger.info(f"Failed to create validation rule. Error: {result}")

    return result


# File: tools/data_loader.py

import os
import csv
import requests
from typing import List, Dict
from simple_salesforce import Salesforce


def initiate_data_loading(sf_conn: Salesforce, operation: str, object_name: str, file_url: str):
    """
    Initiate the data loading process for Salesforce.
    This function doesn't actually load the data, but sets up the process
    and returns instructions for the user.
    """
    valid_operations = ["insert", "update", "delete", "upsert"]
    if operation not in valid_operations:
        return {
            "error": f"Invalid operation. Please choose from {', '.join(valid_operations)}"
        }

    if not is_valid_salesforce_object(sf_conn, object_name):
        return {"error": f"Invalid Salesforce object: {object_name}"}

    return process_file(sf_conn, file_url, operation, object_name)


def process_file(sf_conn: Salesforce, file_url: str, operation: str, object_name: str) -> Dict[str, str]:
    """
    Process the uploaded file and perform the Salesforce operation.
    """
    try:
        records = read_csv_from_url(file_url)
        result = perform_salesforce_operation(sf_conn, operation, object_name, records)
        return result
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}


def read_csv_from_url(file_url: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}
    response = requests.get(file_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to download CSV file")
    content = response.content.decode("utf-8")
    csv_reader = csv.DictReader(content.splitlines())
    return list(csv_reader)


def perform_salesforce_operation(
    sf_conn: Salesforce, operation: str, object_name: str, records: List[Dict]
) -> Dict[str, str]:
    try:
        if operation == "insert":
            results = sf_conn.bulk.__getattr__(object_name).insert(records)
        elif operation == "update":
            results = sf_conn.bulk.__getattr__(object_name).update(records)
        elif operation == "delete":
            results = sf_conn.bulk.__getattr__(object_name).delete(records)
        elif operation == "upsert":
            results = sf_conn.bulk.__getattr__(object_name).upsert(records, "Id")
        else:
            return {"error": "Invalid operation"}
        if results and not results[0]["success"]:
            return {"error": results}

        success_count = sum(1 for result in results if result["success"])
        return {
            "message": f"Operation completed. {success_count} out of {len(records)} records processed successfully."
        }
    except Exception as e:
        return {"error": f"Salesforce operation failed: {str(e)}"}


def is_valid_salesforce_object(sf, object_name: str) -> bool:
    sf_objects = sf.describe()["sobjects"]
    return any(obj["name"].lower() == object_name.lower() for obj in sf_objects)


# File: tools/soql.py

import logging
from typing import Dict, Any
from simple_salesforce import Salesforce
import cachetools.func
from utils.time import timeit
import json
from simple_salesforce.exceptions import SalesforceError
from urllib.parse import quote

general_logger = logging.getLogger("general")


@timeit()
@cachetools.func.ttl_cache(maxsize=10, ttl=600)
def get_all_sf_tables(sf_conn: Salesforce) -> list:
    general_logger.info("Connecting to Salesforce to fetch all tables...")
    return [sobject["name"] for sobject in sf_conn.describe()["sobjects"]]


@timeit()
@cachetools.func.ttl_cache(maxsize=10, ttl=600)
def get_all_cols_of_sf_table(sf_conn: Salesforce, table_name: str) -> list:
    general_logger.info(
        f"Connecting to Salesforce to fetch cols for table: {table_name}"
    )
    describe = getattr(sf_conn, table_name).describe()
    return [field["name"] for field in describe["fields"]]


def execute_soql_query(sf_conn: Salesforce, soql_query: str) -> Dict[str, Any]:
    general_logger.info(f"Executing SOQL query {soql_query}")
    return sf_conn.query_all(soql_query)


@timeit()
def execute_apex_code(sf_conn: Salesforce, apex_code: str):
    general_logger.info(f"Executing Apex code... {apex_code}")
    
    url_encoded_apex = quote(apex_code)
    
    try:
        result = sf_conn.restful(
            f"tooling/executeAnonymous?anonymousBody={url_encoded_apex}",
            method="GET"
        )
        general_logger.info(f"Apex code execution response: {result}")
        
        if result.get('compiled') and result.get('success'):
            return {"success": True, "message": "Apex code executed successfully", "details": result}
        else:
            return {"success": False, "message": "Apex code execution failed", "details": result}
    
    except SalesforceError as e:
        general_logger.error(f"Salesforce error during Apex execution: {str(e)}")
        return {"error": f"Salesforce error: {str(e)}"}
    except Exception as e:
        general_logger.error(f"Unexpected error during Apex execution: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

# File: tools/test_soql.py

import unittest
from unittest.mock import Mock, patch
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceError
from tools.soql import execute_apex_code

class TestExecuteApexCode(unittest.TestCase):

    @patch('tools.soql.general_logger')
    def test_successful_execution(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.return_value = {
            'compiled': True,
            'success': True,
            'compileProblem': None,
            'exceptionMessage': None,
            'line': -1,
            'column': -1,
            'exceptionStackTrace': None,
            'logs': ''
        }

        result = execute_apex_code(mock_sf_conn, "System.debug('Hello, World!');")

        self.assertTrue(result['success'])
        self.assertEqual(result['message'], "Apex code executed successfully")
        mock_sf_conn.restful.assert_called_once()
        mock_logger.info.assert_called()

    @patch('tools.soql.general_logger')
    def test_failed_execution(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.return_value = {
            'compiled': True,
            'success': False,
            'compileProblem': None,
            'exceptionMessage': 'Error message',
            'line': 1,
            'column': 1,
            'exceptionStackTrace': 'Stack trace',
            'logs': ''
        }

        result = execute_apex_code(mock_sf_conn, "Invalid Apex code;")

        self.assertFalse(result['success'])
        self.assertEqual(result['message'], "Apex code execution failed")
        mock_sf_conn.restful.assert_called_once()
        mock_logger.info.assert_called()

    @patch('tools.soql.general_logger')
    def test_salesforce_error(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.side_effect = SalesforceError(
            'Salesforce API Error',
            status=400,
            resource_name='tooling/executeAnonymous',
            content={'error': 'Invalid request'}
        )

        result = execute_apex_code(mock_sf_conn, "System.debug('Hello, World!');")

        self.assertIn('error', result)
        self.assertIn('Salesforce error', result['error'])
        mock_sf_conn.restful.assert_called_once()
        mock_logger.error.assert_called()

    @patch('tools.soql.general_logger')
    def test_unexpected_error(self, mock_logger):
        mock_sf_conn = Mock(spec=Salesforce)
        mock_sf_conn.restful.side_effect = Exception("Unexpected error")

        result = execute_apex_code(mock_sf_conn, "System.debug('Hello, World!');")

        self.assertIn('error', result)
        self.assertIn('Unexpected error', result['error'])
        mock_sf_conn.restful.assert_called_once()
        mock_logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()

# File: tools/flow_management/__init__.py

from .flow_manager import FlowManager
from .flow_executor import FlowExecutor
from .flow_retriever import FlowRetriever

__all__ = ['FlowManager', 'FlowExecutor', 'FlowRetriever']

# File: tools/flow_management/flow_retriever.py

import logging
import json

class FlowRetriever:
    def __init__(self, sf_conn):
        self.sf_conn = sf_conn
        self.logger = logging.getLogger(__name__)

    def get_flow_details(self, flow_api_name):
        try:
            result = self.sf_conn.metadata.read('Flow', [flow_api_name])
            self.logger.info(f"Retrieved details for flow '{flow_api_name}'")
            return json.dumps(result, indent=2)
        except Exception as e:
            self.logger.error(f"Error retrieving flow details: {str(e)}")
            raise

    def list_flows(self):
        try:
            query = "SELECT Id, ApiName, Label, ProcessType, Status FROM FlowDefinitionView"
            result = self.sf_conn.query(query)
            self.logger.info("Retrieved list of flows")
            return result['records']
        except Exception as e:
            self.logger.error(f"Error listing flows: {str(e)}")
            raise

# File: tools/flow_management/flow_executor.py

import logging
import json

class FlowExecutor:
    def __init__(self, sf_conn):
        self.sf_conn = sf_conn
        self.logger = logging.getLogger(__name__)

    def execute_flow(self, flow_api_name, input_variables):
        try:
            url = f"{self.sf_conn.base_url}services/data/v53.0/actions/custom/flow/{flow_api_name}"
            headers = self.sf_conn.headers
            headers['Content-Type'] = 'application/json'

            payload = {
                "inputs": input_variables
            }

            response = self.sf_conn.session.post(url, headers=headers, data=json.dumps(payload))
            result = response.json()
            self.logger.info(f"Flow '{flow_api_name}' executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing flow: {str(e)}")
            raise

    def process_flow_result(self, result):
        # Process and format the flow execution result
        # This method can be expanded based on specific requirements
        return {
            "success": result.get('isSuccess', False),
            "outputs": result.get('outputValues', {})
        }

# File: tools/flow_management/flow_manager.py

import logging
from simple_salesforce import Salesforce
from ..metadata import deploy_validation_rule

class FlowManager:
    def __init__(self, sf_conn: Salesforce):
        self.sf_conn = sf_conn
        self.logger = logging.getLogger(__name__)

    def create_flow(self, flow_metadata):
        try:
            result = self.sf_conn.metadata.create('Flow', flow_metadata)
            if result[0]['success']:
                self.logger.info(f"Flow '{flow_metadata['fullName']}' created successfully")
            else:
                self.logger.error(f"Failed to create flow: {result[0]['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error creating flow: {str(e)}")
            raise

    def update_flow(self, flow_metadata):
        try:
            result = self.sf_conn.metadata.update('Flow', flow_metadata)
            if result[0]['success']:
                self.logger.info(f"Flow '{flow_metadata['fullName']}' updated successfully")
            else:
                self.logger.error(f"Failed to update flow: {result[0]['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error updating flow: {str(e)}")
            raise

    def delete_flow(self, flow_name):
        try:
            result = self.sf_conn.metadata.delete('Flow', flow_name)
            if result[0]['success']:
                self.logger.info(f"Flow '{flow_name}' deleted successfully")
            else:
                self.logger.error(f"Failed to delete flow: {result[0]['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error deleting flow: {str(e)}")
            raise

    def deploy_flow(self, flow_metadata):
        try:
            result = deploy_validation_rule(self.sf_conn, flow_metadata)
            if result['success']:
                self.logger.info(f"Flow '{flow_metadata['fullName']}' deployed successfully")
            else:
                self.logger.error(f"Failed to deploy flow: {result['errors']}")
            return result
        except Exception as e:
            self.logger.error(f"Error deploying flow: {str(e)}")
            raise

# File: prompts/get_filters_new.txt

Based on the Salesforce table {table_name} and its relevant columns 
{relevant_columns}, 
along with the user input described below, 
please provide the SQL WHERE clause conditions in a format suitable for a SOQL query.

- User Input: {user_input}

Output the conditions in this EXACT format:
```
WHERE OwnerId = 'Neil' AND clientell_sf__Location__c = 'New York' AND Industry = 'Healthcare'
```

# File: prompts/generate_code.txt

Generate a JSON formatted string for task (described below) based on the provided parameters. The output should be error-free and accurate such that it can be directly usable in a POST request to execute task in Salesforce.
Ensure all strings are properly quoted and special characters are escaped to avoid JSON formatting errors.

Parameters:
{data}

Output should be a single JSON object with a key 'anonymousBody' for Apex code or 'soqlQuery' for SOQL queries, and its value containing the code or query as a string. Do not include any additional text or explanations.

Example for Apex code:
{{"anonymousBody": "Account[] accounts = [SELECT Id FROM Account WHERE Amount > 1000000]; for(Account acc : accounts) {{ acc.Status = 'Inactive'; }} update accounts;"}}

Example for SOQL query:
{{"soqlQuery": "SELECT Id, Name FROM Account WHERE Status='Active'"}}

IMPORTANT: Avoid using Markdown formatting. 
Specifically, do not wrap the output from generate_final_code in triple backticks
or label it as JSON. 
This formatting is typically used for display 
in Markdown documents but is inappropriate for JSON strings intended 
for functional use in API requests. 
The output from this function will be used DIRECTLY as the input 
for a POST request using the simple_salesforce library, 
without any intermediate assistance.


# File: prompts/user_input.txt

In Accounts update the phone number to '132456789'  where phone number is 123456789


# File: prompts/generate_code_refine.txt

Generate a JSON formatted string for task (described below) based on the provided parameters. The output should be error-free and accurate such that it can be directly usable in a POST request to execute task in Salesforce.
Ensure all strings are properly quoted and special characters are escaped to avoid JSON formatting errors.

Parameters:
- Table Name: {table_name}
- Relevant Columns: {relevant_columns}
- Filter Conditions: {filter_conditions}
- User Input: {user_input}
- Task Name: {task_name}

Output should be a single JSON object with a key 'anonymousBody' for Apex code or 'soqlQuery' for SOQL queries, and its value containing the code or query as a string. Do not include any additional text or explanations.

Example for Apex code:
{{"anonymousBody": "Account[] accounts = [SELECT Id FROM Account WHERE Amount > 1000000]; for(Account acc : accounts) {{ acc.Status = 'Inactive'; }} update accounts;"}}

Example for SOQL query:
{{"soqlQuery": "SELECT Id, Name FROM Account WHERE Status='Active'"}}

IMPORTANT: Avoid using Markdown formatting. 
Specifically, do not wrap the output from generate_final_code in triple backticks
or label it as JSON. 
This formatting is typically used for display 
in Markdown documents but is inappropriate for JSON strings intended 
for functional use in API requests. 
The output from this function will be used DIRECTLY as the input 
for a POST request using the simple_salesforce library, 
without any intermediate assistance.


# File: prompts/code_debugger.txt

Generate a JSON formatted string for an Apex code / SOQL query execution based on the provided parameters, including details from a previous error to improve the code. 

The output should be directly usable in a POST request to execute anonymous Apex code in Salesforce. Ensure all strings are properly quoted and special characters are escaped to avoid JSON formatting errors.

Parameters:
- Table Name: {table_name}
- Relevant Columns: {relevant_columns}
- Filter Conditions: {filter_conditions}
- Previously Generated Apex Code: {previously_generated_apex_code}
- Error Trace: {error_trace}
- User Input: {user_input}

Output should be a single JSON object with a key 'anonymousBody' and its value containing the corrected Apex code as a string. Do not include any additional text or explanations.

Example:
{{"anonymousBody": "SELECT Id, Name FROM Account WHERE Status='Active' AND Region='EMEA'"}}

IMPORTANT: Avoid using Markdown formatting. Specifically, do not wrap the output from generate_final_code in triple backticks or label it as JSON. This formatting is typically used for display in Markdown documents but is inappropriate for JSON strings intended for functional use in API requests. The output from this function will be used DIRECTLY as the input for a POST request using the simple_salesforce library, without any intermediate assistance.

# File: prompts/get_tables.txt

Based on the list of Salesforce tables derived from the system directly 
and the user input described below, 
please provide the exact name of the table the user is referring to in a single word format.

- List of all Salesforce tables: {list_of_salesforce_tables}
- User Input: {user_input}


# File: prompts/identify_task.txt

based on the given user_input (described below)
identify what type of salesforce query is the user trying to solve
and what is the task they want to solve it with.

list of available tasks 
- apex_code_generation
- soql_generation

Note: When the user wants to mutate/modify/delete things, 
you need to choose apex_code_generation 
When the user simply wants to fetch/query/select/view data use soql_generation


user_input = {user_input}

response format should include the task_name as the output

examples:
[
    {{
        "user_input": "How many accounts have stage Closed?",
        "task_name": "soql_generation"
    }},
    {{
        "user_input": "Update all opportunities with amount greater than 100000",
        "task_name": "apex_code_generation"
    }},
    {{
        "user_input": "What is the current temperature in San Francisco?",
        "task_name": "Fallback"
    }},
    {{
        "user_input": "Tell me a joke.",
        "task_name": "Fallback"
    }}
]


# File: prompts/generate_code_old.txt

Generate a JSON formatted string for task (described below) based on the provided parameters. The output should be error-free and accurate such that it can be directly usable in a POST request to execute task in Salesforce.
Ensure all strings are properly quoted and special characters are escaped to avoid JSON formatting errors.

Parameters:
- Table Name: {table_name}
- Relevant Columns: {relevant_columns}
- Filter Conditions: {filter_conditions}
- User Input: {user_input}
- Task Name: {task_name}

Output should be a single JSON object with a key 'anonymousBody' for Apex code or 'soqlQuery' for SOQL queries, and its value containing the code or query as a string. Do not include any additional text or explanations.

Example for Apex code:
{{"anonymousBody": "Account[] accounts = [SELECT Id FROM Account WHERE Amount > 1000000]; for(Account acc : accounts) {{ acc.Status = 'Inactive'; }} update accounts;"}}

Example for SOQL query:
{{"soqlQuery": "SELECT Id, Name FROM Account WHERE Status='Active'"}}

IMPORTANT: Avoid using Markdown formatting. 
Specifically, do not wrap the output from generate_final_code in triple backticks
or label it as JSON. 
This formatting is typically used for display 
in Markdown documents but is inappropriate for JSON strings intended 
for functional use in API requests. 
The output from this function will be used DIRECTLY as the input 
for a POST request using the simple_salesforce library, 
without any intermediate assistance.


# File: prompts/get_columns.txt

Based on the given salesforce table : {table_name} 
and the columns of the table directly derived from system : {column_list}
and the user input described below
- User Input: {user_input}

can you output which columns the user is talking about WITHOUT ANY EXPLANATION
in this EXACT format : 
['column_name_1', 'column_name2', ... 'column_name_n']





# File: prompts/get_filters.txt

Based on the Salesforce table {table_name} and its relevant columns 
{relevant_columns}, 
along with the user input described below, 
please provide the SOQL filter conditions in a format suitable for a SOQL query.

- User Input: {user_input}




# File: prompts/plan_data_loader.md

- first fetch determistically  method of getting table name , columns ,

- don't use any other method to get the table name , columns
- 

table -> columns -> soql generation - fetch csv -> feed to pandas 
-> do manipulation -> output csv -> upload to salesforce using function data loader


- 

# File: prompts/flow_management/modify_flow.txt

You are assisting a user in modifying an existing Salesforce Flow. Follow these steps:

1. Ask for the name of the flow they want to modify.
2. Use the FlowRetriever to get the current flow details.
3. Present a summary of the current flow structure to the user.
4. Ask what changes they want to make. This could include:
   - Adding new elements (screens, decisions, actions)
   - Modifying existing elements
   - Changing flow logic
   - Updating flow metadata (name, description, etc.)

5. Guide the user through making these changes, updating the flow metadata structure as needed.
6. Once all changes are made, confirm with the user and use the FlowManager to update the flow.

Remember to provide clear explanations of each step and potential implications of their changes. If they request a change that could break existing functionality, warn them and suggest alternatives if possible.

# File: prompts/flow_management/execute_flow.txt

You are assisting a user in executing a Salesforce Flow. Follow these steps:

1. Ask for the name of the flow they want to execute.
2. Use the FlowRetriever to get the flow details and identify required input variables.
3. Ask the user to provide values for each required input variable.
4. Once all inputs are collected, use the FlowExecutor to run the flow.
5. Process and present the results of the flow execution to the user.

Here's an example dialogue:

# File: prompts/flow_management/create_flow.txt

You are assisting a user in creating a new Salesforce Flow. Please guide them through the process by asking for the following information:

1. Flow Name: Ask for a unique name for the flow.
2. Flow Type: Ask whether it should be a Screen Flow or Autolaunched Flow.
3. Description: Ask for a brief description of the flow's purpose.
4. Start Element: Ask what the first element of the flow should be (e.g., Screen, Decision, Action).

Based on their responses, create a flow metadata structure. Here's an example structure:

{
  "Metadata": {
    "activeVersionNumber": 1,
    "description": "[User's description]",
    "label": "[Flow Name]",
    "processType": "[Screen or Autolaunched]"
  },
  "FullName": "[Flow Name without spaces]"
}

After creating the basic structure, ask if they want to add any specific elements to the flow, such as screens, decisions, or actions. Guide them through adding these elements based on their responses.

# File: utils/auth.py

import requests
from utils.time import timeit

@timeit()
def get_clientell_token():
    url = "https://rev-prod-k8s.clientellone.com/clientell/api/user/login"
    body = {"email": "ruthuparna@getclientell.com", "password": "Clientell@123"}
    response = requests.post(url, json=body)
    return response.json()["access_token"]

@timeit()
def get_salesforce_token(clientell_token):
    url = "https://rev-prod-k8s.clientellone.com/api/salesforce/getAccessToken"
    headers = {"Authorization": f"Token {clientell_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["access_token"]


# File: utils/time.py

import time
import functools
import logging

timeit_logger = logging.getLogger("timeit")


def timeit(name=None):
    def args_wrapper(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end = time.perf_counter() - start
                func_name = name or func.__name__
                timeit_logger.info(
                    f"Function: {func_name}, Time: {end:.2f} s"
                )

        return _timeit

    return args_wrapper


# File: utils/slack_utils.py

import base64
from io import BytesIO
from typing import Dict


def upload_chart_to_slack(
    slack_app,
    channel: str,
    ts: str,
    base64_image: str,
    filename: str,
    data_summary: Dict[str, any],
):
    """
    Upload a chart image to Slack and post it in the thread with a summary.

    Args:
        slack_app: The Slack app instance
        channel: The Slack channel ID
        ts: The thread timestamp
        base64_image: Base64 encoded image data
        filename: The filename for the chart image
        data_summary: Summary of the data used to generate the chart
    """
    image_data = base64.b64decode(base64_image)
    slack_app.client.files_upload_v2(
        channels=channel,
        thread_ts=ts,
        file=BytesIO(image_data),
        filename=filename,
        title="Generated Chart",
    )
    summary_text = "Chart generated successfully. Data summary:\n" + "\n".join(
        [f"- {key}: {value}" for key, value in data_summary.items()]
    )
    slack_app.client.chat_postMessage(channel=channel, thread_ts=ts, text=summary_text)


def upload_file_to_slack(slack_app, channel: str, ts: str, file_content, filename: str):
    """
    Upload a file to Slack and post it in the thread.

    Args:
        slack_app: The Slack app instance
        channel: The Slack channel ID
        ts: The thread timestamp
        file_content: The content of the file to be uploaded
        filename: The filename for the uploaded file
    """
    slack_app.client.files_upload_v2(
        channels=channel, thread_ts=ts, file=BytesIO(file_content), filename=filename
    )


def img_block(title: str, url: str, block_id: str, alt_text: str) -> list:
    return [
        {
            "type": "image",
            "title": {"type": "plain_text", "text": title},
            "block_id": block_id,
            "image_url": url,
            "alt_text": alt_text,
        }
    ]


# File: utils/connection.py

from simple_salesforce import Salesforce
from utils.auth import get_clientell_token, get_salesforce_token
import cachetools.func
from utils.time import timeit

@timeit()
@cachetools.func.ttl_cache(maxsize=1, ttl=600)
def get_salesforce_connection():
    clientell_token = get_clientell_token()
    salesforce_token = get_salesforce_token(clientell_token)
    sf = Salesforce(
        instance_url="https://clientell4-dev-ed.my.salesforce.com",
        session_id=salesforce_token,
    )
    return sf


