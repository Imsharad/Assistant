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
