import json
from typing import Dict, Any, List
from simple_salesforce import Salesforce
import requests
import urllib.parse
import logging
class CustomTools:
    def __init__(self, sf_connection: Salesforce):
        self.sf_connection = sf_connection
        self.logger = logging.getLogger(__name__)

    def _format_error(self, error_message: str) -> Dict[str, Any]:
        return {
            "error": True,
            "message": error_message
        }

    def get_all_sf_tables(self) -> Dict[str, Any]:
        """Fetches all available Salesforce table names."""
        try:
            tables = self.sf_connection.describe()["sobjects"]
            return {"tables": [table["name"] for table in tables]}
        except Exception as e:
            self.logger.error(f"Error fetching Salesforce tables: {str(e)}")
            return self._format_error(f"Error fetching Salesforce tables: {str(e)}")

    def get_all_cols_of_sf_table(self, table_name: str) -> Dict[str, Any]:
        """Fetches all columns available for the given Salesforce table name."""
        try:
            describe = getattr(self.sf_connection, table_name).describe()
            return {"columns": [field["name"] for field in describe["fields"]]}
        except Exception as e:
            self.logger.error(f"Error fetching columns for table {table_name}: {str(e)}")
            return self._format_error(f"Error fetching columns for table {table_name}: {str(e)}")

    def execute_soql_query(self, soql_query: str) -> Dict[str, Any]:
        """Executes a Salesforce SOQL query."""
        try:
            self.logger.info(f"Executing SOQL query: {soql_query}")
            return self.sf_connection.query_all(soql_query)
        except Exception as e:
            self.logger.error(f"Error executing SOQL query: {str(e)}")
            return self._format_error(f"Error executing SOQL query: {str(e)}")

    def generate_chartjs_config_and_chart(self, config: str, title: str) -> Dict[str, Any]:
        """Generates Chart.js configuration and chart, and returns the configuration in JSON format."""
        try:
            chart_config = json.loads(config)
            chart_config["options"] = chart_config.get("options", {})
            chart_config["options"]["title"] = {"display": True, "text": title}
            
            encoded_config = urllib.parse.quote(json.dumps(chart_config))
            chart_url = f"https://quickchart.io/chart?c={encoded_config}"
            
            return {
                "config": chart_config,
                "url": chart_url
            }
        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}")
            return self._format_error(f"Error generating chart: {str(e)}")

    def initiate_data_loading(self, operation: str, object_name: str, file_url: str) -> Dict[str, Any]:
        """Initiates the data loading process for Salesforce."""
        try:
            valid_operations = ["insert", "update", "delete", "upsert"]
            if operation not in valid_operations:
                return self._format_error(f"Invalid operation. Please choose from {', '.join(valid_operations)}")

            # Download the CSV file
            response = requests.get(file_url)
            if response.status_code != 200:
                return self._format_error("Failed to download the file")

            # Process the CSV content (simplified for this example)
            records = [line.split(',') for line in response.text.split('\n') if line]
            headers = records[0]
            data = [dict(zip(headers, record)) for record in records[1:]]

            # Perform the Salesforce operation
            if operation == "insert":
                result = self.sf_connection.bulk.__getattr__(object_name).insert(data)
            elif operation == "update":
                result = self.sf_connection.bulk.__getattr__(object_name).update(data)
            elif operation == "delete":
                result = self.sf_connection.bulk.__getattr__(object_name).delete(data)
            elif operation == "upsert":
                result = self.sf_connection.bulk.__getattr__(object_name).upsert(data, 'Id')

            success_count = sum(1 for r in result if r['success'])
            return {
                "message": f"Operation completed. {success_count} out of {len(data)} records processed successfully."
            }
        except Exception as e:
            self.logger.error(f"Error in data loading: {str(e)}")
            return self._format_error(f"Error in data loading: {str(e)}")

    def deploy_validation_rule(self, validation_rule: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a validation rule to Salesforce."""
        try:
            result = self.sf_connection.toolingexecute(
                method="POST",
                action="sobjects/ValidationRule",
                json=validation_rule
            )
            if result.get("success"):
                self.logger.info(f"Validation rule created successfully. ID: {result['id']}")
            else:
                self.logger.error(f"Failed to create validation rule. Error: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error deploying validation rule: {str(e)}")
            return self._format_error(f"Error deploying validation rule: {str(e)}")

    def execute_apex_code(self, apex_code: str) -> Dict[str, Any]:
        """Executes a Salesforce Apex code."""
        try:
            result = self.sf_connection.restful(
                f"tooling/executeAnonymous?anonymousBody={urllib.parse.quote(apex_code)}",
                method="GET"
            )
            if result.get('compiled') and result.get('success'):
                return {"success": True, "message": "Apex code executed successfully", "details": result}
            else:
                return {"success": False, "message": "Apex code execution failed", "details": result}
        except Exception as e:
            self.logger.error(f"Error executing Apex code: {str(e)}")
            return self._format_error(f"Error executing Apex code: {str(e)}")

def get_tool_descriptions() -> List[Dict[str, Any]]:
    """Returns a list of tool descriptions in the format expected by Llama 3.1."""
    return [
        {
            "name": "get_all_sf_tables",
            "description": "Fetches all the available Salesforce table names.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_all_cols_of_sf_table",
            "description": "Fetches all the columns available for the given Salesforce table name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "The name of the Salesforce table."}
                },
                "required": ["table_name"]
            }
        },
        {
            "name": "execute_soql_query",
            "description": "Executes a Salesforce SOQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "soql_query": {"type": "string", "description": "The SOQL query to execute."}
                },
                "required": ["soql_query"]
            }
        },
        {
            "name": "generate_chartjs_config_and_chart",
            "description": "Generates Chart.js configuration and chart, and returns the configuration in JSON format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "config": {"type": "string", "description": "The Chart.js configuration."},
                    "title": {"type": "string", "description": "The title of the chart."}
                },
                "required": ["config", "title"]
            }
        },
        {
            "name": "initiate_data_loading",
            "description": "Initiates the data loading process for Salesforce.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["insert", "update", "delete", "upsert"], "description": "The operation to perform."},
                    "object_name": {"type": "string", "description": "The name of the Salesforce object."},
                    "file_url": {"type": "string", "description": "The URL of the file to load."}
                },
                "required": ["operation", "object_name", "file_url"]
            }
        },
        {
            "name": "deploy_validation_rule",
            "description": "Deploy a validation rule to Salesforce.",
            "parameters": {
                "type": "object",
                "properties": {
                    "validation_rule": {"type": "object", "description": "Validation rule object"}
                },
                "required": ["validation_rule"]
            }
        },
        {
            "name": "execute_apex_code",
            "description": "Executes a Salesforce Apex code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "apex_code": {"type": "string", "description": "The Apex code to execute."}
                },
                "required": ["apex_code"]
            }
        }
    ]