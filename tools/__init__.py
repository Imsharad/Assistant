from typing import Dict, Any
from simple_salesforce import Salesforce
from tools.soql import execute_soql_query, get_all_sf_tables, get_all_cols_of_sf_table
from tools.generate_chart import generate_chartjs_config_and_chart
from tools.data_loader import initiate_data_loading
from tools.metadata import deploy_validation_rule

class CustomTools:
    def __init__(self, sf_connection: Salesforce):
        self.sf_connection = sf_connection

    def get_all_sf_tables(self) -> list:
        return get_all_sf_tables(self.sf_connection)

    def get_all_cols_of_sf_table(self, table_name: str) -> list:
        return get_all_cols_of_sf_table(self.sf_connection, table_name)

    def execute_soql_query(self, soql_query: str) -> Dict[str, Any]:
        return execute_soql_query(self.sf_connection, soql_query)

    def generate_chartjs_config_and_chart(self, config: str) -> tuple[dict, str]:
        return generate_chartjs_config_and_chart(config)

    def initiate_data_loading(self, operation: str, object_name: str, file_url: str) -> Dict[str, Any]:
        return initiate_data_loading(self.sf_connection, operation, object_name, file_url)

    def deploy_validation_rule(self, validation_rule: Dict[str, Any]) -> Dict[str, Any]:
        return deploy_validation_rule(self.sf_connection, validation_rule)

    def execute_apex_code(self, apex_code: str) -> Dict[str, Any]:
        # Implement Apex code execution logic here
        # You may need to add this function to the soql.py file if it doesn't exist
        pass