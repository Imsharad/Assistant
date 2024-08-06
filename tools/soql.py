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