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