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