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