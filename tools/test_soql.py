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