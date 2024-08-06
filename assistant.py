from typing import Dict, Any
import json
import re
from utils.groq_client import GroqClient
from utils.connection import get_salesforce_connection
from tools.custom_tools import CustomTools
from utils.config_loader import load_config

class SalesforceAssistant:
    def __init__(self):
        self.groq_client = GroqClient()
        self.sf_connection = get_salesforce_connection()
        self.custom_tools = CustomTools(self.sf_connection)
        self.config = load_config()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        with open('prompts/system_prompt.txt', 'r') as file:
            return file.read()

    def process_user_query(self, user_query: str, thread_ts: str, channel: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.groq_client.generate_response(messages)
        
        # Process the response and execute tools if necessary
        final_response = self._process_tool_calls(response, user_query, thread_ts, channel)
        
        return final_response

    def _process_tool_calls(self, response: str, user_query: str, thread_ts: str, channel: str) -> str:
        function_pattern = r'<function=(\w+)>(.*?)</function>'
        matches = re.findall(function_pattern, response)

        if matches:
            for tool_name, params_str in matches:
                try:
                    # Try to parse params as JSON
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to evaluate as a Python dict
                    try:
                        params = eval(params_str)
                    except:
                        # If both methods fail, return an error message
                        return f"Error: Unable to parse parameters for function {tool_name}"

                tool_result = getattr(self.custom_tools, tool_name)(**params)
                
                if "error" in tool_result:
                    # If there's an error, send it back to the model for processing
                    error_response = self.groq_client.generate_response([
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": response},
                        {"role": "system", "content": f"The previous action resulted in an error: {tool_result['message']}. Please acknowledge this error and suggest a solution or alternative approach."}
                    ])
                    # Recursively process the new response in case it makes another tool call
                    return self._process_tool_calls(error_response, user_query, thread_ts, channel)
                else:
                    # If successful, generate final response using the tool result
                    final_response = self.groq_client.generate_response([
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": response},
                        {"role": "system", "content": f"The action was successful. Here is the result: {tool_result}. Please provide a helpful response based on this information."}
                    ])
                    return final_response
        else:
            return response

    def handle_slack_event(self, event: Dict[str, Any]) -> str:
        user_query = event['text']
        thread_ts = event.get('thread_ts', event['ts'])
        channel = event['channel']
        
        response = self.process_user_query(user_query, thread_ts, channel)
        return response