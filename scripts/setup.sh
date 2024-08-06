#!/bin/bash

# Create main project directory
mkdir -p llama

# Create subdirectories
mkdir -p llama/config
mkdir -p llama/prompts
mkdir -p llama/tools
mkdir -p llama/utils
mkdir -p llama/tests/test_tools
mkdir -p llama/tests/test_utils
mkdir -p llama/scripts
mkdir -p llama/docs

# Create main script
touch llama/main.py

# Create configuration files
touch llama/config/tools_config.yaml
touch llama/config/logging_config.yaml

# Create prompt files
touch llama/prompts/system_prompt.txt
touch llama/prompts/tool_use_prompt.txt
touch llama/prompts/error_prompt.txt
touch llama/prompts/clarification_prompt.txt
touch llama/prompts/salesforce_context_prompt.txt
touch llama/prompts/multi_tool_prompt.txt

# Create tool files
touch llama/tools/__init__.py
touch llama/tools/soql.py
touch llama/tools/generate_chart.py
touch llama/tools/data_loader.py
touch llama/tools/metadata.py
touch llama/tools/apex_executor.py
touch llama/tools/tool_utils.py

# Create utility files
touch llama/utils/__init__.py
touch llama/utils/salesforce_connection.py
touch llama/utils/groq_client.py
touch llama/utils/slack_utils.py
touch llama/utils/error_handling.py
touch llama/utils/config_loader.py

# Create script files
touch llama/scripts/setup_environment.sh
touch llama/scripts/run_tests.sh

# Create documentation files
touch llama/docs/API.md
touch llama/docs/CONTRIBUTING.md
touch llama/docs/DEPLOYMENT.md

# Create other necessary files
touch llama/.env.example
touch llama/.gitignore
touch llama/requirements.txt
touch llama/setup.py
touch llama/Dockerfile
touch llama/docker-compose.yml
touch llama/README.md

echo "Project structure for Llama has been created!"