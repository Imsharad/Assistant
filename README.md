# Salesforce Assistant with LLaMa 3.1 Tool Usage

## Table of Contents
- [Salesforce Assistant with LLaMa 3.1 Tool Usage](#salesforce-assistant-with-llama-31-tool-usage)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [How It Works](#how-it-works)
  - [Tool Usage Highlight](#tool-usage-highlight)
  - [Tool Usage Highlight](#tool-usage-highlight-1)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [Deployment](#deployment)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Overview

This project implements a Salesforce Assistant powered by LLaMa 3.1, featuring advanced tool usage capabilities. The assistant integrates with Slack and provides a range of Salesforce-related functionalities, including SOQL queries, data manipulation, and Apex code execution.

## Key Features

- **LLaMa 3.1 Integration**: Utilizes the LLaMa 3.1 model for natural language understanding and generation.
- **Advanced Tool Usage**: Implements a custom tool calling format compatible with LLaMa 3.1, enabling seamless interaction with Salesforce operations.
- **Salesforce Operations**: Supports various Salesforce tasks such as querying data, executing SOQL, managing metadata, and running Apex code.
- **Slack Integration**: Provides a user-friendly interface through Slack for interacting with the Salesforce Assistant.
- **Chart Generation**: Capability to generate and visualize data using Chart.js.
- **Data Loading**: Supports bulk data operations (insert, update, delete, upsert) in Salesforce.
- **Flow Management**: Includes tools for creating, updating, and executing Salesforce flows.

## How It Works

This Salesforce Assistant leverages the power of LLaMa 3.1 and custom tool integrations to provide a seamless experience for Salesforce operations. Here's an overview of the system architecture and workflow:

1. **User Input**: Users interact with the assistant through a Slack interface, sending natural language queries or commands.

2. **LLaMa 3.1 Processing**: The user's input is processed by the LLaMa 3.1 model, which understands the intent and identifies the appropriate Salesforce operation to perform.

3. **Tool Selection**: Based on the interpreted intent, the system selects the appropriate tool or function to execute the Salesforce operation.

4. **Tool Execution**: The selected tool interacts with the Salesforce API to perform the requested operation, such as querying data, updating records, or executing Apex code.

5. **Result Processing**: The output from the Salesforce operation is processed and formatted for user consumption.

6. **Response Generation**: LLaMa 3.1 generates a natural language response based on the operation results, providing a human-friendly explanation or summary.

7. **Slack Output**: The final response is sent back to the user through the Slack interface.

This architecture allows for flexible expansion of functionalities by adding new tools or enhancing existing ones, all while maintaining a consistent and intuitive user experience.

## Tool Usage Highlight

The core feature of this project is its implementation of tool usage with LLaMa 3.1. This allows the assistant to:

1. Interpret user queries and identify the appropriate Salesforce operation.
2. Generate the correct tool call format expected by LLaMa 3.1.
3. Execute the corresponding Salesforce operation using custom tools.
4. Process the results and provide a natural language response to the user.

Example tool call format:
- **Flow Management**: Includes tools for creating, updating, and executing Salesforce flows.

## Tool Usage Highlight

The core feature of this project is its implementation of tool usage with LLaMa 3.1. This allows the assistant to:

1. Interpret user queries and identify the appropriate Salesforce operation.
2. Generate the correct tool call format expected by LLaMa 3.1.
3. Execute the corresponding Salesforce operation using custom tools.
4. Process the results and provide a natural language response to the user.

Example tool call format:```
<function=get_all_sf_tables>{}</function>
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone [repository_url]
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the required API keys and tokens

4. Run the setup script:
   ```
   ./scripts/setup.sh
   ```

## Usage

Start the Salesforce Assistant server:
```
python server.py
```

Interact with the assistant through your configured Slack workspace.

## Configuration

- Adjust tool configurations in `config/tools_config.yaml`
- Modify logging settings in `config/logging_config.yaml`
- Customize prompts in the `prompts/` directory

## Contributing

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Deployment

For deployment instructions, refer to [DEPLOYMENT.md](docs/DEPLOYMENT.md).

## License

This project is licensed under the [LICENSE NAME] - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- LLaMa 3.1 model by Meta AI
- Salesforce for their API and documentation
- Slack for their bot integration capabilities