#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Cleaning up..."
    jobs_to_kill=$(jobs -p)
    if [ ! -z "$jobs_to_kill" ]; then
        kill $jobs_to_kill
    fi
    exit 0
}

# Trap EXIT signal to ensure cleanup
trap cleanup EXIT

# Load environment variables
source .env.local

# Debugging: Print the SLACK_BOT_TOKEN to ensure it's loaded
echo "SLACK_BOT_TOKEN: $SLACK_BOT_TOKEN"

# Start your local server using Python
echo "Starting local server..."

# Activate Python virtual environment (if you're using one)
# source ~/myvenv/bin/activate

# Run the server
python server.py > server.log 2>&1 &

if [ $? -ne 0 ]; then
    echo "Failed to start the server. Check server.log for details."
else
    echo "Llama 3.1 Salesforce Assistant server is running."
fi

# Keep the script running
wait