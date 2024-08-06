import os
import fnmatch
import sys
from project_tree import generate_project_tree

def build_context(whitelist, blacklist):
    context_output = []
    
    def is_blacklisted(filepath):
        for pattern in blacklist:
            if fnmatch.fnmatch(filepath, pattern) or pattern in filepath:
                return True
        return False
    
    # Generate project tree
    project_tree = generate_project_tree(".", max_depth=3)
    context_output.append("# Project Structure\n\n```\n" + project_tree + "\n```\n\n")
    
    for item in whitelist:
        # Check if the item is a directory
        if os.path.isdir(item):
            # If it's a directory, include all files in that directory
            for dir_root, dir_dirs, dir_files in os.walk(item):
                for dir_file in dir_files:
                    filepath = os.path.join(dir_root, dir_file)
                    if not is_blacklisted(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as file:
                                content = file.read()
                            context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                        except UnicodeDecodeError:
                            try:
                                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                                    content = file.read()
                                context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                            except Exception as e:
                                print(f"Error reading file {filepath}: {e}")
        else:
            # If it's a file pattern, use fnmatch to filter files
            for root, dirs, files in os.walk('.'):
                for filename in fnmatch.filter(files, item):
                    filepath = os.path.join(root, filename)
                    if not is_blacklisted(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as file:
                                content = file.read()
                            context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                        except UnicodeDecodeError:
                            try:
                                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                                    content = file.read()
                                context_output.append(f"# File: {filepath}\n\n{content}\n\n")
                            except Exception as e:
                                print(f"Error reading file {filepath}: {e}")
    
    with open('context_output.md', 'w') as output_file:
        output_file.write(''.join(context_output))
    
    return 'context_output.md'

# Usage example:
if __name__ == "__main__":
    # Hardcoded whitelist
    whitelist = [
        '*.py',  # All Python files
        '*.md',  # All Markdown files (including README.md)
        '*.txt',  # All text files (including requirements.txt)
        '*.yaml', '*.yml',  # All YAML files (including config files and docker-compose.yml)
        '*.sh',  # All shell scripts (including setup.sh)
        'Dockerfile',  # Dockerfile
        '.env.example',  # Environment variables example file
        '.gitignore',  # Git ignore file
        'prompts/',  # All files in the prompts directory
        'utils/',  # All files in the utils directory
        'config/',  # All files in the config directory
        'tools/',  # All files in the tools directory
        'scripts/',  # All files in the scripts directory (if it exists)
        'tests/',  # All files in the tests directory (if it exists)
        'docs/',  # All files in the docs directory (if it exists)
    ]  # Comprehensive whitelist for LLM context
    blacklist = ['*.log', '*.db', 'node_modules/', '__pycache__/']  # Example blacklist patterns
    
    if not whitelist:
        print("Please provide file patterns or directories as arguments, e.g., '*.py' '*.js' 'README.md' 'src/'")
        sys.exit(1)
    
    output_file = build_context(whitelist, blacklist)
    print(f"Context written to {output_file}")