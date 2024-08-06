import os

def generate_project_tree(root_dir, max_depth=2):
    tree = []
    
    # Add this list of extensions to ignore
    ignore_extensions = ['.pyc', '.pyo', '.pyd', '.class', '.dll', '.exe', '.so', '.cache']
    ignore_dirs = ['.git']  # Add this line
    
    def walk(directory, depth):
        if depth > max_depth:
            return
        
        items = sorted(os.listdir(directory))
        for item in items:
            path = os.path.join(directory, item)
            
            # Skip .git directory and its contents
            if os.path.isdir(path) and item in ignore_dirs:
                continue
            
            # Skip files with ignored extensions
            if any(item.endswith(ext) for ext in ignore_extensions):
                continue
            
            relative_path = os.path.relpath(path, root_dir)
            indent = "  " * (depth - 1)
            tree.append(f"{indent}{'└── ' if depth > 0 else ''}{item}")
            
            if os.path.isdir(path) and depth < max_depth:
                walk(path, depth + 1)
    
    walk(root_dir, 0)
    return "\n".join(tree)

if __name__ == "__main__":
    root_directory = "."  # Current directory, or specify a different path
    tree = generate_project_tree(root_directory)
    print(tree)