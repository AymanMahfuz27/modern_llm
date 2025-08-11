#script that creates a tree of directories and files along with what is in each file and makes sure the path to the file is corect 

import os

def print_tree_with_contents(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Calculate the depth for pretty printing
        rel_path = os.path.relpath(dirpath, root_dir)
        depth = 0 if rel_path == '.' else rel_path.count(os.sep) + 1
        indent = '    ' * depth
        # Print directory name
        if rel_path == '.':
            print(os.path.basename(os.path.abspath(root_dir)) + '/')
        else:
            print('    ' * (depth - 1) + os.path.basename(dirpath) + '/')
        # Print files and their first line (or a message if empty)
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            print(f"{indent}{filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().rstrip('\n')
                    if first_line:
                        print(f"{indent}    {first_line}")
                    else:
                        print(f"{indent}    [empty file]")
            except Exception as e:
                print(f"{indent}    [could not read file: {e}]")

if __name__ == "__main__":
    # You can change '.' to any directory you want to start from
    print_tree_with_contents('src')

