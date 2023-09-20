import os
import sys

# Get the directory path from the environment variable
directory_path = os.environ.get('DIRECTORY_PATH')

# Check if the directory path is provided
if directory_path is None:
    print("Please set the DIRECTORY_PATH environment variable.", file=sys.stderr)
    sys.exit(1)

# Check if the directory exists
if not os.path.exists(directory_path):
    print(f"The directory '{directory_path}' does not exist.", file=sys.stderr)
    sys.exit(1)

# Check if 'model.pkl' file exists in the directory
model_pkl_path = os.path.join(directory_path, 'model.pkl')
if not os.path.isfile(model_pkl_path):
    print(f"'model.pkl' file not found in '{directory_path}'.", file=sys.stderr)
    sys.exit(1)

# Print a success message
print("'model.pkl' file found in '{directory_path}'.")
