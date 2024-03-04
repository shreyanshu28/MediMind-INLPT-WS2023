import os
import glob

def get_project_directories():
    """
    Retrieve the project and data directories.

    This function finds the project directory as the parent directory of the current working directory
    and constructs the data directory path by appending 'data/PubMed_Format' to the project directory.

    Returns:
        tuple: A tuple containing the project directory and data directory paths.
    """
    project_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(project_dir, 'data', 'PubMed_Format')
    return project_dir, data_dir

def read_files(file_pattern):
    """
    Generator function to read files matching a specific pattern.

    Iterates over all files matching the given pattern, opens each file in read mode with UTF-8 encoding,
    and yields the file path and content.

    Parameters:
        file_pattern (str): The glob pattern to match files.

    Yields:
        tuple: A tuple containing the file path and file content as a string.
    """
    for file_path in glob.glob(file_pattern):
        with open(file_path, 'r', encoding="utf-8") as file:
            yield file_path, file.read()