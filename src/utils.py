import os

def ensure_dir(directory):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
