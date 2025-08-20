import os
import kaggle
from utils import ensure_dir

def download_dataset():
    """
    Download the Sign Language Digits Dataset from Kaggle.
    """
    # Create data directory if it doesn't exist
    ensure_dir('../data/')
    
    # Download dataset
    kaggle.api.dataset_download_files(
        'ardamavi/sign-language-digits-dataset',
        path='../data/',
        unzip=True
    )
    
    print("Dataset downloaded successfully!")

if __name__ == '__main__':
    download_dataset()
