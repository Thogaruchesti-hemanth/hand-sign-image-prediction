import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path='../data/', test_size=0.15, random_state=42):
    """
    Load and preprocess the sign language digits dataset.
    
    Args:
        data_path (str): Path to the data directory
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, Y_train, Y_test
    """
    # Load the data
    X = np.load(f'{data_path}X.npy')
    Y = np.load(f'{data_path}Y.npy')
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    # Reshape and normalise
    X_train = X_train.reshape(-1, 64, 64, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 64, 64, 1).astype('float32') / 255.0
    
    return X_train, X_test, Y_train, Y_test
