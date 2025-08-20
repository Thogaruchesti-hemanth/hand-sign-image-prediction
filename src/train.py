import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint

from data_loader import load_data
from model import create_model
from utils import ensure_dir

def train_model(epochs=20, batch_size=250):
    """
    Train the CNN model on the sign language digits dataset.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        history: Training history object
    """
    # Load data
    X_train, X_test, Y_train, Y_test = load_data()
    
    # Create model
    model = create_model()
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    ensure_dir('../models/')
    checkpoint = ModelCheckpoint(
        '../models/best_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        callbacks=[checkpoint]
    )
    
    # Save the final model
    model.save('../models/final_model.h5')
    
    return history, model

def plot_training_history(history):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history object
    """
    # Create results directory if it doesn't exist
    ensure_dir('../results/')
    
    # Convert the history to a DataFrame
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['loss'], label='Training Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('../results/training_curves.png')
    plt.show()

if __name__ == '__main__':
    # Train the model
    history, model = train_model(epochs=20, batch_size=250)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    X_train, X_test, Y_train, Y_test = load_data()
    scores = model.evaluate(X_test, Y_test)
    print(f'Test loss: {scores[0]:.4f}')
    print(f'Test accuracy: {scores[1]:.4f}')
