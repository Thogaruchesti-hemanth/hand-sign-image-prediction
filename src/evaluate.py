import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_data
from model import create_model
from utils import ensure_dir

def evaluate_model(model_path='../models/final_model.h5'):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path (str): Path to the trained model
    """
    # Load data
    X_train, X_test, Y_train, Y_test = load_data()
    
    # Load model
    from keras.models import load_model
    model = load_model(model_path)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Ensure results directory exists
    ensure_dir('../results/')
    plt.savefig('../results/confusion_matrix.png')
    plt.show()
    
    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=[str(i) for i in range(10)])
    print("Classification Report:")
    print(report)
    
    # Save classification report
    with open('../results/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Calculate and print final metrics
    scores = model.evaluate(X_test, Y_test)
    print(f'Test loss: {scores[0]:.4f}')
    print(f'Test accuracy: {scores[1]:.4f}')

if __name__ == '__main__':
    evaluate_model()
