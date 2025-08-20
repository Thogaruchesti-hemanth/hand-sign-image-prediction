# Sign Language Digits Recognition 👋🔢

A deep learning model that accurately classifies hand sign images representing digits 0-9 using convolutional neural networks. Achieves over 95% accuracy on the Sign Language Digits Dataset.

## 🌟 Features

- CNN architecture with 4 convolutional layers
- Data preprocessing and normalization
- Comprehensive training and evaluation pipeline
- Visualization of training metrics and confusion matrix
- Model checkpointing and saving

## 📊 Results

- **Training accuracy**: 95.26%
- **Test accuracy**: 95.81%
- **Precision**: 96% (macro average)
- **Recall**: 96% (macro average)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/sign-language-digits-recognition.git
cd sign-language-digits-recognition

# Install dependencies
pip install -r requirements.txt

# Download dataset (place X.npy and Y.npy in data folder)
# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py
```

##📁 Project Structure

```
sign-language-digits-recognition/
├── data/           # Dataset files
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for exploration
├── models/         # Saved models
├── results/        # Output visualizations and reports
└── scripts/        # Utility scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/sign-language-digits-recognition.git
cd sign-language-digits-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset (if not already present):
```bash
python scripts/download_data.py
```

## Usage

1. Train the model:
```bash
python src/train.py
```

2. Evaluate the model:
```bash
python src/evaluate.py
```

3. Explore the data and model in the Jupyter notebook:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Results

The model achieves the following performance:
- Training accuracy: 95.26%
- Test accuracy: 95.81%

See the results directory for training curves, confusion matrix, and classification report.

## Model Architecture

The CNN architecture consists of:
- 4 convolutional layers with max pooling and dropout
- 2 fully connected layers
- Output layer with softmax activation
- Total parameters: 164,618

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

Thogaruchesti Hemanth
