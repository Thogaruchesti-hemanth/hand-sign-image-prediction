# Sign Language Digits Recognition

A deep learning project to recognise sign language digits (0-9) using convolutional neural networks.

## Project Overview

This project implements a CNN model to classify hand sign images representing digits 0-9 from the Sign Language Digits Dataset. The model achieves over 95% accuracy on the test set.

## Dataset

The dataset consists of 64×64 grayscale images of hand signs representing digits 0-9. The dataset is stored in .npy format:
- X.npy: Images array
- Y.npy: Labels array

Source: [Sign Language Digits Dataset on Kaggle](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset)

## Project Structure

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

Total parameters: 164,618

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

Thogaruchesti Hemanth

```
