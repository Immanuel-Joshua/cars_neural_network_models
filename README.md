# cars_neural_network_models
 This repository contains Python code to clean and preprocess a dataset of car attributes, followed by the creation and training of a neural network and an LSTM model to predict the car model.

## This code contains a UK cars sales dataset taken from Kaggle and consisting of different folders combined for the purpose analysis

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Immanuel-Joshua/cars_neural_network_models.git
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure the dataset `combined_dataset.csv` is in the same directory as the script.
2. Run the script:
    ```bash
    python main.py
    ```

## Data Cleaning and Preparation

- The dataset is loaded using `pandas`.
- Duplicate rows are removed.
- Columns are combined and missing values are handled.
- Numerical columns are converted to the appropriate data types.

## Feature Encoding and Scaling

- Categorical features are one-hot encoded using `OneHotEncoder`.
- Numerical features are scaled using `StandardScaler`.

## Neural Network Model

A neural network with dense layers is created and trained on the preprocessed data.

## LSTM Model

An LSTM model is created to handle sequential data.

## Bidirectional LSTM Model

A Bidirectional LSTM model is used for better performance in sequence prediction.

## Results

- The models are evaluated using cross-validation and on a test set.
- Accuracy and loss are plotted for training and validation.


## Acknowledgements

This project uses the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `keras`