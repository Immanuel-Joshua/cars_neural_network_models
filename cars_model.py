"""Import all necessary packages"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras import models, layers, optimizers
from keras.utils import to_categorical
from keras.layers import Bidirectional
# Import necessary libraries for LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense


"""Data Cleaning and preparation"""
df = pd.read_csv('combined_dataset.csv')
print(df.head())

# Investigate columns in dataset
for column in df.columns:
    print(f"Unique values in column {column}:")
    print(df[column].unique())

# Clear all duplicate rows
duplicate_rows = df.duplicated()
print(duplicate_rows.sum())
df = df.drop_duplicates()

# Change the datatype of certain columns before combining
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['engineSize'] = pd.to_numeric(df['engineSize'], errors='coerce')
df['engine size2'] = pd.to_numeric(df['engine size2'], errors='coerce')
df['engine size'] = pd.to_numeric(df['engine size'], errors='coerce')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['mileage2'] = pd.to_numeric(df['mileage2'], errors='coerce')

# Combine similar columns
df['all_fuel type'] = df['fuelType'].fillna('') + df['fuel type'].fillna('') + df['fuel type2'].fillna('')
df['all_tax'] = df['tax'].fillna(0) + df['tax(£)'].fillna(0)
df['all_engineSize'] = df['engineSize'].fillna(0) + df['engine size2'].fillna(0) + df['engine size'].fillna(0)
df['all_mileage'] = df['mileage'].fillna(0) + df['mileage2'].fillna(0)

# Drop the original columns
df.drop(['fuelType', 'fuel type', 'fuel type2', 'tax', 'tax(£)', 'engineSize', 'engine size2', 'mileage', 'mileage2', 'engine size'], axis=1, inplace=True)

# Check for missing values after combining
missing_values = df.isnull().sum()
print("Missing values after combining columns:\n", missing_values)

# Check column datatypes
print("Column datatypes:\n", df.dtypes)
print(df.head())

# The use of the column 'reference' is unknown and it possesses too many missing values and will be dropped
df.drop(['reference'], axis=1, inplace=True)

# Fill missing values for numeric columns
df.fillna({'year': df['year'].median(),
           'price': df['price'].mean(),
           'mpg': df['mpg'].mean()}, inplace=True)

# Fill missing values for categorical columns
df.fillna({'model': df['model'].mode()[0],
           'transmission': df['transmission'].mode()[0]}, inplace=True)

# Verify if there are any remaining missing values
missing_values_after_imputation = df.isnull().sum()
print("Missing values after imputation:\n", missing_values_after_imputation)

"""Encode and scale categorical and numerical columns"""
# Define columns 
categorical_cols = ['model', 'transmission', 'all_fuel type']  
numeric_cols = ['year', 'price', 'all_tax', 'all_engineSize', 'all_mileage', 'mpg']

# Categorical features
categorical_encoder = OneHotEncoder(sparse_output=False)
categorical_features = pd.DataFrame(categorical_encoder.fit_transform(df[categorical_cols]))

# Numerical features
numerical_scaler = StandardScaler()
numerical_features = pd.DataFrame(numerical_scaler.fit_transform(df[numeric_cols]))

# Combine features
df_encoded_scaled = pd.concat([categorical_features, numerical_features], axis=1)
print(df_encoded_scaled.head())

total_unique_values = df[categorical_cols].nunique().sum()
print(f"Total unique values across categorical features: {total_unique_values}")

# Encode target variable as one-hot
target = df['model']
target_encoded = pd.get_dummies(target).values

"""Begin the creation and training of the neural network"""
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Split the data into train and test sets
X = df_encoded_scaled.to_numpy()
y = target_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the cross-validation procedure
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform k-fold cross-validation
cv_scores = []
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

# Train the final model on the entire training set
model = create_model(input_shape=X_train.shape[1], num_classes=y_train.shape[1])
final_history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_val_fold, y_val_fold))
# Evaluate the model
val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=1)
cv_scores.append(val_acc)
    
print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (np.mean(cv_scores)*100, np.std(cv_scores)*100))

# Evaluate the final model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot accuracy and loss curves
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_history(final_history)


"""LSTM model"""
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(32, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Reshape the data for LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_val_lstm = X_val_fold.reshape(X_val_fold.shape[0], 1, X_val_fold.shape[1])

# Create and train the LSTM model
lstm_model = create_lstm_model(input_shape=X_train_lstm.shape, num_classes=y_train.shape[1])
lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_val_lstm, y_val_fold))

# Evaluate the LSTM model
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=1)
print(f"LSTM Model Accuracy: {lstm_acc:.4f}")

# Plot LSTM accuracy and loss curves
plot_history(lstm_history)

def create_bidirectional_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(input_shape[1], input_shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Reshape the data for LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Assuming that the validation data needs to be reshaped similarly
X_val_lstm = X_val_fold.reshape(X_val_fold.shape[0], 1, X_val_fold.shape[1])

# Create and train the Bidirectional LSTM model
bidirectional_lstm_model = create_bidirectional_lstm_model(input_shape=X_train_lstm.shape, num_classes=y_train.shape[1])
bidirectional_lstm_history = bidirectional_lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(X_val_lstm, y_val_fold))

# Evaluate the Bidirectional LSTM model
bidirectional_lstm_loss, bidirectional_lstm_acc = bidirectional_lstm_model.evaluate(X_test_lstm, y_test, verbose=1)
print(f"Bidirectional LSTM Model Accuracy: {bidirectional_lstm_acc:.4f}")

# Plot Bidirectional LSTM accuracy and loss curves
plot_history(bidirectional_lstm_history)