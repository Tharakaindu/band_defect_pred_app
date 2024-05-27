import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from lime import lime_tabular

## Load Data
file_path = 'Data_final (2).csv'
try:
  df = pd.read_csv(file_path)
except FileNotFoundError:
  st.error(f"Error: Data file '{file_path}' not found. Please upload the data file.")
  exit()

feature_names = df.columns.tolist()
feature_names.remove("iDefect")

# Create data and targets (with data cleaning)
data = df[feature_names].apply(lambda x: pd.to_numeric(x, errors='coerce'))  # Try converting to numeric, handle errors with 'coerce'
targets = df["iDefect"].values

target_names = df["iDefect"].unique()

# Option 1: Check and handle missing values in targets
# if np.isnan(targets).any():
#   # Handle missing values in targets (e.g., remove rows with missing targets)
#   targets = targets.dropna()

# Option 2: Drop rows with missing values together
data_with_targets = pd.concat([df[feature_names], df["iDefect"]], axis=1)
data_with_targets = data_with_targets.dropna(axis=0)

data = data_with_targets[feature_names]
targets = data_with_targets["iDefect"]

# ... rest of your code for splitting data, training model, etc.

# Split data into train and test sets (consider handling missing values if necessary)
X_train, X_test, Y_train, Y_test = train_test_split(data.dropna(), targets, train_size=0.8, random_state=123)

## Load Model
try:
  xgb_classif = load("Welding_crack_xgb.joblib")
except FileNotFoundError:
  st.error(f"Error: Trained model 'Welding_crack_xgb.joblib' not found. Please ensure the model file is uploaded or trained beforehand.")
  exit()

# Prediction with error handling (optional, data cleaning is preferred)
try:
  Y_test_preds = xgb_classif.predict(X_test)
except ValueError as e:
  if 'could not convert string to float' in str(e):
    st.warning("Encountered string values during prediction. Consider data cleaning (e.g., encoding categorical features).")
    # ... (consider alternative actions, e.g., remove rows, log the error)
  else:
    raise e  # Re-raise other ValueErrors

from sklearn.preprocessing import LabelEncoder

# Identify and handle non-numeric features (replace with your approach)
le = LabelEncoder()
for col in X_test.columns:
  if X_test[col].dtype == object:  # Check for string data type
    X_test[col] = le.fit_transform(X_test[col])

# Now predict using the cleaned data
Y_test_preds = xgb_classif.predict(X_test)

probability_predictions = xgb_classif.predict_proba(X_test)

# Apply a threshold to convert probabilities to class labels
threshold = 0.5
y_pred = np.where(probability_predictions[:, 1] > threshold, "Welding_Crack", "Good")  # Assuming second column is positive class

# Assuming you have trained your model (rf_classif) and split data (X_test, Y_test)

# Predict labels for test data
Y_test_preds = xgb_classif.predict(X_test)

# Plot the confusion matrix
conf_mat_fig = plt.figure(figsize=(6, 6))
ax1 = conf_mat_fig.add_subplot(111)
skplt.metrics.plot_confusion_matrix(Y_test, y_pred, ax=ax1, normalize=True)
st.pyplot(conf_mat_fig, use_container_width=True)

## Dashboard
st.title("Band Defects
