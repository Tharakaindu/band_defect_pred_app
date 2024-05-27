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

# Prediction with error handling and data cleaning (optional)
try:
  # Data cleaning before prediction (optional)
  # You can add specific data cleaning steps based on your data
  X_test_cleaned = X_test.copy()  # Create a copy to avoid modifying original data
  for col in X_test_cleaned.columns:
    if X_test_cleaned[col].dtype == object:  # Check for string data type
      # Consider encoding categorical features here (e.g., one-hot encoding)
      # ...
      pass

  # Prediction on cleaned data
  Y_test_preds = xgb_classif.predict(X_test_cleaned)
except ValueError as e:
  if 'could not convert string to float' in str(e):
    st.warning("Encountered string values during prediction. Consider data cleaning (e.g., encoding categorical features).")
    # ... (consider alternative actions, e.g., remove rows, log the error)
  else:
    raise e  # Re-raise other ValueErrors

from sklearn.preprocessing import LabelEncoder

# ... (rest of your code for user input, feature importance, etc.)
# User Input for Prediction
st.subheader("Predict Band Defect")

user_input = {}
for feature_name in feature_names:
  user_input[feature_name] = st.text_input(feature_name)

user_data = pd.DataFrame([user_input])

# Data Cleaning for User Input (optional)
# You might need to perform similar cleaning as in the model loading section

# Prediction
try:
  # Data cleaning for user input (optional)
  user_data_cleaned = user_data.copy()  # Create a copy to avoid modifying original data
  for col in user_data_cleaned.columns:
    if user_data_cleaned[col].dtype == object:  # Check for string data type
      # Consider encoding categorical features here (e.g., one-hot encoding)
      # ...
      pass

  # Prediction on cleaned user data
  user_prediction = xgb_classif.predict(user_data_cleaned)[0]
  prediction_proba = xgb_classif.predict_proba(user_data_cleaned)[0][1] * 100
except ValueError as e:
  if 'could not convert string to float' in str(e):
    st.warning("Encountered string values during prediction. Please enter numeric values or appropriate data for categorical features.")
  else:
    raise e  # Re-raise other ValueErrors

# Display Prediction Results
if 'user_prediction' in locals():
  if user_prediction == "Welding_Crack":
    st.error(f"The model predicts a **welding crack** with {prediction_proba:.2f}% probability.")
  else:
    st.success(f"The model predicts **no welding crack** with {prediction_proba:.2f}% probability.")

# Feature Importance (optional)
 explainer = lime_tabular.LimeTabularExplainer(
     data=X_train,
     feature_names=feature_names,
     class_names=target_names,
     random_state=1
 )
 
 exp = explainer.explain_instance(user_data_cleaned.iloc[0], xgb_classif.predict_proba, num_features=5)

 # Plot feature importances (optional)
 if 'exp' in locals():
   fig = exp.as_pyplot_figure(figsize=(10, 6))
   st.pyplot(fig, use_container_width=True)
