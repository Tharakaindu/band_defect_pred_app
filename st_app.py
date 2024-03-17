import streamlit as st

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular

## Load Data
file_path = '/content/drive/MyDrive/CWD Band Quality Prediction/Band Data/Band Data 8X15 Only - Selected data.csv'
df = pd.read_csv(file_path)

feature_names = df.columns.tolist()
feature_names.remove("Defect")  

# Create data and targets
data = df[feature_names].values  
targets = df["Defect"].values 

target_names = df["Defect"].unique()

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data, targets, train_size=0.8, random_state=123)


## Load Model
rf_classif = load("rf_classif.model")

Y_test_preds = rf_classif.predict(X_test)

## Dashboard
st.title("Band Defects :red[Prediction] :bar_chart: :chart_with_upwards_trend:")
st.markdown("Predict Band Defects")

tab1, tab2, tab3 = st.tabs(["Data :clipboard:", "Global Performance :weight_lifter:", "Local Performance :bicyclist:"])

with tab1:
    st.header("Band Dataset")
    st.write(df)

    # Summary of the table
    st.subheader("Summary")
    total_rows = len(df)
    good_quality_count = (df["Defect"] == "Good_Product").sum()
    welding_crack_count = (df["Defect"] == "Welding_Crack").sum()
    st.write(f"Total data rows: {total_rows}")
    st.write(f"Predicted as Good Quality: {good_quality_count}")
    st.write(f"Predicted as Welding Crack Defects: {welding_crack_count}")

    # Bar chart
    st.subheader("Predictions")
    prediction_counts = df["Defect"].value_counts()
    st.bar_chart(prediction_counts)

with tab2:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6, 6))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, ax=ax1, normalize=True)
        st.pyplot(conf_mat_fig, use_container_width=True)

    with col2:
        feat_imp_fig, ax2 = plt.subplots(figsize=(8, 6))  # Increase figsize for better visualization
        
        # Calculate feature importances from the trained random forest classifier
        feature_importances = rf_classif.feature_importances_
        
        # Get the top 10 feature importances and their corresponding feature names
        top_10_indices = np.argsort(feature_importances)[::-1][:10]
        top_10_features = np.array(feature_names)[top_10_indices]
        top_10_importances = feature_importances[top_10_indices]

        # Plot feature importances as bar chart
        ax2.bar(top_10_features, top_10_importances, color='skyblue')
        ax2.set_title('Top 10 Feature Importances')
        ax2.set_ylabel('Importance')
        ax2.set_xlabel('Feature')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        st.pyplot(feat_imp_fig, use_container_width=True)

    # Metric for model accuracy
    accuracy = accuracy_score(Y_test, Y_test_preds)
    st.subheader("Model Accuracy")
    st.markdown("<h2 style='color: green;'>{:.2f}%</h2>".format(accuracy * 100), unsafe_allow_html=True)

    st.divider()
    st.header(" Classification Report")

    # Get classification report as a string
    report = classification_report(Y_test, Y_test_preds, output_dict=True)

    # Create a DataFrame from the classification report
    report_df = pd.DataFrame(report).transpose()

    # Rename the columns
    report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]

    # Display the DataFrame
    st.write(report_df)

with tab3:
    st.header("Predictions on New Data")

    # Option to load new dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)

        # Drop any existing prediction column if present
        if "Prediction" in new_df.columns:
            new_df.drop(columns=["Prediction"], inplace=True)

        # Predict using the model
        new_predictions = rf_classif.predict(new_df[feature_names])
        new_probs = rf_classif.predict_proba(new_df[feature_names])

        # Convert prediction to human-readable labels
        new_df["Prediction"] = np.where(new_predictions == "Welding_Crack", "Welding Crack", "Good Product")

        # Summary table
        st.subheader("Summary")
        st.write("Total data rows:", len(new_df))
        st.write("Predicted as Good Quality:", (new_df["Prediction"] == "Good Product").sum())
        st.write("Predicted as Welding Crack Defects:", (new_df["Prediction"] == "Welding Crack").sum())

        # Bar chart
        st.subheader("Predictions")
        prediction_counts = new_df["Prediction"].value_counts()
        st.bar_chart(prediction_counts)

        # Display DataFrame with predictions
        st.subheader("Data with Predictions")
        st.write(new_df)
    else:
        st.write("Please upload a CSV file to make predictions.")
