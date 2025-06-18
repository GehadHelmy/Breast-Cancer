import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import joblib

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
.highlight {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def load_data():
    """Load and preprocess the breast cancer dataset"""
    # Replace this with your actual data loading method
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='diagnosis')
    return X, y

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train SVM and Neural Network models"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM Model
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]
    
    # Neural Network Model
    nn = MLPClassifier(hidden_layer_sizes=(15,), max_iter=1000, random_state=42)
    nn.fit(X_train_scaled, y_train)
    nn_pred = nn.predict(X_test_scaled)
    nn_pred_proba = nn.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'svm': {
            'model': svm,
            'predictions': svm_pred,
            'probabilities': svm_pred_proba
        },
        'nn': {
            'model': nn,
            'predictions': nn_pred,
            'probabilities': nn_pred_proba
        },
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }

def plot_confusion_matrix(y_true, y_pred, title):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def plot_roc_curve(y_true, y_pred_proba, title):
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return plt

# Streamlit App
def run_app():
    # Title and introduction
    st.title("🩺 Breast Cancer Classification Dashboard")
    st.markdown("""
    ## Machine Learning Model for Breast Cancer Diagnosis
    This interactive dashboard demonstrates machine learning models 
    for classifying breast cancer tumors as benign or malignant.
    """)

    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train models
    trained_models = train_models(X_train, X_test, y_train, y_test)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the App Mode", 
        [
            "Home", 
            "Dataset Overview", 
            "Model Performance", 
            "Prediction"
        ]
    )

    # Home Page
    if app_mode == "Home":
        st.markdown("""
        ### Welcome to Breast Cancer Classification App
        
        #### What this App Does:
        - Classify breast tumors as benign or malignant
        - Compare SVM and Neural Network models
        - Explore dataset characteristics
        - Make individual predictions
        
        #### Key Models:
        - Support Vector Machine (SVM)
        - Multi-Layer Perceptron Neural Network
        """)
        
        # Quick model performance summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SVM Accuracy", f"{accuracy_score(y_test, trained_models['svm']['predictions']):.2%}")
        with col2:
            st.metric("Neural Network Accuracy", f"{accuracy_score(y_test, trained_models['nn']['predictions']):.2%}")

    # Dataset Overview
    elif app_mode == "Dataset Overview":
        st.header("Dataset Characteristics")
        
        # Basic dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", len(X))
            st.metric("Features", X.shape[1])
        with col2:
            class_dist = y.value_counts(normalize=True)
            st.metric("Benign (%)", f"{class_dist[0]:.2%}")
            st.metric("Malignant (%)", f"{class_dist[1]:.2%}")
        
        # Feature distribution
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox(
            "Select a Feature to Visualize", 
            X.columns.tolist()
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=X, x=selected_feature, hue=y, multiple="stack", ax=ax)
        st.pyplot(fig)

    # Model Performance
    elif app_mode == "Model Performance":
        st.header("Model Performance Comparison")
        
        # Performance metrics
        st.subheader("Classification Metrics")
        metrics_tab, cm_tab, roc_tab = st.tabs(
            ["Metrics", "Confusion Matrices", "ROC Curves"]
        )
        
        with metrics_tab:
            # Create columns for metrics
            svm_metrics = classification_report(
                y_test, 
                trained_models['svm']['predictions'], 
                output_dict=True
            )
            nn_metrics = classification_report(
                y_test, 
                trained_models['nn']['predictions'], 
                output_dict=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### SVM Metrics")
                st.metric("Accuracy", f"{svm_metrics['accuracy']:.2%}")
                st.metric("Precision", f"{svm_metrics['weighted avg']['precision']:.2%}")
                st.metric("Recall", f"{svm_metrics['weighted avg']['recall']:.2%}")
            
            with col2:
                st.markdown("#### Neural Network Metrics")
                st.metric("Accuracy", f"{nn_metrics['accuracy']:.2%}")
                st.metric("Precision", f"{nn_metrics['weighted avg']['precision']:.2%}")
                st.metric("Recall", f"{nn_metrics['weighted avg']['recall']:.2%}")
        
        with cm_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### SVM Confusion Matrix")
                svm_cm_plot = plot_confusion_matrix(
                    y_test, 
                    trained_models['svm']['predictions'], 
                    "SVM Confusion Matrix"
                )
                st.pyplot(svm_cm_plot)
            
            with col2:
                st.markdown("#### Neural Network Confusion Matrix")
                nn_cm_plot = plot_confusion_matrix(
                    y_test, 
                    trained_models['nn']['predictions'], 
                    "Neural Network Confusion Matrix"
                )
                st.pyplot(nn_cm_plot)
        
        with roc_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### SVM ROC Curve")
                svm_roc_plot = plot_roc_curve(
                    y_test, 
                    trained_models['svm']['probabilities'], 
                    "SVM ROC Curve"
                )
                st.pyplot(svm_roc_plot)
            
            with col2:
                st.markdown("#### Neural Network ROC Curve")
                nn_roc_plot = plot_roc_curve(
                    y_test, 
                    trained_models['nn']['probabilities'], 
                    "Neural Network ROC Curve"
                )
                st.pyplot(nn_roc_plot)

    # Prediction Page
    elif app_mode == "Prediction":
        st.header("Individual Tumor Classification")
        
        # Retrieve feature names from trained models
        all_features = trained_models['feature_names']
        
        # Create input fields for ALL features
        input_data = {}
        for feature in all_features:
            input_data[feature] = st.number_input(
                f"Enter {feature}", 
                float(X[feature].min()), 
                float(X[feature].max()), 
                float(X[feature].mean())
            )
        
        # Prepare input for prediction
        if st.button("Predict Tumor Classification"):
            # Convert input to DataFrame with ALL features
            input_df = pd.DataFrame([input_data])
            
            # Ensure the input DataFrame has the exact same columns as training data
            input_df = input_df.reindex(columns=all_features, fill_value=0)
            
            # Standardize input
            input_scaled = trained_models['scaler'].transform(input_df)
            
            # Predict using both models
            svm_pred = trained_models['svm']['model'].predict(input_scaled)[0]
            nn_pred = trained_models['nn']['model'].predict(input_scaled)[0]
            
            # Prediction probabilities
            svm_prob = trained_models['svm']['model'].predict_proba(input_scaled)[0]
            nn_prob = trained_models['nn']['model'].predict_proba(input_scaled)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### SVM Prediction")
                st.metric("Classification", 
                    "Malignant" if svm_pred == 1 else "Benign"
                )
                st.metric("Probability", f"{max(svm_prob):.2%}")
            
            with col2:
                st.markdown("#### Neural Network Prediction")
                st.metric("Classification", 
                    "Malignant" if nn_pred == 1 else "Benign"
                )
                st.metric("Probability", f"{max(nn_prob):.2%}")

# Run the app
def main():
    run_app()

if __name__ == "__main__":
    main()