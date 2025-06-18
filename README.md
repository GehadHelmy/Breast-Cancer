# Breast Cancer Awareness App

A Streamlit-based web application designed to raise breast cancer awareness, promote early detection, and provide data-driven insights using the Breast Cancer Wisconsin (Diagnostic) Dataset. Built by Gehad Helmy and Alaa Adel, this app combines machine learning with educational resources to empower users in understanding and managing breast cancer risks.

## Features
- **Data Analysis**: Visualizes tumor characteristics (e.g., radius, texture) using Pandas, Matplotlib, and Seaborn, with feature selection via RFE.
- **Tumor Classification**: Predicts benign/malignant tumors using trained SVM and Neural Network models (Scikit-learn), achieving high accuracy on selected features.
- **Educational Resources**: Offers evidence-based information on breast cancer types, prevention, and self-examination techniques.
- **Interactive UI**: Streamlit-powered interface with a pink-themed design, navigation for Home, Analysis, Prevention, Self-Check, Prediction, and About pages.
- **Responsive Design**: Custom CSS for a user-friendly experience, including motivational quotes and a fixed footer.

## Tech Stack
- **Python**: Core language for data processing and model training.
- **Streamlit**: Web app framework for interactive UI.
- **Scikit-learn**: For RFE, SVM, and Neural Network models.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization (histograms, heatmaps, ROC curves).
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset (569 samples, 30 features).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/GehadHelmy/Breast-Cancer.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Navigate through sections to explore data insights, prevention tips, or self-exam guides.
- Use the Prediction page to input feature values and classify tumors (for educational purposes only).
- Consult a healthcare provider for medical advice.

## Disclaimer
This app is for educational purposes only and does not replace professional medical advice. Always consult a qualified healthcare provider for diagnosis or treatment.

## Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, feature enhancements, or documentation improvements.

## Contact
For feedback or inquiries, reach out at contactContact us on LinkedIn  [Gehad Helmy](https://www.linkedin.com/in/gehad-helmy-505445296/) and  [Alaa Adel](http://linkedin.com/in/alaa-adel-64735034b)
        .

---

Developed by Gehad Helmy & Alaa Adel | © 2025
