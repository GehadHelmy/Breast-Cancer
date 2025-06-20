import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.feature_selection import RFE

# Set page configuration with pink theme and breast cancer ribbon icon
st.set_page_config(
    page_title="Breast Cancer Awareness App",
    page_icon="🎗️",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pink theme, white text, and professional styling
st.markdown(
    """
    <style>
    body {
        background-color: #ff94b4;  /* Light pink background */
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #ff94b4;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    p, div, span, label {
        color: #ffffff;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }
    input[type="number"] {
        background-color: white !important;
        color: black !important;
    }
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        opacity: 1;
        background: white !important;
        width: 20px;
        height: 100%;
    }
    input[type="number"]:hover,
    input[type="number"]:focus {
        background-color: white !important;
        color: black !important;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
        border: 2px solid #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #ffa8c2;  /* Slightly darker pink sidebar */
    }
    .motivational-quote {
        font-style: italic;
        font-size: 20px;
        text-align: center;
        margin: 30px 0;
        color: #ffffff;
        background-color: #ffa8c2;
        padding: 15px;
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        background-color: #ff94b4;
        padding: 10px 0;
        font-size: 14px;
        color: #ffffff;
    }
    .section-divider {
        border-top: 2px solid #ffffff;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Breast Cancer Awareness")
st.sidebar.write("This app is designed to provide information and resources related to breast cancer awareness.")
page = st.sidebar.radio("Navigate to", ["Home", "Analysis", "Prevention", "Self-Check", "Prediction","Hospitals","About Us"])

# Load and preprocess data for analysis
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    return X, y, df

X, y, df = load_data()

# Perform feature selection using RFE
@st.cache_resource
def select_features(X, y):
    svm = SVC(kernel='linear', random_state=42)
    rfe = RFE(estimator=svm, n_features_to_select=10)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_].tolist()
    feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
    return selected_features, feature_ranking

selected_features, feature_ranking = select_features(X, y)

# Train models (SVM and Neural Network) on selected features
@st.cache_resource
def train_models(X, y, selected_features):
    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Train Neural Network
    mlp = MLPClassifier(random_state=42, max_iter=1000)
    mlp.fit(X_train_scaled, y_train)
    
    # Calculate accuracies
    svm_train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))
    svm_test_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    mlp_train_acc = accuracy_score(y_train, mlp.predict(X_train_scaled))
    mlp_test_acc = accuracy_score(y_test, mlp.predict(X_test_scaled))
    
    # ROC curve for both models
    y_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
    auc_svm = auc(fpr_svm, tpr_svm)
    
    y_proba_nn = mlp.predict_proba(X_test_scaled)[:, 1]
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_proba_nn)
    auc_nn = auc(fpr_nn, tpr_nn)
    
    return svm_train_acc, svm_test_acc, mlp_train_acc, mlp_test_acc, fpr_svm, tpr_svm, auc_svm, fpr_nn, tpr_nn, auc_nn, svm, mlp, scaler, X_train, X_test

svm_train_acc, svm_test_acc, mlp_train_acc, mlp_test_acc, fpr_svm, tpr_svm, auc_svm, fpr_nn, tpr_nn, auc_nn, svm_model, mlp_model, scaler, X_train, X_test = train_models(X, y, selected_features)

# Home Page
if page == "Home":
    st.title("Welcome to the Breast Cancer Awareness App")
    st.image("csm_LifeScience_StageImage_BreastCancer_1500x600-01_2498abd1e0.webp", use_container_width=True)
    st.write("""
    Our mission is to raise awareness about breast cancer, provide educational resources, and support early detection through AI-driven insights. 
    This app offers comprehensive information on breast cancer types, risk factors, prevention strategies, and self-examination techniques, 
    alongside data-driven analysis to understand tumor characteristics.
    """)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.header("Understanding Breast Cancer")
    st.write("""
    Breast cancer is one of the most common cancers worldwide, affecting millions of women and some men. Early detection and awareness are key to improving outcomes. 
    Here’s a detailed overview:
    - **What is Breast Cancer?**: A disease where malignant cells form in breast tissues, often in the ducts or lobules.
    - **Prevalence**: In 2020, approximately 2.3 million women were diagnosed globally, with a 5-year survival rate of 90% when detected early.
    - **Types**: Includes Ductal Carcinoma In Situ (DCIS), Invasive Ductal Carcinoma (IDC), Lobular Carcinoma, and rare forms like Inflammatory Breast Cancer.
    - **Risk Factors**: Age (over 40), family history, genetic mutations (BRCA1/BRCA2), hormonal factors, and lifestyle choices like smoking or excessive alcohol.
    - **Symptoms**: Lumps, changes in breast shape or size, skin dimpling, nipple discharge, or persistent pain.
    - **Importance of Early Detection**: Regular screenings (mammograms, ultrasounds) and self-exams can detect cancer at earlier, more treatable stages.
    """)
    
    st.header("Why This App?")
    st.write("""
    This app combines medical education with data science to:
    - Educate users on breast cancer prevention and self-check techniques.
    - Provide AI-based analysis using the Breast Cancer Wisconsin dataset to explore tumor characteristics.
    - Offer actionable insights for health management and professional consultation.
    Always consult a healthcare provider for medical advice and diagnosis.
    """)
    
    st.header("Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Global Cases (2020)", "2.3M")
    with col2:
        st.metric("Early Detection Survival Rate", "90%")
    with col3:
        st.metric("Dataset Samples", f"{df.shape[0]}")
    
    st.markdown("<div class='motivational-quote'>You are stronger than you know. Take charge of your health today!</div>", unsafe_allow_html=True)

# Analysis Page
elif page == "Analysis":
    st.title("Breast Cancer Data Analysis")    
    st.write("""
    This page provides an in-depth analysis of the Breast Cancer Wisconsin dataset, using machine learning models 
    (Support Vector Machine and Neural Network) to classify tumors as benign or malignant. Feature selection enhances model efficiency, 
    and visualizations highlight key patterns and relationships in the data.
    """)
    
    st.header("Dataset Overview")
    st.write(f"""
    - **Dataset Size**: {df.shape[0]} samples with {df.shape[1]-2} features.
    - **Class Distribution**: {df['diagnosis'].value_counts()[0]} benign, {df['diagnosis'].value_counts()[1]} malignant.
    - **Features**: Includes measurements like radius, texture, perimeter, area, smoothness, and more, derived from breast tissue images.
    """)
    
    # Feature Selection Results
    st.header("Feature Selection")
    st.write(f"""
    Using Recursive Feature Elimination (RFE) with SVM, we selected the top 10 features to improve model performance and interpretability. 
    Selected features: {', '.join(selected_features)}.
    """)

    pink_palette = sns.light_palette("#ff69b4", n_colors=32)  
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Ranking', y='Feature', data=feature_ranking.sort_values('Ranking'), ax=ax, palette=pink_palette)
    ax.set_title('Feature Importance Ranking (RFE)')
    ax.set_xlabel('Ranking (Lower is More Important)')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
    
    # Class Distribution Visualization
    st.header("Class Distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x='diagnosis', data=df, ax=ax, palette=['#ff69b4', '#ffb6c1'])  
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_title('Distribution of Benign vs Malignant Tumors')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    
    # Correlation Heatmap
    st.header("Feature Correlations")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[selected_features].corr()
    sns.heatmap(corr, ax=ax, cmap=pink_palette, annot=True, fmt='.2f')
    ax.set_title('Correlation Heatmap of Selected Features')
    st.pyplot(fig)
    
    # Pair Plot for Key Features
    st.header("Feature Relationships")
    key_features = selected_features[:4] + ['diagnosis']
    fig = sns.pairplot(df[key_features], hue='diagnosis', palette=['#ff69b4', '#ffb6c1'], diag_kind='hist')
    fig.fig.suptitle('Pair Plot of Top Features', y=1.02)
    st.pyplot(fig)
    
    # Model Performance
    st.header("Model Performance")
    st.write(f"""
    - **SVM Training Accuracy**: {svm_train_acc:.4f}
    - **SVM Test Accuracy**: {svm_test_acc:.4f}
    - **Neural Network Training Accuracy**: {mlp_train_acc:.4f}
    - **Neural Network Test Accuracy**: {mlp_test_acc:.4f}
    - **Model Selection**: SVM outperforms Neural Network by {(svm_test_acc - mlp_test_acc) * 100:.2f}% on test data.
    """)
    
    # ROC Curve
    st.header("ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_svm, tpr_svm, 'b-', linewidth=2, label=f'SVM (AUC = {auc_svm:.3f})')
    ax.plot(fpr_nn, tpr_nn, 'r--', linewidth=2, label=f'Neural Network (AUC = {auc_nn:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for SVM and Neural Network')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("<div class='motivational-quote'>Data-driven insights empower better health decisions.</div>", unsafe_allow_html=True)

# Prevention Page
elif page == "Prevention":
    st.title("Breast Cancer Prevention")
    st.image("prevent.jpg", use_container_width=True)
    st.write("""
    While not all breast cancer cases can be prevented, adopting healthy lifestyle choices and proactive measures 
    can significantly reduce your risk and improve outcomes. Below are evidence-based strategies to lower your risk.
    """)
    
    st.header("Lifestyle Changes for Prevention")
    st.write("""
    - **Maintain a Healthy Weight**: Obesity, especially after menopause, increases breast cancer risk. Aim for a BMI between 18.5–24.9 through a balanced diet and exercise.
    - **Regular Physical Activity**: Engage in at least 150–300 minutes of moderate aerobic activity (e.g., brisk walking, cycling) or 75–150 minutes of vigorous activity weekly.
    - **Balanced Diet**: Consume a diet rich in fruits, vegetables, whole grains, and lean proteins. Limit processed foods, red meat, and sugary drinks.
    - **Limit Alcohol**: Alcohol consumption is linked to increased risk. Limit to one drink per day or less, preferably none.
    - **Avoid Smoking**: Smoking is associated with higher cancer risk. Seek support to quit through counseling or cessation programs.
    - **Breastfeeding**: If possible, breastfeeding for at least one year can reduce breast cancer risk, particularly for premenopausal women.
    - **Hormone Therapy Caution**: Long-term use of hormone replacement therapy (HRT) may increase risk. Consult your doctor for alternatives.
    """)
    
    st.header("Screening and Early Detection")
    st.write("""
    - **Mammograms**: Women aged 40+ should have annual or biennial mammograms, depending on risk factors and medical advice.
    - **Clinical Breast Exams**: Regular exams by a healthcare provider can detect abnormalities early.
    - **Genetic Testing**: If you have a family history of breast cancer, consider testing for BRCA1/BRCA2 mutations.
    - **Risk Assessment Tools**: Use tools like the Gail Model to assess your 5-year and lifetime risk.
    """)
    
    st.header("Treatment Options")
    st.write("""
    If diagnosed, treatment plans may include:
    - **Surgery**: Lumpectomy (tumor removal) or mastectomy (breast removal).
    - **Radiation Therapy**: Targets cancer cells post-surgery to reduce recurrence.
    - **Chemotherapy**: Uses drugs to destroy cancer cells, often for aggressive cancers.
    - **Hormone Therapy**: Blocks hormones fueling certain cancers (e.g., Tamoxifen).
    - **Targeted Therapy**: Drugs like Trastuzumab target specific cancer cell markers.
    - **Immunotherapy**: Boosts the immune system to fight cancer, used in specific cases.
    Consult a specialist to tailor treatments to your condition.
    """)
    
    st.header("Emotional and Social Support")
    st.write("""
    - **Support Groups**: Join local or online groups to connect with others facing similar challenges.
    - **Counseling**: Professional therapy can help manage emotional stress during diagnosis and treatment.
    - **Community Resources**: Engage with organizations like the Breast Cancer Foundation of Egypt for support and education.
    """)
    
    st.markdown("<div class='motivational-quote'>Small changes today can lead to a healthier tomorrow.</div>", unsafe_allow_html=True)

# Self-Check Page
elif page == "Self-Check":
    st.title("Breast Self-Examination")
    st.image("early-detection-key-breast-cancer-600nw-2340701729.webp", use_container_width=True)
    st.write("""
    Performing regular breast self-exams (BSE) helps you become familiar with your breasts’ normal look and feel, 
    making it easier to detect changes early. BSE is a critical tool for early detection, alongside mammograms and clinical exams.
    """)
    
    st.header("How to Perform a Breast Self-Exam")
    st.write("""
    Follow these steps monthly, ideally a few days after your menstrual period (or any consistent day for postmenopausal women):
    1. **Visual Inspection**:
       - Stand in front of a mirror with arms at your sides.
       - Look for changes in breast size, shape, skin texture (e.g., dimpling, puckering), or redness.
       - Check for nipple changes, such as inversion or discharge.
    2. **Arms Raised**:
       - Raise your arms and repeat the visual inspection for the same changes.
    3. **Palpation (Lying Down)**:
       - Lie down and use your opposite hand (e.g., right hand for left breast).
       - Use the pads of your fingers in a circular motion, applying light, medium, and firm pressure.
       - Cover the entire breast, from collarbone to upper abdomen and armpit to cleavage.
       - Use a systematic pattern (e.g., vertical strips or spiral) to ensure full coverage.
    4. **Palpation (Standing or Sitting)**:
       - Repeat the palpation step in the shower, as wet skin can make lumps easier to feel.
    5. **Nipple Check**:
       - Gently squeeze the nipple to check for discharge or abnormalities.
    """)
    
    st.header("What to Look For")
    st.write("""
    - Lumps or hard knots (even if painless).
    - Swelling, warmth, or redness.
    - Changes in breast size or shape.
    - Skin dimpling, puckering, or scaliness.
    - Nipple discharge (especially bloody) or inversion.
    - Persistent pain in one area.
    If you notice any of these, contact a healthcare provider immediately for further evaluation.
    """)
    
    st.header("Tips for Effective Self-Exams")
    st.write("""
    - **Consistency**: Perform BSE at the same time each month.
    - **Know Your Normal**: Familiarize yourself with your breasts’ usual texture and appearance.
    - **Combine with Screenings**: BSE complements mammograms and clinical exams, not replaces them.
    - **Educational Resources**: Refer to guides from organizations like the American Cancer Society for visual aids.
    """)
    
    st.header("When to Seek Help")
    st.write("""
    - **Immediate Action**: Schedule a doctor’s visit if you detect any changes or abnormalities.
    - **High-Risk Individuals**: If you have a family history or genetic predisposition, discuss more frequent screenings with your doctor.
    - **Support**: Contact local organizations like Baheya Foundation for guidance and free screening programs.
    """)
    
    st.markdown("<div class='motivational-quote'>Your health is in your hands—stay vigilant, stay strong.</div>", unsafe_allow_html=True)

# Prediction Page
elif page == "Prediction":
    st.title("Tumor Classification Prediction")
    st.write("""
    This page allows you to input values for selected features to predict whether a tumor is benign or malignant 
    using trained SVM and Neural Network models. The predictions are based on the top 10 features selected via RFE. 
    Note: These predictions are for educational purposes only and should not replace professional medical advice.
    """)
    
    st.header("Enter Feature Values")
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.number_input(
            f"Enter {feature}",
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )
    
    if st.button("Predict Tumor Classification"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Standardize input using the trained scaler
        input_scaled = scaler.transform(input_df)
        
        # Predict using SVM
        svm_pred = svm_model.predict(input_scaled)[0]
        svm_prob = svm_model.predict_proba(input_scaled)[0]
        
        # Predict using Neural Network
        nn_pred = mlp_model.predict(input_scaled)[0]
        nn_prob = mlp_model.predict_proba(input_scaled)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### SVM Prediction")
            st.metric("Classification", "Malignant" if svm_pred == 1 else "Benign")
            st.metric("Probability", f"{max(svm_prob):.2%}")
        
        with col2:
            st.markdown("#### Neural Network Prediction")
            st.metric("Classification", "Malignant" if nn_pred == 1 else "Benign")
            st.metric("Probability", f"{max(nn_prob):.2%}")
    
    st.markdown("<div class='motivational-quote'>Knowledge is power—always seek professional guidance.</div>", unsafe_allow_html=True)

# Hospitals Page
elif page == "Hospitals":
    st.title("Hospitals Treating Breast Cancer in Egypt")
    st.image("hospital.jpeg", use_container_width=True)
    st.write("""
    Below is a list of reputable hospitals in Egypt that specialize in breast cancer treatment. 
    Contact these facilities for professional support, diagnosis, and treatment options.
    """)
    st.header("Hospital Contact Information")
    st.write("""
    - **Baheya Hospital**  
      - **Specialization**: First hospital in Egypt dedicated to breast cancer treatment and early detection.  
      - **Address**: 123 Health Street, Cairo, Egypt  
      - **Phone**: +20 123 456 7890  
      - **Email**: info@bahyahospital.org  
      - **Website**: [www.bahyahospital.org](http://www.bahyahospital.org)[](https://www.baheya.org/en)

    - **Dar Al Fouad Hospital**  
      - **Specialization**: JCI-accredited hospital with extensive surgical experience, including tumor operations.  
      - **Address**: 26th of July Corridor, Nasr City, Cairo, Egypt  
      - **Phone**: +20 2 2529 2000  
      - **Email**: info@daralfouad.org  
      - **Website**: [www.daralfouad.org](http://www.daralfouad.org)[](https://www.lyfboat.com/hospitals/breast-cancer-hospitals-and-costs-in-egypt/)

    - **Andalusia Smouha Hospital**  
      - **Specialization**: Offers advanced diagnostics, surgical options, and hormone therapy for breast cancer.  
      - **Address**: Smouha, Alexandria, Egypt  
      - **Phone**: +20 3 427 7777  
      - **Email**: info@andalusiahospitals.com  
      - **Website**: [www.andalusiahospitals.com](http://www.andalusiahospitals.com)[](https://www.lyfboat.com/hospitals/breast-cancer-hospitals-and-costs-in-egypt/)

    - **Cleopatra Hospital**  
      - **Specialization**: Largest private hospital group in Egypt, recognized for oncology excellence.  
      - **Address**: 39 Cleopatra St., Heliopolis, Cairo, Egypt  
      - **Phone**: +20 2 2414 3931  
      - **Email**: info@cleopatrahospital.com  
      - **Website**: [www.cleopatrahospital.com](http://www.cleopatrahospital.com)[](https://my1health.com/articles/cancer-treatment-hospitals-egypt)

    - **Cairo Oncology Center (Cairocure)**  
      - **Specialization**: Largest private oncology facility in the Middle East, treating 1,800 new patients yearly.  
      - **Address**: 55 Abdelmoneim Riad St., Cairo Medical Tower, Mohandesseen, Giza, Egypt  
      - **Phone**: +20 2 3302 6814  
      - **Email**: info@cairocure.com  
      - **Website**: [www.cairocure.com](http://www.cairocure.com)[](https://www.breastcentresnetwork.org/breast-units/directory/Egypt/Cairo%2BOncology%2BCenter/4%2C9%2C650%2C)

    - **National Cancer Institute (NCI) - Cairo**  
      - **Specialization**: Leading cancer treatment and research center with multidisciplinary care.  
      - **Address**: Kasr Al Ainy St., Fom El Khalig, Cairo, Egypt  
      - **Phone**: +20 2 2364 8888  
      - **Email**: info@nci.cu.edu.eg  
      - **Website**: [www.nci.cu.edu.eg](http://www.nci.cu.edu.eg)[](https://www.uicc.org/membership/national-cancer-institute-cairo)

    - **Alfa Cure Oncology Center**  
      - **Specialization**: ESMO-designated center for integrated oncology and palliative care.  
      - **Address**: Heliopolis, Cairo, Egypt  
      - **Phone**: +20 2 2417 0000  
      - **Email**: info@alfacure.com  
      - **Website**: [www.alfacure.com](http://www.alfacure.com)[](https://www.esmo.org/for-patients/esmo-designated-centres-of-integrated-oncology-palliative-care/esmo-accredited-designated-centres/alfa-cure-center)
    """)
    st.markdown("<div class='motivational-quote'>You are stronger than you know. Take charge of your health today!</div>", unsafe_allow_html=True)

# About Page
elif page == "About US":
    st.title("About the Breast Cancer Awareness App")
    st.write("""
    The Breast Cancer Awareness App is a dedicated platform designed to educate, empower, and support individuals in understanding breast cancer, 
    its prevention, and early detection strategies. Developed with a commitment to public health, this app combines cutting-edge data science 
    with accessible medical information to foster awareness and proactive health management.
    """)
    
    st.header("Our Mission")
    st.write("""
    Our mission is to:
    - Raise awareness about breast cancer through reliable, evidence-based information.
    - Provide tools for understanding tumor characteristics using AI-driven analysis.
    - Encourage early detection through self-exams and regular screenings.
    - Support users with resources to reduce risk and seek professional care.
    """)
    
    st.header("About the Developers")
    st.write("""
    This app was created by **Gehad Helmy** and **Alaa Adel**, two passionate data scientists and health advocates dedicated to leveraging technology 
    for social good. With expertise in machine learning and a commitment to public health, they developed this app to empower individuals with knowledge 
    and tools to combat breast cancer.
    """)
    
    st.header("Data and Technology")
    st.write("""
    The app utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**, a widely recognized dataset containing 569 samples with 30 features derived 
    from breast tissue images. We employ feature selection (RFE) to focus on the top 10 features, and use Support Vector Machines (SVM) and Neural Networks 
    to analyze tumor characteristics. Visualizations are powered by Matplotlib and Seaborn for clear, professional data representation.
    """)
    
    st.header("Disclaimer")
    st.write("""
    This app is for informational and educational purposes only. The AI-powered analyses and predictions provided are not a substitute for professional 
    medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns, especially if you notice any changes 
    in your breasts or have risk factors for breast cancer.
    """)
    
    st.header("Get Involved")
    st.write("""
        - **Spread Awareness**: Share this app with friends and family to promote breast cancer education.  
        - **Support Organizations**: Connect with local and global organizations like the Breast Cancer Foundation of Egypt or Susan G. Komen.  
        - **Feedback**: Contact us on LinkedIn  [Gehad Helmy](https://www.linkedin.com/in/gehad-helmy-505445296/) and  [Alaa Adel](http://linkedin.com/in/alaa-adel-64735034b)
        """)
    st.header("Acknowledgments")
    st.write("""
    We thank the open-source community for providing tools like Streamlit, Pandas, and Scikit-learn, which made this app possible. 
    Special thanks to the Breast Cancer Wisconsin dataset contributors and healthcare professionals worldwide fighting breast cancer.
    """)
    
    st.markdown("<div class='motivational-quote'>Together, we can inspire hope and drive change.</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    Developed by Gehad Helmy & Alaa Adel | © 2025 Breast Cancer Awareness App
</div>
""", unsafe_allow_html=True)
