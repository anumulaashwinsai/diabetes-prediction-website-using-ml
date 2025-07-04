import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model_optimal.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model
model, feature_names = load_model()

# Title and header
st.title("🩺 Diabetes Risk Prediction")
st.markdown("### AI-Powered Health Assessment Tool")
st.markdown("*Based on BRFSS 2015 Health Indicators Dataset*")

if model is not None:
    st.success(f"✅ Model loaded successfully! (Accuracy: 83.6%)")
else:
    st.error("❌ Model failed to load")
    st.stop()

# Create input form
with st.form("diabetes_prediction_form"):
    st.markdown("## 📋 Health Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.selectbox("Age Group", options=list(range(1, 14)), 
                          format_func=lambda x: {
                              1: "18-24 years", 2: "25-29 years", 3: "30-34 years",
                              4: "35-39 years", 5: "40-44 years", 6: "45-49 years",
                              7: "50-54 years", 8: "55-59 years", 9: "60-64 years",
                              10: "65-69 years", 11: "70-74 years", 12: "75-79 years",
                              13: "80+ years"
                          }[x])
        
        education = st.selectbox("Education Level", options=list(range(1, 7)),
                               format_func=lambda x: {
                                   1: "Never attended school", 2: "Elementary", 
                                   3: "Some high school", 4: "High school graduate",
                                   5: "Some college", 6: "College graduate"
                               }[x])
        
        income = st.selectbox("Income Level", options=list(range(1, 9)),
                            format_func=lambda x: {
                                1: "Less than $10,000", 2: "$10,000-$14,999",
                                3: "$15,000-$19,999", 4: "$20,000-$24,999",
                                5: "$25,000-$34,999", 6: "$35,000-$49,999",
                                7: "$50,000-$74,999", 8: "$75,000 or more"
                            }[x])
        
        st.markdown("### 🏥 Medical History")
        high_bp = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        high_chol = st.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        chol_check = st.selectbox("Cholesterol Check (past 5 years)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        stroke = st.selectbox("Ever had a Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart Disease or Attack", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        st.markdown("### 🏃 Physical Health")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=12.0, max_value=98.0, value=25.0, step=0.1)
        gen_health = st.selectbox("General Health", options=list(range(1, 6)),
                                format_func=lambda x: {
                                    1: "Excellent", 2: "Very Good", 3: "Good", 
                                    4: "Fair", 5: "Poor"
                                }[x])
        
        mental_health = st.number_input("Mental Health (days of poor mental health in past 30 days)", 
                                      min_value=0, max_value=30, value=0)
        physical_health = st.number_input("Physical Health (days of poor physical health in past 30 days)", 
                                        min_value=0, max_value=30, value=0)
        diff_walk = st.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        phys_activity = st.selectbox("Physical Activity (past 30 days)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("### 🍎 Lifestyle")
        smoker = st.selectbox("Smoked 100+ cigarettes in lifetime", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        fruits = st.selectbox("Consume fruit 1+ times per day", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        veggies = st.selectbox("Consume vegetables 1+ times per day", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("### 🏥 Healthcare Access")
        healthcare = st.selectbox("Have Healthcare Coverage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        no_doc_cost = st.selectbox("Couldn't see doctor due to cost (past 12 months)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)
    
    if submitted:
        # Create feature array
        features = np.array([
            high_bp, high_chol, chol_check, bmi, smoker, stroke,
            heart_disease, phys_activity, fruits, veggies, alcohol,
            healthcare, no_doc_cost, gen_health, mental_health,
            physical_health, diff_walk, sex, age, education, income
        ]).reshape(1, -1)
        
        # Make prediction
        try:
            prediction = model.predict(features)[0]
            
            # Get probabilities
            try:
                probabilities = model.predict_proba(features)[0]
                probs = {
                    "no_diabetes": probabilities[0],
                    "prediabetes": probabilities[1] if len(probabilities) > 2 else 0.0,
                    "diabetes": probabilities[-1]
                }
            except:
                probs = None
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            if prediction == 2:
                st.error("⚠️ **High Risk - Diabetes Indicated**")
                st.markdown("""
                **Recommendation:** Based on the health indicators provided, there is a strong indication of diabetes risk. 
                Please consult with a healthcare professional immediately for proper testing and evaluation.
                """)
            elif prediction == 1:
                st.warning("⚡ **Moderate Risk - Prediabetes Indicated**")
                st.markdown("""
                **Recommendation:** Based on the health indicators provided, there may be signs of prediabetes. 
                Consider lifestyle modifications and consult with a healthcare professional for further evaluation.
                """)
            else:
                st.success("✅ **Lower Risk - No Diabetes Indicated**")
                st.markdown("""
                **Recommendation:** Based on the current health indicators, the risk appears to be lower. 
                Continue maintaining a healthy lifestyle and regular health checkups.
                """)
            
            if probs:
                st.markdown("### 📈 Risk Probabilities")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("No Diabetes", f"{probs['no_diabetes']:.1%}")
                with col2:
                    st.metric("Prediabetes", f"{probs['prediabetes']:.1%}")
                with col3:
                    st.metric("Diabetes", f"{probs['diabetes']:.1%}")
                    
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Disclaimer
st.markdown("---")
st.markdown("""
### ⚠️ Important Disclaimer
This tool provides educational estimates only and should not replace professional medical advice. 
Always consult healthcare providers for medical decisions. The model is based on BRFSS 2015 data 
and achieves 83.6% accuracy on test data.
""")

# Model info
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    - **Model Type:** Optimized Random Forest Classifier
    - **Training Data:** BRFSS 2015 Health Indicators Dataset (253,680 records)
    - **Features:** 21 health and demographic indicators
    - **Accuracy:** 83.6% on test data
    - **Model Size:** 1.6MB (optimized for deployment)
    """)