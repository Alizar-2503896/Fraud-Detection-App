# ============================================
# FINAL WORKING APP.PY
# NO user_stats.pkl needed
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection System")
st.markdown("MSc Project - Leeds Trinity University")
st.markdown("---")

# ============================================
# LOAD MODEL AND FILES
# ============================================

@st.cache_resource
def load_assets():
    try:
        # Check if files exist
        required_files = ['best_model.pkl', 'scaler.pkl', 'feature_names.pkl', 'encoders.pkl']
        for f in required_files:
            if not os.path.exists(f):
                st.error(f"❌ Missing file: {f}")
                return None, None, None, None, None
        
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Load threshold
        if os.path.exists('best_threshold.txt'):
            with open('best_threshold.txt', 'r') as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.40
            
        return model, scaler, encoders, feature_names, threshold
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None

# Load assets
model, scaler, encoders, feature_names, threshold = load_assets()

if model is None:
    st.stop()

st.success(f"✅ Model loaded successfully! (Threshold: {threshold*100:.0f}%)")

# ============================================
# SIDEBAR INPUTS
# ============================================

st.sidebar.header("📥 Transaction Details")

# Get options from encoders
device_options = encoders['Device_Used'].classes_.tolist()
location_options = encoders['Location'].classes_.tolist()
payment_options = encoders['Payment_Method'].classes_.tolist()
type_options = encoders['Transaction_Type'].classes_.tolist()

# Input fields
transaction_amount = st.sidebar.number_input(
    "💰 Transaction Amount ($)", 
    min_value=0.0, 
    value=100.0, 
    step=10.0
)

time_of_transaction = st.sidebar.slider(
    "🕐 Time of Transaction (Hour)", 
    0, 23, 14
)

transaction_type = st.sidebar.selectbox(
    "📊 Transaction Type",
    type_options
)

device_used = st.sidebar.selectbox(
    "📱 Device Used",
    device_options
)

location = st.sidebar.selectbox(
    "📍 Location",
    location_options
)

payment_method = st.sidebar.selectbox(
    "💳 Payment Method",
    payment_options
)

previous_fraud = st.sidebar.number_input(
    "Previous Fraudulent Transactions", 
    min_value=0, 
    max_value=20, 
    value=0
)

transactions_24h = st.sidebar.number_input(
    "Transactions in Last 24 Hours", 
    min_value=0, 
    max_value=50, 
    value=2
)

account_age = st.sidebar.number_input(
    "Account Age (Days)", 
    min_value=0, 
    max_value=2000, 
    value=365
)

# ============================================
# PREDICT BUTTON
# ============================================

predict_button = st.sidebar.button("🔍 Predict Transaction Status", use_container_width=True)

# ============================================
# PREDICTION LOGIC
# ============================================

if predict_button:
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    with st.spinner("🔍 Analyzing transaction..."):
        
        # Encode categorical values
        encoded_type = encoders['Transaction_Type'].transform([transaction_type])[0]
        encoded_device = encoders['Device_Used'].transform([device_used])[0]
        encoded_location = encoders['Location'].transform([location])[0]
        encoded_payment = encoders['Payment_Method'].transform([payment_method])[0]
        
        # Feature engineering
        is_late_night = 1 if time_of_transaction >= 22 or time_of_transaction <= 6 else 0
        is_high_amount = 1 if transaction_amount > 2000 else 0
        fraud_risk_score = previous_fraud * 10
        transaction_frequency = transactions_24h / 10
        age_risk = 1 if account_age < 30 else 0
        amount_deviation = 0.5
        device_change = 0
        location_change = 0
        
        # Create input data
        input_data = pd.DataFrame([{
            'Transaction_Amount': transaction_amount,
            'Time_of_Transaction': time_of_transaction,
            'Previous_Fraudulent_Transactions': previous_fraud,
            'Account_Age': account_age,
            'Number_of_Transactions_Last_24H': transactions_24h,
            'is_late_night': is_late_night,
            'is_high_amount': is_high_amount,
            'fraud_risk_score': fraud_risk_score,
            'transaction_frequency': transaction_frequency,
            'age_risk': age_risk,
            'amount_deviation': amount_deviation,
            'device_change': device_change,
            'location_change': location_change,
            'Transaction_Type_encoded': encoded_type,
            'Device_Used_encoded': encoded_device,
            'Location_encoded': encoded_location,
            'Payment_Method_encoded': encoded_payment
        }])
        
        # Ensure features are in the correct order
        try:
            input_data = input_data[feature_names]
        except KeyError as e:
            st.error(f"❌ Feature mismatch: {e}")
            st.stop()
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        probability = model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if probability >= threshold else 0
        
        # ============================================
        # DISPLAY RESULTS
        # ============================================
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.error(f"⚠️ FRAUD ALERT DETECTED")
            else:
                st.success(f"✅ LEGITIMATE TRANSACTION")
            
            st.metric("Fraud Probability", f"{probability*100:.1f}%")
            st.caption(f"Decision Threshold: {threshold*100:.0f}%")
            
            st.markdown("### 💡 Why This Decision?")
            
            if prediction == 1:
                st.write("**🚨 Flagged as FRAUD because:**")
                
                risk_factors = []
                if is_late_night == 1:
                    risk_factors.append("• Late night transaction (higher risk)")
                if transaction_amount > 1000:
                    risk_factors.append(f"• High transaction amount (${transaction_amount:.2f})")
                if previous_fraud > 0:
                    risk_factors.append(f"• Past fraud history ({previous_fraud} previous frauds)")
                if account_age < 30:
                    risk_factors.append("• New account (less than 30 days old)")
                if transactions_24h > 5:
                    risk_factors.append(f"• High velocity ({transactions_24h} transactions in 24 hours)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("• The model detected suspicious patterns")
                    
                st.warning("💡 **Recommendation:** Block this transaction and verify with the customer.")
                
            else:
                st.write("**✅ Classified as LEGITIMATE because:**")
                st.write("• No significant fraud indicators were detected")
                st.write("• Transaction pattern is consistent with normal behavior")
                st.success("💡 **Recommendation:** Allow this transaction to proceed.")
        
        with col2:
            st.markdown("### 📊 Transaction Summary")
            st.write(f"**Amount:** ${transaction_amount:.2f}")
            st.write(f"**Time:** {time_of_transaction}:00")
            st.write(f"**Type:** {transaction_type}")
            st.write(f"**Device:** {device_used}")
            st.write(f"**Location:** {location}")
            st.write(f"**Account Age:** {account_age} days")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>Fraud Detection System v1.0 | Built with Streamlit, XGBoost, and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)