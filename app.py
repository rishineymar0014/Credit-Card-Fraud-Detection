import streamlit as st
import numpy as np
import joblib
import re

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.main-title {
    text-align:center;
    font-size:60px;
    font-weight:700;
    color:white;
    margin-top:10px;
}

.sub-title{
    text-align:center;
    color:#B0B0B0;
    font-size:20px;
    margin-bottom:30px;
}

.result-box{
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:#888;
    margin-top:60px;
    font-size:18px;
}

.block-container{
    padding-top:2rem;
    max-width:1200px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("credit_card_model.pkl")

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered Fraud Prediction System</div>',
    unsafe_allow_html=True
)

st.write("### Enter all feature values")

# ---------------- INPUT ---------------- #

user_input = st.text_area(
    "",
    height=180,
    placeholder="""Example:

0    -1.359807134    -0.072781173    2.536346738 ...
"""
)

# ---------------- BUTTON ---------------- #

if st.button("🚀 Predict Transaction", use_container_width=True):

    try:

        # Split tabs/spaces/commas
        values = [
            float(x)
            for x in re.split(r'[\s,]+', user_input.strip())
            if x != ""
        ]

        EXPECTED_FEATURES = 30

        if len(values) != EXPECTED_FEATURES:
            st.error(
                f"Expected {EXPECTED_FEATURES} values but found {len(values)}"
            )
            st.stop()

        data = np.array(values).reshape(1, -1)

        prediction = model.predict(data)[0]

        # Probability if supported
        fraud_prob = None

        if hasattr(model, "predict_proba"):
            fraud_prob = model.predict_proba(data)[0][1] * 100

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("🚨 FRAUD TRANSACTION")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

        with col2:

            if fraud_prob is not None:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_prob:.2f}%"
                )

        st.divider()

        st.write("### Transaction Summary")

        st.write(f"Total Features Received: **{len(values)}**")

        st.write("Prediction completed successfully.")

    except ValueError:
        st.error(
            "Invalid input. Please paste only numeric values."
        )

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- FOOTER ---------------- #

st.markdown(
    """
    <div class="footer">
        Made with  by <b>Rishi Kumar</b>
    </div>
    """,
    unsafe_allow_html=True
)