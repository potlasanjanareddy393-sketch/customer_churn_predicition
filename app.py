import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Travel Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* -----------------------------
       GENERAL PAGE
    ----------------------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f0f7ff 0%,
            #e6f2ff 50%,
            #f8fbff 100%
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* -----------------------------
       SIDEBAR
    ----------------------------- */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #ffffff 0%,
            #f5f9ff 100%
        );

        border-right: 2px solid #dbe7f5;
    }

    [data-testid="stSidebar"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #0f172a !important;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0066ff;
        margin-bottom: 1rem;
    }

    .sidebar-description {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* -----------------------------
       HERO
    ----------------------------- */

    .hero {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #f8fbff 100%
        );

        padding: 2.5rem;
        border-radius: 20px;

        border: 2px solid #dce8f5;

        box-shadow:
            0 10px 35px rgba(0, 102, 255, 0.08);

        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0066ff;
        margin-bottom: 0.7rem;
    }

    .hero-description {
        font-size: 1.1rem;
        color: #64748b;
        line-height: 1.6;
        margin: 0;
    }

    /* -----------------------------
       METRIC CARDS
    ----------------------------- */

    .metric-card {
        background: #ffffff;

        padding: 1.5rem;

        border-radius: 16px;

        border: 1.5px solid #dce8f5;

        box-shadow:
            0 8px 25px rgba(0, 102, 255, 0.06);

        text-align: center;

        min-height: 150px;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0066ff;
    }

    .metric-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.4rem;
    }

    /* -----------------------------
       CARDS
    ----------------------------- */

    .card {
        background: #ffffff;

        padding: 1.8rem;

        border-radius: 18px;

        border: 1.5px solid #dce8f5;

        box-shadow:
            0 8px 25px rgba(0, 102, 255, 0.06);

        min-height: 100%;
    }

    .card-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #1e3a8a;
        margin-bottom: 1.2rem;
    }

    /* -----------------------------
       CUSTOMER INFORMATION
    ----------------------------- */

    .customer-info {
        background: linear-gradient(
            135deg,
            #f0f7ff,
            #e8f2ff
        );

        border-left: 4px solid #0066ff;

        padding: 0.9rem;

        border-radius: 8px;

        margin-bottom: 0.8rem;

        color: #1e293b;

        font-size: 1rem;
    }

    .customer-info strong {
        color: #0f172a;
    }

    /* -----------------------------
       PREDICTION RESULT
    ----------------------------- */

    .result-box {
        padding: 2rem;

        border-radius: 16px;

        text-align: center;

        color: white;

        margin-top: 1rem;

        margin-bottom: 1rem;
    }

    .result-success {
        background: linear-gradient(
            135deg,
            #10b981,
            #059669
        );

        border: 2px solid #6ee7b7;

        box-shadow:
            0 10px 30px rgba(16, 185, 129, 0.25);
    }

    .result-warning {
        background: linear-gradient(
            135deg,
            #f59e0b,
            #d97706
        );

        border: 2px solid #fbbf24;

        box-shadow:
            0 10px 30px rgba(245, 158, 11, 0.25);
    }

    .result-danger {
        background: linear-gradient(
            135deg,
            #ef4444,
            #dc2626
        );

        border: 2px solid #fca5a5;

        box-shadow:
            0 10px 30px rgba(239, 68, 68, 0.25);
    }

    .result-icon {
        font-size: 2.7rem;
        margin-bottom: 0.5rem;
    }

    .result-title {
        font-size: 1.4rem;
        font-weight: 800;
    }

    .result-percentage {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }

    .result-description {
        font-size: 0.95rem;
        margin-top: 0.7rem;
    }

    /* -----------------------------
       BUTTON
    ----------------------------- */

    .stButton > button {
        width: 100%;

        background: linear-gradient(
            135deg,
            #0066ff,
            #00a3ff
        );

        color: white;

        border: none;

        border-radius: 12px;

        padding: 0.8rem 1.5rem;

        font-size: 1rem;

        font-weight: 700;

        box-shadow:
            0 7px 20px rgba(0, 102, 255, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #0052cc,
            #0088dd
        );

        color: white;

        border: none;
    }

    /* -----------------------------
       FOOTER
    ----------------------------- */

    .footer {
        text-align: center;

        color: #64748b;

        padding: 2rem 0;

        font-size: 0.95rem;
    }

    .footer-small {
        font-size: 0.82rem;

        opacity: 0.7;

        margin-top: 0.8rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

if not MODEL_PATH.exists():

    st.error("❌ model.pkl was not found.")

    st.info(
        "Please make sure model.pkl is in the same folder as app.py."
    )

    st.stop()


try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("❌ Unable to load the machine learning model.")

    st.code(str(e))

    st.stop()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        ✈️ Travel Customer Churn Predictor
    </div>

    <p class="hero-description">
        Machine learning application that predicts whether a
        travel customer is likely to leave the service.
        Enter customer details and click Predict Churn to
        generate a prediction.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="metric-card">

        <div class="metric-label">
            📊 Dataset Size
        </div>

        <div class="metric-value">
            954
        </div>

        <div class="metric-subtitle">
            Training samples
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="metric-card">

        <div class="metric-label">
            ⚠️ Baseline Churn
        </div>

        <div class="metric-value">
            23.5%
        </div>

        <div class="metric-subtitle">
            Historical rate
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="metric-card">

        <div class="metric-label">
            🤖 Model Type
        </div>

        <div class="metric-value">
            RF
        </div>

        <div class="metric-subtitle">
            Random Forest
        </div>

    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR - CUSTOMER INPUT
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        📋 Customer Profile
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-description">
        Enter customer details below:
    </div>
    """, unsafe_allow_html=True)

    # AGE
    age = st.slider(
        "Age 🎂",
        min_value=18,
        max_value=75,
        value=35,
        step=1
    )

    # FREQUENT FLYER
    frequent = st.selectbox(
        "Frequent Flyer ✈️",
        ["No", "Yes"]
    )

    # INCOME
    income = st.selectbox(
        "Income Class 💰",
        ["Low Income", "Middle Income"]
    )

    # SERVICES
    services = st.slider(
        "Services Opted 🎁",
        min_value=1,
        max_value=6,
        value=2,
        step=1
    )

    # SOCIAL MEDIA
    social = st.selectbox(
        "Social Media Sync 📱",
        ["No", "Yes"]
    )

    # HOTEL
    hotel = st.selectbox(
        "Booked Hotel 🏨",
        ["No", "Yes"]
    )

    st.markdown("---")

    st.caption(
        "All fields are required. Click 'Predict Churn' "
        "to get results."
    )


# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown("""
<h2 style="
    color:#0f172a;
    margin-bottom:1rem;
">
    Customer Information & Prediction
</h2>
""", unsafe_allow_html=True)


left_col, right_col = st.columns(
    [1.3, 1],
    gap="large"
)


# ============================================================
# CUSTOMER PROFILE SUMMARY
# ============================================================

with left_col:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            👤 Customer Profile Summary
        </div>

    """, unsafe_allow_html=True)

    # AGE
    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Age:</strong>
            {age} years old
        </div>
        """,
        unsafe_allow_html=True
    )

    # TRAVEL STATUS
    if frequent == "Yes":

        travel_status = "✈️ Frequent Flyer"

    else:

        travel_status = "🚫 Occasional Traveler"

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Travel Status:</strong>
            {travel_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    # INCOME
    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Income Class:</strong>
            {income}
        </div>
        """,
        unsafe_allow_html=True
    )

    # SERVICES
    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Services Opted:</strong>
            {services} out of 6 services
        </div>
        """,
        unsafe_allow_html=True
    )

    # SOCIAL MEDIA
    if social == "Yes":

        social_status = "✅ Connected"

    else:

        social_status = "❌ Not Connected"

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Social Media Sync:</strong>
            {social_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    # HOTEL
    if hotel == "Yes":

        hotel_status = "✅ Has booked"

    else:

        hotel_status = "❌ No booking"

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Hotel Booking:</strong>
            {hotel_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

with right_col:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🎯 Churn Prediction
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "🔮 Predict Churn",
        use_container_width=True
    )


    if predict_clicked:

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Age": [age],

            "FrequentFlyer": [frequent],

            "AnnualIncomeClass": [income],

            "ServicesOpted": [services],

            "AccountSyncedToSocialMedia": [social],

            "BookedHotelOrNot": [hotel]

        })


        # ----------------------------------------------------
        # ENCODE CATEGORICAL VALUES
        # ----------------------------------------------------
        #
        # These mappings are used because the model needs
        # numerical values.
        #
        # No = 0
        # Yes = 1
        #
        # Low Income = 0
        # Middle Income = 1
        #
        # ----------------------------------------------------

        input_data["FrequentFlyer"] = (
            input_data["FrequentFlyer"]
            .map({
                "No": 0,
                "Yes": 1
            })
        )

        input_data["AnnualIncomeClass"] = (
            input_data["AnnualIncomeClass"]
            .map({
                "Low Income": 0,
                "Middle Income": 1
            })
        )

        input_data["AccountSyncedToSocialMedia"] = (
            input_data["AccountSyncedToSocialMedia"]
            .map({
                "No": 0,
                "Yes": 1
            })
        )

        input_data["BookedHotelOrNot"] = (
            input_data["BookedHotelOrNot"]
            .map({
                "No": 0,
                "Yes": 1
            })
        )


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(input_data)[0]


            # ------------------------------------------------
            # GET PROBABILITY
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                # Find probability belonging to predicted class
                if hasattr(model, "classes_"):

                    classes = list(model.classes_)

                    if prediction in classes:

                        predicted_index = classes.index(
                            prediction
                        )

                        prediction_probability = (
                            probabilities[predicted_index]
                        )

                    else:

                        prediction_probability = max(
                            probabilities
                        )

                else:

                    prediction_probability = max(
                        probabilities
                    )

            else:

                prediction_probability = None


            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)


            if prediction == 1:

                # HIGH CHURN
                if prediction_probability is not None:

                    percentage = (
                        prediction_probability * 100
                    )

                else:

                    percentage = 100


                st.markdown(
                    f"""
                    <div class="result-box result-danger">

                        <div class="result-icon">
                            🚨
                        </div>

                        <div class="result-title">
                            HIGH CHURN RISK
                        </div>

                        <div class="result-percentage">
                            {percentage:.0f}%
                        </div>

                        <div class="result-description">
                            Probability of churn
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.warning(
                    "⚠️ This customer may be at risk of "
                    "leaving the service. The company can "
                    "consider suitable customer-retention "
                    "actions."
                )


            else:

                # LOW CHURN
                if prediction_probability is not None:

                    percentage = (
                        prediction_probability * 100
                    )

                else:

                    percentage = 100


                st.markdown(
                    f"""
                    <div class="result-box result-success">

                        <div class="result-icon">
                            ✅
                        </div>

                        <div class="result-title">
                            LOW CHURN RISK
                        </div>

                        <div class="result-percentage">
                            {percentage:.0f}%
                        </div>

                        <div class="result-description">
                            Likelihood to stay
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "✨ This prediction indicates a lower "
                    "likelihood of customer churn."
                )


        except Exception as e:

            st.error(
                "❌ Error making prediction."
            )

            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div class="footer">

    <div>
        🔒 <strong>Enterprise-Grade Security</strong>
        • All data processed locally
    </div>

    <div style="margin-top:0.5rem;">
        Powered by Random Forest ML • Built with Streamlit
    </div>

    <div class="footer-small">
        © 2026 Travel Churn Predictor • Machine Learning Project
    </div>

</div>
""", unsafe_allow_html=True)
