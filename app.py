import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Travel Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file is missing.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    st.error("Unable to load the prediction model.")
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   MAIN PAGE
   ========================================================= */

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(135deg, #f4f9ff 0%, #ffffff 45%, #eef7ff 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #075985 0%,
        #0369a1 50%,
        #0284c7 100%
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] input {
    color: #111827 !important;
}

[data-testid="stSidebar"] label {
    font-weight: 600;
}

.sidebar-logo {
    text-align: center;
    font-size: 46px;
    margin-top: 5px;
    margin-bottom: 5px;
}

.sidebar-title {
    text-align: center;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    text-align: center;
    font-size: 13px;
    opacity: 0.85;
    margin-bottom: 28px;
}

.sidebar-section {
    font-size: 14px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #075985 0%,
            #0284c7 48%,
            #38bdf8 100%
        );

    border-radius: 28px;

    padding: 48px 55px;

    color: white;

    box-shadow:
        0 20px 45px rgba(2, 132, 199, 0.22);

    margin-bottom: 28px;
}

.hero::after {
    content: "✈";
    position: absolute;
    right: 45px;
    top: 15px;

    font-size: 130px;
    opacity: 0.08;

    transform: rotate(-12deg);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-badge {
    display: inline-block;

    background: rgba(255,255,255,0.16);

    border: 1px solid rgba(255,255,255,0.25);

    border-radius: 30px;

    padding: 7px 15px;

    font-size: 13px;
    font-weight: 700;

    margin-bottom: 15px;
}

.hero-title {
    font-size: 43px;
    font-weight: 900;

    margin: 0;

    letter-spacing: -1px;
}

.hero-description {
    font-size: 16px;

    max-width: 760px;

    line-height: 1.7;

    margin-top: 13px;

    opacity: 0.94;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.section-heading {
    font-size: 25px;
    font-weight: 850;

    color: #0f172a;

    margin-top: 20px;
    margin-bottom: 6px;
}

.section-subheading {
    color: #64748b;

    font-size: 14px;

    margin-bottom: 20px;
}


/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 7px 25px rgba(15, 23, 42, 0.05);

    transition: 0.2s ease;
}

.info-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 12px 30px rgba(15, 23, 42, 0.09);
}

.card-icon {
    font-size: 28px;
}

.card-label {
    color: #64748b;

    font-size: 12px;

    font-weight: 800;

    text-transform: uppercase;

    margin-top: 8px;
}

.card-value {
    color: #075985;

    font-size: 25px;

    font-weight: 900;

    margin-top: 3px;
}


/* =========================================================
   PROFILE CARD
   ========================================================= */

.profile-card {
    background: white;

    border-radius: 22px;

    border: 1px solid #e2e8f0;

    padding: 28px;

    box-shadow:
        0 10px 30px rgba(15, 23, 42, 0.06);
}

.profile-header {
    display: flex;
    align-items: center;

    gap: 15px;

    margin-bottom: 25px;
}

.profile-avatar {
    width: 55px;
    height: 55px;

    border-radius: 16px;

    background: linear-gradient(
        135deg,
        #0284c7,
        #38bdf8
    );

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 27px;
}

.profile-heading {
    font-size: 21px;

    font-weight: 850;

    color: #0f172a;
}

.profile-subtitle {
    font-size: 13px;

    color: #64748b;

    margin-top: 2px;
}


/* =========================================================
   DETAIL ITEMS
   ========================================================= */

.detail-grid {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 12px;
}

.detail-item {
    background: #f8fbff;

    border: 1px solid #e0f2fe;

    border-radius: 13px;

    padding: 15px;
}

.detail-label {
    font-size: 12px;

    color: #64748b;

    font-weight: 700;

    margin-bottom: 5px;
}

.detail-value {
    font-size: 15px;

    color: #0f172a;

    font-weight: 800;
}


/* =========================================================
   PREDICTION AREA
   ========================================================= */

.prediction-panel {
    background: white;

    border-radius: 22px;

    border: 1px solid #e2e8f0;

    padding: 28px;

    min-height: 370px;

    box-shadow:
        0 10px 30px rgba(15, 23, 42, 0.06);
}

.prediction-header {
    font-size: 21px;

    font-weight: 850;

    color: #0f172a;

    margin-bottom: 5px;
}

.prediction-description {
    color: #64748b;

    font-size: 13px;

    margin-bottom: 22px;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {

    width: 100%;

    border: none;

    border-radius: 14px;

    padding: 15px 20px;

    background:
        linear-gradient(
            135deg,
            #0284c7,
            #0ea5e9
        );

    color: white;

    font-size: 16px;

    font-weight: 800;

    box-shadow:
        0 8px 22px rgba(14,165,233,0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 28px rgba(14,165,233,0.35);
}


/* =========================================================
   EMPTY PREDICTION
   ========================================================= */

.empty-result {
    margin-top: 25px;

    padding: 35px 20px;

    text-align: center;

    background: #f8fbff;

    border: 1px dashed #bae6fd;

    border-radius: 18px;
}

.empty-icon {
    font-size: 48px;

    margin-bottom: 10px;
}

.empty-title {
    font-size: 18px;

    font-weight: 800;

    color: #0f172a;
}

.empty-text {
    color: #64748b;

    font-size: 13px;

    margin-top: 6px;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {

    border-radius: 20px;

    padding: 30px;

    text-align: center;

    color: white;

    margin-top: 20px;
}

.high-risk {

    background:
        linear-gradient(
            135deg,
            #b91c1c,
            #ef4444
        );

    box-shadow:
        0 15px 35px rgba(239,68,68,0.22);
}

.low-risk {

    background:
        linear-gradient(
            135deg,
            #047857,
            #10b981
        );

    box-shadow:
        0 15px 35px rgba(16,185,129,0.22);
}

.result-icon {
    font-size: 52px;
}

.result-title {
    font-size: 25px;

    font-weight: 900;

    margin-top: 8px;
}

.result-percentage {

    font-size: 46px;

    font-weight: 900;

    margin-top: 5px;
}

.result-caption {

    font-size: 13px;

    opacity: 0.9;

    margin-top: 4px;
}


/* =========================================================
   RECOMMENDATION
   ========================================================= */

.recommendation {

    background: #f8fafc;

    border-radius: 14px;

    padding: 17px;

    margin-top: 15px;

    border-left: 5px solid #0284c7;
}

.recommendation-title {

    color: #075985;

    font-weight: 800;

    font-size: 14px;
}

.recommendation-text {

    color: #475569;

    font-size: 13px;

    margin-top: 5px;

    line-height: 1.5;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    padding: 30px 10px;

    margin-top: 30px;

    color: #64748b;

    font-size: 13px;
}

.footer-brand {

    color: #075985;

    font-size: 16px;

    font-weight: 850;

    margin-bottom: 5px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .hero {
        padding: 35px 25px;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-description {
        font-size: 14px;
    }

    .detail-grid {
        grid-template-columns: 1fr;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-content">

        <div class="hero-badge">
            🤖 MACHINE LEARNING • CUSTOMER ANALYTICS
        </div>

        <div class="hero-title">
            ✈️ Travel Customer Churn Predictor
        </div>

        <div class="hero-description">
            Predict whether a travel customer is likely to leave
            the service based on their profile and travel behaviour.
            Enter the customer details and receive an instant prediction.
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# QUICK OVERVIEW
# ============================================================

st.markdown("""
<div class="section-heading">
    📊 Customer Analysis
</div>

<div class="section-subheading">
    Review the customer information and generate a churn prediction.
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        ✈️
    </div>

    <div class="sidebar-title">
        Customer Details
    </div>

    <div class="sidebar-subtitle">
        Enter the customer's information below
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-section">PERSONAL INFORMATION</div>',
        unsafe_allow_html=True
    )

    age = st.slider(
        "🎂 Customer Age",
        min_value=18,
        max_value=75,
        value=31
    )

    st.markdown(
        '<div class="sidebar-section">TRAVEL BEHAVIOUR</div>',
        unsafe_allow_html=True
    )

    frequent = st.selectbox(
        "✈️ Frequent Flyer",
        ["No", "Yes"]
    )

    services = st.slider(
        "🎁 Services Opted",
        min_value=1,
        max_value=6,
        value=2
    )

    st.markdown(
        '<div class="sidebar-section">CUSTOMER DETAILS</div>',
        unsafe_allow_html=True
    )

    income = st.selectbox(
        "💰 Annual Income Class",
        ["Low Income", "Middle Income"]
    )

    social = st.selectbox(
        "📱 Social Media Connected",
        ["No", "Yes"]
    )

    hotel = st.selectbox(
        "🏨 Hotel Booking",
        ["No", "Yes"]
    )

    st.markdown("---")

    st.caption(
        "💡 Enter the details and use the prediction "
        "button to analyze the customer."
    )


# ============================================================
# MAIN TWO COLUMN LAYOUT
# ============================================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

with left_col:

    st.markdown("""
    <div class="profile-card">

        <div class="profile-header">

            <div class="profile-avatar">
                👤
            </div>

            <div>
                <div class="profile-heading">
                    Customer Profile
                </div>

                <div class="profile-subtitle">
                    Current customer information
                </div>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)

    # Detail grid displayed separately so Streamlit renders cleanly
    st.markdown(f"""
    <div class="profile-card" style="margin-top:-25px;">

        <div class="detail-grid">

            <div class="detail-item">
                <div class="detail-label">🎂 AGE</div>
                <div class="detail-value">{age} Years</div>
            </div>

            <div class="detail-item">
                <div class="detail-label">✈️ FREQUENT FLYER</div>
                <div class="detail-value">{frequent}</div>
            </div>

            <div class="detail-item">
                <div class="detail-label">💰 INCOME CLASS</div>
                <div class="detail-value">{income}</div>
            </div>

            <div class="detail-item">
                <div class="detail-label">🎁 SERVICES</div>
                <div class="detail-value">{services} / 6</div>
            </div>

            <div class="detail-item">
                <div class="detail-label">📱 SOCIAL MEDIA</div>
                <div class="detail-value">{social}</div>
            </div>

            <div class="detail-item">
                <div class="detail-label">🏨 HOTEL BOOKING</div>
                <div class="detail-value">{hotel}</div>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION PANEL
# ============================================================

with right_col:

    st.markdown("""
    <div class="prediction-panel">

        <div class="prediction-header">
            🎯 Churn Prediction
        </div>

        <div class="prediction-description">
            Use the trained machine learning model to analyze
            the customer's likelihood of churn.
        </div>

    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button(
        "🔮  Predict Churn Risk",
        use_container_width=True
    )

    if not predict_clicked:

        st.markdown("""
        <div class="empty-result">

            <div class="empty-icon">
                🔍
            </div>

            <div class="empty-title">
                Ready for Prediction
            </div>

            <div class="empty-text">
                Enter customer details from the sidebar
                and click the button above.
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_clicked:

        input_data = pd.DataFrame({
            "Age": [age],
            "FrequentFlyer": [frequent],
            "AnnualIncomeClass": [income],
            "ServicesOpted": [services],
            "AccountSyncedToSocialMedia": [social],
            "BookedHotelOrNot": [hotel]
        })


        # ----------------------------------------------------
        # ENCODING
        # ----------------------------------------------------

        input_data["FrequentFlyer"] = input_data[
            "FrequentFlyer"
        ].map({
            "No": 0,
            "Yes": 1
        })

        input_data["AnnualIncomeClass"] = input_data[
            "AnnualIncomeClass"
        ].map({
            "Low Income": 0,
            "Middle Income": 1
        })

        input_data["AccountSyncedToSocialMedia"] = input_data[
            "AccountSyncedToSocialMedia"
        ].map({
            "No": 0,
            "Yes": 1
        })

        input_data["BookedHotelOrNot"] = input_data[
            "BookedHotelOrNot"
        ].map({
            "No": 0,
            "Yes": 1
        })


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(input_data)[0]

            probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                if hasattr(model, "classes_"):

                    classes = list(model.classes_)

                    if prediction in classes:

                        index = classes.index(prediction)

                        probability = probabilities[index]

                elif len(probabilities) > 1:

                    probability = probabilities[1]


            # =================================================
            # HIGH CHURN
            # =================================================

            if prediction == 1:

                probability_text = (
                    f"{probability:.0%}"
                    if probability is not None
                    else "High"
                )

                st.markdown(f"""
                <div class="result-card high-risk">

                    <div class="result-icon">
                        🚨
                    </div>

                    <div class="result-title">
                        HIGH CHURN RISK
                    </div>

                    <div class="result-percentage">
                        {probability_text}
                    </div>

                    <div class="result-caption">
                        Estimated probability of churn
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="recommendation">

                    <div class="recommendation-title">
                        💡 Customer Retention Suggestion
                    </div>

                    <div class="recommendation-text">
                        This customer may require additional attention.
                        Consider providing personalized offers,
                        improved service options, or retention benefits.
                    </div>

                </div>
                """, unsafe_allow_html=True)


            # =================================================
            # LOW CHURN
            # =================================================

            else:

                if probability is not None:

                    stay_probability = 1 - probability

                    probability_text = (
                        f"{stay_probability:.0%}"
                    )

                else:

                    probability_text = "Low"

                st.markdown(f"""
                <div class="result-card low-risk">

                    <div class="result-icon">
                        ✅
                    </div>

                    <div class="result-title">
                        LOW CHURN RISK
                    </div>

                    <div class="result-percentage">
                        {probability_text}
                    </div>

                    <div class="result-caption">
                        Estimated likelihood of staying
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="recommendation">

                    <div class="recommendation-title">
                        ✨ Customer Status
                    </div>

                    <div class="recommendation-text">
                        The customer is predicted to have a lower
                        likelihood of leaving the service.
                    </div>

                </div>
                """, unsafe_allow_html=True)


        except Exception:

            # Do not show Python/backend details to the customer
            st.error(
                "Unable to generate the prediction. "
                "Please check the model configuration."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <div class="footer-brand">
        ✈️ Travel Customer Churn Predictor
    </div>

    <div>
        Customer Churn Prediction • Machine Learning Application
    </div>

    <div style="margin-top:8px;">
        Built with Python & Streamlit
    </div>

</div>
""", unsafe_allow_html=True)
