import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Travel Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

* {
    box-sizing: border-box;
}

.main {
    background: linear-gradient(
        135deg,
        #f0f7ff 0%,
        #e6f2ff 50%,
        #f0f7ff 100%
    );
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}


/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f8fbff 100%
    );

    border-right: 2px solid #e0e8f5;
}

.sidebar-header {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0066ff;
    margin-bottom: 1rem;
}


/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #f8fbff 100%
    );

    padding: 2.5rem 2rem;
    border-radius: 20px;

    box-shadow:
        0 10px 40px rgba(59, 130, 246, 0.08);

    border: 2px solid #e0e8f5;

    margin-bottom: 2rem;

    text-align: center;
}

.hero h1 {
    color: #0066ff;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.7rem;
}

.hero p {
    color: #64748b;
    font-size: 1.1rem;
    font-weight: 500;
    line-height: 1.6;
}


/* ================= METRIC CARDS ================= */

.metric-card {
    background: white;

    padding: 1.5rem;

    border-radius: 16px;

    box-shadow:
        0 8px 24px rgba(59, 130, 246, 0.08);

    border: 1.5px solid #e0e8f5;

    text-align: center;

    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-4px);

    box-shadow:
        0 12px 36px rgba(59, 130, 246, 0.15);

    border-color: #0066ff;
}

.metric-label {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0066ff;
    margin-top: 0.4rem;
}


/* ================= CARDS ================= */

.card {
    background: white;

    padding: 2rem;

    border-radius: 18px;

    box-shadow:
        0 8px 24px rgba(59, 130, 246, 0.08);

    border: 1.5px solid #e0e8f5;

    margin-bottom: 1rem;
}

.card h3 {
    color: #1e3a8a;
    font-size: 1.35rem;
    font-weight: 700;
}


/* ================= CUSTOMER INFORMATION ================= */

.customer-info {
    font-size: 1.05rem;

    color: #1e293b;

    margin: 0.8rem 0;

    font-weight: 500;

    padding: 0.8rem;

    background:
        linear-gradient(
            135deg,
            #f0f7ff 0%,
            #e6f2ff 100%
        );

    border-left: 4px solid #0066ff;

    border-radius: 8px;
}


/* ================= BUTTON ================= */

.stButton > button {

    background:
        linear-gradient(
            135deg,
            #0066ff 0%,
            #00a3ff 100%
        );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 0.8rem 2rem;

    font-weight: 700;

    font-size: 1.05rem;

    box-shadow:
        0 8px 20px rgba(0, 102, 255, 0.3);

    transition: 0.3s;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 32px rgba(0, 102, 255, 0.5);
}


/* ================= RESULT BOX ================= */

.result-box {

    padding: 1.8rem;

    border-radius: 16px;

    color: white;

    font-weight: 700;

    text-align: center;

    margin-top: 1rem;
}

.result-high {

    background:
        linear-gradient(
            135deg,
            #ef4444 0%,
            #dc2626 100%
        );

    box-shadow:
        0 12px 32px rgba(239, 68, 68, 0.3);
}

.result-low {

    background:
        linear-gradient(
            135deg,
            #10b981 0%,
            #059669 100%
        );

    box-shadow:
        0 12px 32px rgba(16, 185, 129, 0.3);
}


/* ================= FOOTER ================= */

.footer {

    text-align: center;

    color: #64748b;

    padding: 2rem 0;

    font-size: 0.95rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "model.pkl"


if not model_path.exists():

    st.error("❌ Model file not found.")

    st.info(
        "Please make sure that model.pkl is in the same folder as app.py."
    )

    st.stop()


try:

    model = joblib.load(model_path)

except Exception as e:

    st.error("❌ Error loading model.")

    st.code(str(e))

    st.stop()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <h1>
        ✈️ Travel Customer Churn Predictor
    </h1>

    <p>
        Machine learning application that predicts whether
        a travel customer is likely to leave the service.
        Enter customer details and click Predict Churn
        to generate a prediction.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT INFORMATION
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

        <div style="color:#64748b;">
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

        <div style="color:#64748b;">
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

        <div style="color:#64748b;">
            Random Forest
        </div>

    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ============================================================
# SIDEBAR - CUSTOMER INPUT
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-header">
        📋 Customer Profile
    </div>
    """, unsafe_allow_html=True)

    st.write("Enter customer details below:")

    age = st.slider(
        "Age 🎂",
        min_value=18,
        max_value=75,
        value=31,
        step=1,
        help="Customer age"
    )

    frequent = st.selectbox(
        "Frequent Flyer ✈️",
        ["No", "Yes"],
        help="Is the customer a frequent flyer?"
    )

    income = st.selectbox(
        "Income Class 💰",
        ["Low Income", "Middle Income"],
        help="Customer annual income class"
    )

    services = st.slider(
        "Services Opted 🎁",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help="Number of services used by customer"
    )

    social = st.selectbox(
        "Social Media Sync 📱",
        ["No", "Yes"],
        help="Is the account synced with social media?"
    )

    hotel = st.selectbox(
        "Booked Hotel 🏨",
        ["No", "Yes"],
        help="Has the customer booked a hotel?"
    )

    st.info(
        "Enter all customer details and click "
        "'Predict Churn Risk'."
    )


# ============================================================
# MAIN SECTION
# ============================================================

st.markdown(
    "<h2 style='color:#0a1428;'>Customer Information & Prediction</h2>",
    unsafe_allow_html=True
)


left_col, right_col = st.columns(
    [1.3, 1],
    gap="medium"
)


# ============================================================
# CUSTOMER SUMMARY
# ============================================================

with left_col:

    st.markdown("""
    <div class="card">

        <h3>
            👤 Customer Profile Summary
        </h3>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Age:</strong> {age} years old
        </div>
        """,
        unsafe_allow_html=True
    )

    travel_status = (
        "✈️ Frequent Flyer"
        if frequent == "Yes"
        else "🚫 Occasional Traveler"
    )

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Travel Status:</strong> {travel_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Income Class:</strong> {income}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Services Opted:</strong>
            {services} out of 6 services
        </div>
        """,
        unsafe_allow_html=True
    )

    social_status = (
        "✅ Connected"
        if social == "Yes"
        else "❌ Not Connected"
    )

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Social Media Sync:</strong>
            {social_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    hotel_status = (
        "✅ Has booked"
        if hotel == "Yes"
        else "❌ No booking"
    )

    st.markdown(
        f"""
        <div class="customer-info">
            <strong>Hotel Booking:</strong>
            {hotel_status}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

with right_col:

    st.markdown("""
    <div class="card">

        <h3>
            🎯 Churn Prediction
        </h3>

    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button(
        "🔮 Predict Churn Risk",
        use_container_width=True
    )

    if predict_clicked:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
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
        # No  = 0
        # Yes = 1
        #
        # Low Income    = 0
        # Middle Income = 1
        #
        # These mappings correspond to the alphabetical
        # LabelEncoder ordering used by the original code.
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
        # MAKE PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(input_data)[0]


            # ------------------------------------------------
            # GET PROBABILITY
            # ------------------------------------------------

            probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                if hasattr(model, "classes_"):

                    classes = list(model.classes_)

                    if prediction in classes:

                        prediction_index = classes.index(
                            prediction
                        )

                        probability = probabilities[
                            prediction_index
                        ]

                else:

                    probability = probabilities[1]


            # ------------------------------------------------
            # DISPLAY HIGH CHURN
            # ------------------------------------------------

            if prediction == 1:

                if probability is not None:
                    probability_text = f"{probability:.0%}"
                else:
                    probability_text = "High"

                st.markdown(
                    f"""
                    <div class="result-box result-high">

                        <div style="
                            font-size:3rem;
                            margin-bottom:0.5rem;
                        ">
                            🚨
                        </div>

                        <div style="
                            font-size:1.4rem;
                        ">
                            HIGH CHURN RISK
                        </div>

                        <div style="
                            font-size:2rem;
                            margin-top:0.5rem;
                        ">
                            {probability_text}
                        </div>

                        <div style="
                            font-size:0.9rem;
                            margin-top:1rem;
                        ">
                            Probability of churn
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.warning(
                    "⚠️ Action Required: "
                    "This customer shows indicators "
                    "associated with churn."
                )


            # ------------------------------------------------
            # DISPLAY LOW CHURN
            # ------------------------------------------------

            else:

                if probability is not None:

                    stay_probability = 1 - probability

                    probability_text = (
                        f"{stay_probability:.0%}"
                    )

                else:

                    probability_text = "Low"

                st.markdown(
                    f"""
                    <div class="result-box result-low">

                        <div style="
                            font-size:3rem;
                            margin-bottom:0.5rem;
                        ">
                            ✅
                        </div>

                        <div style="
                            font-size:1.4rem;
                        ">
                            LOW CHURN RISK
                        </div>

                        <div style="
                            font-size:2rem;
                            margin-top:0.5rem;
                        ">
                            {probability_text}
                        </div>

                        <div style="
                            font-size:0.9rem;
                            margin-top:1rem;
                        ">
                            Likelihood to stay
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "✨ The model predicts that this "
                    "customer has a lower churn risk."
                )


        except Exception as e:

            st.error(
                "❌ Error making prediction."
            )

            st.code(str(e))

            st.info(
                "Please check whether the input columns "
                "and encoding match the model used during training."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div class="footer">

    <p>
        🔒 <strong>Customer Churn Prediction</strong>
    </p>

    <p style="margin-top:0.5rem;">
        Powered by Random Forest Machine Learning
        • Built with Streamlit
    </p>

    <p style="
        margin-top:1rem;
        font-size:0.85rem;
    ">
        Travel Customer Churn Predictor
    </p>

</div>
""", unsafe_allow_html=True)
