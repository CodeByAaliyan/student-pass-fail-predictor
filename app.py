
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="StudentIQ | Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "student_model.joblib"
DATA_PATH = BASE_DIR / "students.csv"

# =====================================================
# CUSTOM UI
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1020;
    color: #e8edf8;
}

[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.9);
}

[data-testid="stSidebar"] {
    background: #11182b;
    border-right: 1px solid #26314b;
}

[data-testid="stSidebar"] * {
    color: #e8edf8;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #182747, #20234a 60%, #183b50);
    border: 1px solid #334365;
    margin-bottom: 25px;
}

.eyebrow {
    color: #73e0d0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #ffffff;
    font-size: 36px;
    font-weight: 800;
    margin: 10px 0;
}

.hero p {
    color: #bdc9df;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #ffffff;
    margin-top: 12px;
    margin-bottom: 16px;
}

.metric-card {
    background: linear-gradient(145deg, #151f35, #11192a);
    border: 1px solid #293752;
    border-radius: 17px;
    padding: 20px;
    min-height: 125px;
    margin-bottom: 12px;
}

.metric-label {
    font-size: 12px;
    font-weight: 700;
    color: #a5b3cb;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 8px;
}

.metric-note {
    font-size: 12px;
    color: #73e0d0;
    margin-top: 5px;
}

.panel {
    background: #121b2e;
    border: 1px solid #293752;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.result-pass {
    background: linear-gradient(135deg, #103d36, #132c35);
    border: 1px solid #238b76;
    border-radius: 18px;
    padding: 25px;
}

.result-fail {
    background: linear-gradient(135deg, #49232f, #291c30);
    border: 1px solid #a64f69;
    border-radius: 18px;
    padding: 25px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.result-description {
    color: #d2dceb;
    font-size: 14px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(100deg, #6c7cff, #39cdbb);
    color: #081321;
    font-weight: 800;
    border: 0;
    border-radius: 11px;
    padding: 12px 18px;
    min-height: 46px;
    transition: all 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    filter: brightness(1.12);
    border: 0;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: #151f35;
    border: 1px solid #293752;
    padding: 15px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #a5b3cb;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stTabs [data-baseweb="tab"] {
    color: #c5d2e8;
}

hr {
    border-color: #293752;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD FILES
# =====================================================
@st.cache_resource
def load_model(path, modified_time):
    saved = joblib.load(path)

    if isinstance(saved, dict):
        model = saved["model"]
        features = saved.get("features", [
            "study_hours",
            "attendance",
            "previous_score",
            "assignments_completed",
            "sleep_hours"
        ])
    else:
        model = saved
        features = [
            "study_hours",
            "attendance",
            "previous_score",
            "assignments_completed",
            "sleep_hours"
        ]

    return model, features


@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Place student_model.joblib in the same folder as app.py.")
    st.stop()

try:
    model, features = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

df = None
if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception:
        df = None

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    st.markdown("## 🎓 StudentIQ")
    st.caption("STUDENT PERFORMANCE INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "🔮 Predict Result",
            "📊 Student Analytics",
            "🗂️ Dataset Explorer",
            "🧾 Prediction History"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Model loaded successfully")
    st.caption("Algorithm: Random Forest Classifier")
    st.caption("Data source: Synthetic learning dataset")
    st.divider()
    st.caption("StudentIQ • Educational ML project")

# =====================================================
# REUSABLE COMPONENTS
# =====================================================
def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def hero(title, description, eyebrow="STUDENT ANALYTICS PLATFORM"):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def get_result(student):
    student = student[features]
    prediction = model.predict(student)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student)[0]
        classes = list(model.classes_)
        confidence = float(
            probabilities[classes.index(prediction)] * 100
        )

    return str(prediction), confidence


# =====================================================
# OVERVIEW PAGE
# =====================================================
if page == "🏠 Overview":
    hero(
        "Understand performance. Predict possibilities.",
        "Explore student patterns, run predictions, and turn learning "
        "data into actionable insights."
    )

    total = len(df) if df is not None else 0

    if df is not None and "result" in df.columns and total:
        pass_count = int((df["result"] == "Pass").sum())
        fail_count = int((df["result"] == "Fail").sum())
        pass_rate = pass_count / total * 100
        fail_rate = fail_count / total * 100
    else:
        pass_count = fail_count = 0
        pass_rate = fail_rate = 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Sample students", f"{total:,}", "Available records")

    with c2:
        metric_card("Pass rate", f"{pass_rate:.1f}%", "Generated labels")

    with c3:
        metric_card("Fail rate", f"{fail_rate:.1f}%", "Generated labels")

    with c4:
        metric_card("Predictions", str(len(st.session_state.history)),
                    "This session")

    st.markdown('<div class="section-title">Explore your workspace</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <h3>🔮 Predict student outcomes</h3>
            <p>Enter study hours, attendance, previous scores,
            completed assignments, and sleep hours.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open prediction studio →", key="go_predict"):
            st.session_state.nav_target = "🔮 Predict Result"
            st.rerun()

    with right:
        st.markdown("""
        <div class="panel">
            <h3>📊 Explore student data</h3>
            <p>Review the generated dataset, compare features,
            and explore pass/fail patterns visually.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open analytics →", key="go_analytics"):
            st.session_state.nav_target = "📊 Student Analytics"
            st.rerun()

    if df is not None and "result" in df.columns:
        st.markdown('<div class="section-title">Outcome distribution</div>',
                    unsafe_allow_html=True)

        chart_data = (
            df["result"].value_counts()
            .reindex(["Pass", "Fail"], fill_value=0)
            .rename_axis("Outcome")
            .reset_index(name="Students")
        )

        st.bar_chart(chart_data, x="Outcome", y="Students",
                     color="#7285ff", use_container_width=True)

    st.info(
        "This demonstration uses synthetic data. Its statistics describe "
        "the generated dataset, not real students."
    )

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "🔮 Predict Result":
    hero(
        "Prediction Studio",
        "Configure student indicators and generate a model prediction.",
        "INTERACTIVE MACHINE LEARNING"
    )

    st.markdown('<div class="section-title">Student indicators</div>',
                unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.slider(
                "📚 Daily study hours", 0.0, 10.0, 5.0, 0.5
            )
            attendance = st.slider(
                "📅 Attendance (%)", 40.0, 100.0, 80.0, 1.0
            )
            previous_score = st.slider(
                "📝 Previous score", 20.0, 100.0, 70.0, 1.0
            )

        with col2:
            assignments = st.slider(
                "✅ Assignments completed", 0, 10, 7, 1
            )
            sleep_hours = st.slider(
                "🌙 Daily sleep hours", 3.0, 10.0, 7.0, 0.5
            )

            st.markdown("""
            <div class="panel">
                <b>What happens next?</b>
                <p>The saved Random Forest model evaluates these
                inputs and returns its predicted class.</p>
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "✨ Generate Prediction", use_container_width=True
        )

    if submitted:
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "assignments_completed": assignments,
            "sleep_hours": sleep_hours
        }])

        try:
            prediction, confidence = get_result(student)

            record = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Study Hours": study_hours,
                "Attendance": attendance,
                "Previous Score": previous_score,
                "Assignments": assignments,
                "Sleep Hours": sleep_hours,
                "Prediction": prediction,
                "Confidence (%)": round(confidence, 2)
                if confidence is not None else None
            }

            st.session_state.history.append(record)

            st.markdown('<div class="section-title">Prediction report</div>',
                        unsafe_allow_html=True)

            if prediction.strip().lower() == "pass":
                st.markdown("""
                <div class="result-pass">
                    <div class="eyebrow">MODEL OUTPUT</div>
                    <div class="result-title">PASS ✓</div>
                    <div class="result-description">
                        The model predicts the Pass class for these inputs.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif prediction.strip().lower() == "fail":
                st.markdown("""
                <div class="result-fail">
                    <div class="eyebrow">MODEL OUTPUT</div>
                    <div class="result-title">FAIL</div>
                    <div class="result-description">
                        The model predicts the Fail class for these inputs.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"Predicted class: {prediction}")

            if confidence is not None:
                st.metric("Model confidence", f"{confidence:.1f}%")
                st.progress(max(0.0, min(1.0, confidence / 100)))
                st.caption(
                    "Confidence is the model's estimated probability for "
                    "the predicted class; it is not a guarantee of success."
                )

            st.markdown('<div class="section-title">Input summary</div>',
                        unsafe_allow_html=True)
            st.dataframe(student, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# =====================================================
# ANALYTICS PAGE
# =====================================================
elif page == "📊 Student Analytics":
    hero(
        "Student Analytics",
        "Visualize the patterns and distributions in your training dataset.",
        "DATA EXPLORATION"
    )

    if df is None:
        st.warning("students.csv was not found. Add it beside app.py.")
        st.stop()

    required = {
        "study_hours", "attendance", "previous_score",
        "assignments_completed", "sleep_hours", "result"
    }

    if not required.issubset(df.columns):
        st.error("The dataset is missing one or more expected columns.")
        st.stop()

    a, b, c = st.columns(3)

    with a:
        metric_card("Total records", f"{len(df):,}", "Dataset size")
    with b:
        metric_card("Numeric features", str(len(df.select_dtypes(
            include=np.number).columns)), "Input variables")
    with c:
        metric_card("Outcome classes", str(df["result"].nunique()),
                    "Target categories")

    tab1, tab2, tab3 = st.tabs([
        "Outcome distribution", "Feature comparison", "Correlations"
    ])

    with tab1:
        outcomes = (
            df["result"].value_counts()
            .reindex(["Pass", "Fail"], fill_value=0)
            .rename_axis("Outcome")
            .reset_index(name="Students")
        )
        st.bar_chart(outcomes, x="Outcome", y="Students",
                     color="#43c6b7", use_container_width=True)

    with tab2:
        feature = st.selectbox(
            "Choose a feature",
            ["study_hours", "attendance", "previous_score",
             "assignments_completed", "sleep_hours"]
        )

        st.bar_chart(
            df.groupby("result")[feature].mean().reset_index(),
            x="result", y=feature,
            color="#7285ff", use_container_width=True
        )
        st.caption("Bars show mean feature values by generated outcome class.")

    with tab3:
        numeric = df.select_dtypes(include=np.number)
        st.dataframe(
            numeric.corr().round(2),
            use_container_width=True
        )
        st.caption("Correlation measures linear association, not causation.")

# =====================================================
# DATASET EXPLORER
# =====================================================
elif page == "🗂️ Dataset Explorer":
    hero(
        "Dataset Explorer",
        "Search, filter, inspect, and export the student records.",
        "DATA MANAGEMENT"
    )

    if df is None:
        st.warning("students.csv was not found beside app.py.")
        st.stop()

    search = st.text_input("🔎 Search records by outcome", placeholder="Pass or Fail")

    filtered = df.copy()

    if search.strip():
        filtered = filtered[
            filtered.astype(str).apply(
                lambda col: col.str.contains(search.strip(), case=False, na=False)
            ).any(axis=1)
        ]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="student_records.csv",
        mime="text/csv",
        use_container_width=True
    )

# =====================================================
# PREDICTION HISTORY
# =====================================================
elif page == "🧾 Prediction History":
    hero(
        "Prediction History",
        "Review and export predictions made during this app session.",
        "RECENT ACTIVITY"
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        c1, c2 = st.columns(2)
        with c1:
            metric_card("Predictions made", str(len(history_df)),
                        "Current session")
        with c2:
            metric_card(
                "Pass predictions",
                str((history_df["Prediction"].str.lower() == "pass").sum()),
                "Current session"
            )

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Export prediction history",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Visit Prediction Studio to get started.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "StudentIQ | Built with Streamlit and scikit-learn | "
    "For educational demonstration only"
)
