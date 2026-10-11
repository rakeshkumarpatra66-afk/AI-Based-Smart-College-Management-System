import sqlite3
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Smart College Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"


# ==========================================================
# PREMIUM UI DESIGN
# ==========================================================

NAVY = "#142B4A"
TEAL = "#13B8A6"
WHITE = "#FFFFFF"
BG = "#F3F6FB"
TEXT = "#26374A"


st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F6FB;
    }

    [data-testid="stSidebar"] {
        background-color: #142B4A;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(135deg, #142B4A, #245477);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(20,43,74,0.12);
    }

    .hero h1 {
        color: white;
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #DCEAF7;
        font-size: 15px;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #40E0D0;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #13B8A6;
        border-radius: 16px;
        padding: 20px;
        min-height: 130px;
        box-shadow: 0 5px 18px rgba(20,43,74,0.05);
    }

    .metric-label {
        color: #718096;
        font-size: 13px;
        font-weight: 700;
    }

    .metric-value {
        color: #142B4A;
        font-size: clamp(22px, 2vw, 30px);
        font-weight: 800;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }

    .section-title {
        color: #142B4A;
        font-size: 22px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #718096;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .panel {
        background: white;
        padding: 22px;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(20,43,74,0.04);
    }

    .footer {
        background: #142B4A;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin-top: 35px;
        color: white;
    }

    .stButton button,
    .stFormSubmitButton button {
        background-color: #142B4A;
        color: white;
        border: 1px solid #142B4A;
        border-radius: 9px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background-color: #13B8A6;
        color: white;
        border-color: #13B8A6;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 12px;
            padding-right: 12px;
        }

        .hero {
            padding: 20px;
        }

        .metric-card {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# DATABASE
# ==========================================================

def get_connection():
    return sqlite3.connect(
        str(DB_PATH),
        timeout=30
    )


def init_database():
    with get_connection() as conn:

        # Create the table only when it does not exist.
        # Existing records and tables are not dropped.

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                department TEXT NOT NULL,
                attendance REAL NOT NULL,
                assignment_marks REAL NOT NULL,
                previous_marks REAL NOT NULL
            )
            """
        )


init_database()


def load_students():

    with get_connection() as conn:

        return pd.read_sql_query(
            """
            SELECT
                id,
                name,
                email,
                department,
                attendance,
                assignment_marks,
                previous_marks
            FROM students
            ORDER BY id DESC
            """,
            conn
        )


# ==========================================================
# MACHINE LEARNING MODEL
# ==========================================================

# This training dataset is for the model demonstration.
# Real student records are NOT replaced by this dataset.

X = [
    [95, 90, 92],
    [90, 85, 88],
    [85, 88, 85],
    [80, 80, 82],
    [75, 75, 78],
    [70, 72, 74],
    [65, 68, 70],
    [60, 65, 63],
    [55, 55, 58],
    [50, 45, 48],
    [92, 95, 96],
    [88, 89, 90],
    [82, 84, 86],
    [78, 79, 80],
    [72, 73, 75],
    [68, 69, 68],
    [62, 61, 65],
    [58, 58, 55],
    [45, 50, 45],
    [40, 35, 38],
    [98, 98, 97],
    [93, 92, 94],
    [87, 86, 89],
    [81, 83, 84],
    [76, 77, 79],
    [71, 70, 72],
    [66, 64, 66],
    [59, 57, 60],
    [52, 49, 50],
    [48, 40, 42]
]

y = [
    "Good", "Good", "Good", "Good", "Good",
    "Good", "Average", "Average", "Average",
    "Needs Improvement", "Good", "Good", "Good",
    "Good", "Good", "Average", "Average",
    "Average", "Needs Improvement", "Needs Improvement",
    "Good", "Good", "Good", "Good", "Good",
    "Good", "Average", "Average", "Needs Improvement",
    "Needs Improvement"
]


@st.cache_resource
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    return model, accuracy


model, model_accuracy = train_model()


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def valid_email(email):

    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    return re.fullmatch(pattern, email) is not None


def section_title(title, subtitle=""):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def metric_card(icon, label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:25px">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_student_table(data):

    result = data.rename(
        columns={
            "id": "Student ID",
            "name": "Student Name",
            "email": "Email Address",
            "department": "Department",
            "attendance": "Attendance (%)",
            "assignment_marks": "Assignment Marks",
            "previous_marks": "Previous Marks"
        }
    ).copy()

    for col in [
        "Attendance (%)",
        "Assignment Marks",
        "Previous Marks"
    ]:

        if col in result.columns:
            result[col] = result[col].round(1)

    return result


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:18px 5px">
            <div style="font-size:45px">🎓</div>
            <h2 style="color:white;font-size:20px">
                AI SMART COLLEGE
            </h2>
            <p style="color:#DCEAF7;font-size:12px">
                MANAGEMENT SYSTEM
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Register Student",
            "View Students",
            "Attendance Management",
            "AI/ML Prediction"
        ],
        format_func=lambda value: {
            "Dashboard": "📊 Dashboard",
            "Register Student": "👨‍🎓 Register Student",
            "View Students": "📋 View Students",
            "Attendance Management": "📅 Attendance",
            "AI/ML Prediction": "🤖 AI/ML Prediction"
        }[value],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### ⚙️ Technology")

    st.caption("🐍 Python")
    st.caption("⚡ Streamlit")
    st.caption("🗄️ SQLite")
    st.caption("🧠 Scikit-learn")
    st.caption("📈 Plotly")


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            SMART CAMPUS • INTELLIGENT ANALYTICS
        </div>
        <h1>AI Smart College Management System</h1>
        <p>
            Manage students, monitor attendance, and analyze
            academic performance using machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


df = load_students()


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    section_title(
        "📊 Dashboard Overview",
        "Your college management system at a glance."
    )

    total_students = len(df)

    average_attendance = (
        df["attendance"].mean()
        if total_students else 0
    )

    average_marks = (
        (df["assignment_marks"].mean()
         + df["previous_marks"].mean()) / 2
        if total_students else 0
    )

    cards = st.columns(4)

    with cards[0]:
        metric_card(
            "👨‍🎓",
            "TOTAL STUDENTS",
            f"{total_students}"
        )

    with cards[1]:
        metric_card(
            "📅",
            "AVERAGE ATTENDANCE",
            f"{average_attendance:.1f}%"
        )

    with cards[2]:
        metric_card(
            "📚",
            "AVERAGE MARKS",
            f"{average_marks:.1f}%"
        )

    with cards[3]:
        metric_card(
            "🤖",
            "ML TEST ACCURACY",
            f"{model_accuracy:.1f}%"
        )

    st.write("")

    # Department chart

    left, right = st.columns(2)

    with left:

        section_title(
            "🏢 Department Distribution",
            "Students registered in each department."
        )

        if not df.empty:

            dept_data = (
                df["department"]
                .value_counts()
                .rename_axis("Department")
                .reset_index(name="Students")
            )

            fig = px.bar(
                dept_data,
                x="Department",
                y="Students",
                color="Students",
                text="Students",
                color_continuous_scale=[
                    "#B9F3E9",
                    "#13B8A6",
                    "#142B4A"
                ]
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                template="plotly_white",
                height=350,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=30, b=10)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("Register students to display this chart.")

    # Attendance chart

    with right:

        section_title(
            "📅 Attendance Statistics",
            "Attendance eligibility overview."
        )

        if not df.empty:

            eligible = int(
                (df["attendance"] >= 75).sum()
            )

            low = total_students - eligible

            attendance_data = pd.DataFrame(
                {
                    "Status": [
                        "Attendance ≥ 75%",
                        "Attendance < 75%"
                    ],
                    "Students": [
                        eligible,
                        low
                    ]
                }
            )

            fig = px.pie(
                attendance_data,
                names="Status",
                values="Students",
                hole=0.65,
                color="Status",
                color_discrete_map={
                    "Attendance ≥ 75%": "#13B8A6",
                    "Attendance < 75%": "#F59E0B"
                }
            )

            fig.update_layout(
                template="plotly_white",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("Attendance statistics will appear here.")

    # Performance chart

    section_title(
        "📈 Student Performance",
        "Average marks across academic assessments."
    )

    if not df.empty:

        performance_data = pd.DataFrame(
            {
                "Assessment": [
                    "Assignment Marks",
                    "Previous Marks"
                ],
                "Average Marks": [
                    df["assignment_marks"].mean(),
                    df["previous_marks"].mean()
                ]
            }
        )

        fig = px.bar(
            performance_data,
            x="Assessment",
            y="Average Marks",
            color="Assessment",
            text_auto=".1f",
            color_discrete_sequence=[
                "#13B8A6",
                "#142B4A"
            ]
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_yaxes(
            range=[0, 110]
        )

        fig.update_layout(
            template="plotly_white",
            height=320,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No student performance data available.")

    # Recent students

    section_title(
        "🆕 Recent Students",
        "The five most recently registered records."
    )

    if not df.empty:

        st.dataframe(
            format_student_table(df.head(5)),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No students have been registered yet.")


# ==========================================================
# REGISTER STUDENT
# ==========================================================

elif page == "Register Student":

    section_title(
        "👨‍🎓 Student Registration",
        "Add a student to the college management database."
    )

    with st.form("student_registration_form"):

        st.markdown("### Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Student Name *",
                placeholder="Enter full name"
            )

        with col2:
            email = st.text_input(
                "Email Address *",
                placeholder="student@example.com"
            )

        department = st.selectbox(
            "Department *",
            [
                "Computer Science",
                "Information Technology",
                "Mechanical Engineering",
                "Electrical Engineering",
                "Civil Engineering",
                "Electronics Engineering"
            ],
            index=None,
            placeholder="Select department"
        )

        st.divider()

    st.markdown(
    '<h3 style="color: black !important;">Academic Information</h3>',
    unsafe_allow_html=True
)


        col3, col4, col5 = st.columns(3)

        with col3:
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

        with col4:
            assignment_marks = st.number_input(
                "Assignment Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0
            )

        with col5:
            previous_marks = st.number_input(
                "Previous Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0
            )

        submitted = st.form_submit_button(
            "➕ Register Student",
            use_container_width=True
        )

    if submitted:

        name = name.strip()
        email = email.strip()

        if not name or not email or not department:

            st.error("Please complete all required fields.")

        elif not valid_email(email):

            st.error("Please enter a valid email address.")

        else:

            try:

                with get_connection() as conn:

                    conn.execute(
                        """
                        INSERT INTO students (
                            name,
                            email,
                            department,
                            attendance,
                            assignment_marks,
                            previous_marks
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            name,
                            email,
                            department,
                            attendance,
                            assignment_marks,
                            previous_marks
                        )
                    )

                st.success(
                    f"Student {name} registered successfully!"
                )

                st.rerun()

            except sqlite3.IntegrityError:

                st.error(
                    "This email already exists. "
                    "Please use another email address."
                )

            except sqlite3.Error as error:

                st.error(
                    f"Database error: {error}"
                )


# ==========================================================
# VIEW AND DELETE STUDENTS
# ==========================================================

elif page == "View Students":

    section_title(
        "📋 Student Records",
        "Search, review, and delete student records."
    )

    if not df.empty:

        search = st.text_input(
            "🔎 Search Students",
            placeholder="Search by name, email, or department"
        )

        filtered_df = df.copy()

        if search.strip():

            query = search.strip()

            mask = (
                filtered_df["name"].str.contains(
                    query,
                    case=False,
                    na=False
                )
                |
                filtered_df["email"].str.contains(
                    query,
                    case=False,
                    na=False
                )
                |
                filtered_df["department"].str.contains(
                    query,
                    case=False,
                    na=False
                )
            )

            filtered_df = filtered_df[mask]

        st.caption(
            f"Showing {len(filtered_df)} of {len(df)} students."
        )

        st.dataframe(
            format_student_table(filtered_df),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        section_title(
            "🗑️ Delete Student",
            "Deletion permanently removes the selected database record."
        )

        with get_connection() as conn:

            student_rows = conn.execute(
                """
                SELECT id, name, email
                FROM students
                ORDER BY id DESC
                """
            ).fetchall()

        if student_rows:

            student_options = {
                f"{row[0]} - {row[1]} ({row[2]})": row[0]
                for row in student_rows
            }

            with st.form("delete_student_form"):

                selected_student = st.selectbox(
                    "Select Student",
                    list(student_options.keys())
                )

                confirm = st.checkbox(
                    "I confirm that I want to delete this record."
                )

                delete_button = st.form_submit_button(
                    "🗑️ Delete Selected Student",
                    use_container_width=True
                )

            if delete_button:

                if not confirm:

                    st.warning(
                        "Please confirm before deleting."
                    )

                else:

                    try:

                        with get_connection() as conn:

                            conn.execute(
                                "DELETE FROM students WHERE id = ?",
                                (
                                    student_options[selected_student],
                                )
                            )

                        st.success(
                            "Student deleted successfully."
                        )

                        st.rerun()

                    except sqlite3.Error as error:

                        st.error(
                            f"Unable to delete student: {error}"
                        )

    else:

        st.info(
            "No students found. Register a student first."
        )


# ==========================================================
# ATTENDANCE MANAGEMENT
# ==========================================================

elif page == "Attendance Management":

    section_title(
        "📅 Attendance Management",
        "Monitor attendance percentages and identify low-attendance students."
    )

    st.info(
        "The 75% threshold is used as an eligibility indicator. "
        "Confirm your institution's actual attendance policy separately."
    )

    if not df.empty:

        eligible_count = int(
            (df["attendance"] >= 75).sum()
        )

        low_count = len(df) - eligible_count

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "👨‍🎓",
                "TOTAL STUDENTS",
                str(len(df))
            )

        with c2:
            metric_card(
                "✅",
                "ELIGIBLE BY PERCENTAGE",
                str(eligible_count)
            )

        with c3:
            metric_card(
                "⚠️",
                "LOW ATTENDANCE",
                str(low_count)
            )

        attendance_table = df[
            [
                "id",
                "name",
                "department",
                "attendance"
            ]
        ].copy()

        attendance_table["Status"] = attendance_table[
            "attendance"
        ].apply(
            lambda value:
                "Eligible"
                if value >= 75
                else "Low Attendance"
        )

        attendance_table = attendance_table.rename(
            columns={
                "id": "Student ID",
                "name": "Student Name",
                "department": "Department",
                "attendance": "Attendance (%)"
            }
        )

        attendance_table["Attendance (%)"] = (
            attendance_table["Attendance (%)"].round(1)
        )

        attendance_table = attendance_table.sort_values(
            "Attendance (%)"
        )

        st.dataframe(
            attendance_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Attendance (%)": st.column_config.ProgressColumn(
                    "Attendance",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                )
            }
        )

        section_title(
            "📊 Attendance Distribution",
            "Student attendance distribution and the 75% threshold."
        )

        fig = px.histogram(
            df,
            x="attendance",
            nbins=10,
            color_discrete_sequence=["#13B8A6"],
            labels={
                "attendance": "Attendance (%)",
                "count": "Students"
            }
        )

        fig.add_vline(
            x=75,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="75% threshold"
        )

        fig.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No student records available for attendance analysis."
        )


# ==========================================================
# AI/ML PERFORMANCE PREDICTION
# ==========================================================

elif page == "AI/ML Prediction":

    section_title(
        "🤖 AI Student Performance Prediction",
        "Predict an academic performance category using Random Forest."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### Student Academic Inputs")

        with st.form("prediction_form"):

            input_attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

            input_assignment = st.number_input(
                "Assignment Marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

            input_previous = st.number_input(
                "Previous Marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

            predict_button = st.form_submit_button(
                "🤖 Predict Performance",
                use_container_width=True
            )

    with right:

        st.markdown("### Model Information")

        metric_card(
            "🎯",
            "HELD-OUT TEST ACCURACY",
            f"{model_accuracy:.1f}%"
        )

        st.write("")

        st.markdown(
            """
            **Algorithm:** Random Forest Classifier

            **Number of trees:** 100

            **Training split:** 75%

            **Test split:** 25%

            **Features:** Attendance, assignment marks,
            and previous marks.
            """
        )

        st.caption(
            "Accuracy is calculated using the held-out subset "
            "of the application's existing sample dataset. "
            "It does not guarantee real-world accuracy."
        )

    if predict_button:

        input_data = pd.DataFrame(
            [
                {
                    "attendance": input_attendance,
                    "assignment_marks": input_assignment,
                    "previous_marks": input_previous
                }
            ]
        )

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        probability_df = pd.DataFrame(
            {
                "Category": model.classes_,
                "Probability (%)": probabilities * 100
            }
        )

        st.divider()

        section_title(
            "🎯 Prediction Result",
            "Estimated student performance category."
        )

        result1, result2 = st.columns(2)

        with result1:

            if prediction == "Good":

                st.success(
                    f"### 🟢 {prediction}"
                )

            elif prediction == "Average":

                st.warning(
                    f"### 🟡 {prediction}"
                )

            else:

                st.error(
                    f"### 🔴 {prediction}"
                )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        PREDICTED PERFORMANCE
                    </div>
                    <div class="metric-value">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result2:

            st.markdown("### Input Summary")

            st.write(
                f"Attendance: {input_attendance:.1f}%"
            )

            st.progress(int(input_attendance))

            st.write(
                f"Assignment marks: {input_assignment:.1f}%"
            )

            st.progress(int(input_assignment))

            st.write(
                f"Previous marks: {input_previous:.1f}%"
            )

            st.progress(int(input_previous))

        section_title(
            "📊 Model Class Probabilities",
            "Probabilities assigned by the trained classifier."
        )

        fig = px.bar(
            probability_df.sort_values("Probability (%)"),
            x="Probability (%)",
            y="Category",
            orientation="h",
            text="Probability (%)",
            color="Category",
            color_discrete_map={
                "Good": "#13B8A6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444"
            }
        )

        fig.update_traces(
            texttemplate="%{x:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_white",
            height=300,
            xaxis_range=[0, 110]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        predicted_probability = probabilities[
            list(model.classes_).index(prediction)
        ]

        st.caption(
            f"The model assigns {predicted_probability:.1%} "
            "probability to the predicted class. This is a model "
            "probability, not a calibrated guarantee of correctness."
        )

        st.info(
            "This is an educational prediction based on the "
            "existing sample training dataset. It should not "
            "be treated as a guaranteed academic outcome."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        <div style="font-size:18px;font-weight:800">
            🎓 AI Smart College Management System
        </div>
        <div style="margin-top:8px;color:#DCEAF7">
            Developed for AI/ML Internship Project
        </div>
        <div style="margin-top:6px;color:#DCEAF7;font-size:12px">
            Python • Streamlit • SQLite • Machine Learning • Plotly
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
