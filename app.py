# app.py
import sqlite3
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Smart College Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"

# ============================================================
# PREMIUM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #102A43;
        --navy-light: #173F5F;
        --teal: #0D9488;
        --teal-light: #CCFBF1;
        --background: #F4F7FB;
        --white: #FFFFFF;
        --text: #243B53;
        --muted: #627D98;
        --border: #E2E8F0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,0.95);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102A43 0%, #173F5F 100%);
        border-right: 1px solid #274C68;
    }

    [data-testid="stSidebar"] * {
        color: #F0F9FF;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #D9EAF5;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 10px 12px;
        border-radius: 10px;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.12);
    }

    .brand-box {
        padding: 22px 15px;
        margin: 8px 0 25px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        text-align: center;
    }

    .brand-icon {
        font-size: 40px;
        margin-bottom: 8px;
    }

    .brand-title {
        font-size: 18px;
        font-weight: 800;
        color: white;
        margin: 0;
    }

    .brand-caption {
        font-size: 11px;
        color: #99F6E4;
        margin-top: 8px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .top-header {
        background: linear-gradient(115deg, #102A43 0%, #173F5F 65%, #0D9488 100%);
        padding: 28px 30px;
        border-radius: 20px;
        margin-bottom: 26px;
        box-shadow: 0 12px 28px rgba(16,42,67,0.13);
    }

    .top-header h1 {
        color: #FFFFFF !important;
        font-size: clamp(22px, 3vw, 34px);
        font-weight: 800;
        margin: 0 0 10px;
        letter-spacing: -1px;
    }

    .top-header p {
        color: #D9F7F4 !important;
        font-size: 14px;
        margin: 0;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 800;
        color: #102A43;
        margin: 15px 0 5px;
    }

    .section-description {
        font-size: 13px;
        color: #627D98;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 20px 16px;
        box-shadow: 0 5px 18px rgba(16,42,67,0.045);
        border-top: 4px solid #0D9488;
    }

    div[data-testid="stMetricLabel"] {
        color: #627D98;
        font-size: 13px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #102A43;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 12px;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px;
        margin: 8px 0 18px;
        box-shadow: 0 5px 18px rgba(16,42,67,0.035);
    }

    .panel-title {
        color: #102A43;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        color: #627D98;
        font-size: 12px;
        margin-bottom: 16px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #0D9488;
        color: #FFFFFF;
        border: 1px solid #0D9488;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #0F766E;
        color: #FFFFFF;
        border-color: #0F766E;
        box-shadow: 0 5px 14px rgba(13,148,136,0.2);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        border-radius: 9px;
        border-color: #CBD5E1;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
    }

    .status-good {
        display: inline-block;
        padding: 6px 12px;
        background: #DCFCE7;
        color: #166534;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-warning {
        display: inline-block;
        padding: 6px 12px;
        background: #FEF3C7;
        color: #92400E;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-danger {
        display: inline-block;
        padding: 6px 12px;
        background: #FEE2E2;
        color: #991B1B;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .prediction-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 26px;
        margin: 18px 0;
        box-shadow: 0 6px 20px rgba(16,42,67,0.06);
    }

    .prediction-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #627D98;
    }

    .prediction-value {
        font-size: 32px;
        font-weight: 800;
        color: #102A43;
        margin-top: 8px;
    }

    .footer {
        margin-top: 45px;
        padding: 22px 15px;
        background: #102A43;
        border-radius: 15px;
        text-align: center;
    }

    .footer p {
        color: #D9EAF5;
        font-size: 12px;
        margin: 4px;
    }

    .footer strong {
        color: #99F6E4;
    }

    @media (max-width: 768px) {
        .top-header {
            padding: 22px 18px;
            border-radius: 14px;
        }

        .top-header h1 {
            font-size: 23px;
        }

        [data-testid="stMetric"] {
            padding: 14px;
        }

        .panel {
            padding: 15px;
        }

        [data-testid="stDataFrame"] {
            overflow-x: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATABASE
# Preserve the existing database path, schema, and records.
# ============================================================

def get_db():
    connection = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as connection:
        connection.execute(
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


init_db()


def get_students():
    with get_db() as connection:
        return connection.execute(
            "SELECT * FROM students ORDER BY id DESC"
        ).fetchall()


def students_dataframe():
    records = get_students()
    if not records:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "email",
                "department",
                "attendance",
                "assignment_marks",
                "previous_marks",
            ]
        )

    return pd.DataFrame([dict(row) for row in records])


# ============================================================
# MACHINE LEARNING
# Preserve existing training data, classifier, and labels.
# ============================================================

X = [
    [95, 90, 92], [90, 85, 88], [85, 88, 85], [80, 80, 82],
    [75, 75, 78], [70, 72, 74], [65, 68, 70], [60, 65, 63],
    [55, 55, 58], [50, 45, 48], [92, 95, 96], [88, 89, 90],
    [82, 84, 86], [78, 79, 80], [72, 73, 75], [68, 69, 68],
    [62, 61, 65], [58, 58, 55], [45, 50, 45], [40, 35, 38],
    [98, 98, 97], [93, 92, 94], [87, 86, 89], [81, 83, 84],
    [76, 77, 79], [71, 70, 72], [66, 64, 66], [59, 57, 60],
    [52, 49, 50], [48, 40, 42],
]

y = [
    "Good", "Good", "Good", "Good", "Good", "Good", "Average",
    "Average", "Average", "Needs Improvement", "Good", "Good",
    "Good", "Good", "Good", "Average", "Average", "Average",
    "Needs Improvement", "Needs Improvement", "Good", "Good",
    "Good", "Good", "Good", "Good", "Average", "Average",
    "Needs Improvement", "Needs Improvement",
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

model_accuracy = accuracy_score(
    y_test,
    model.predict(X_test),
) * 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section_heading(title, description=""):
    st.markdown(
        f'<div class="section-heading">{title}</div>'
        f'<div class="section-description">{description}</div>',
        unsafe_allow_html=True,
    )


def plot_layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, sans-serif",
            color="#486581",
            size=12,
        ),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def attendance_status(value):
    if value >= 75:
        return "Eligible"
    return "Low Attendance"


def performance_prediction(attendance, assignment, previous):
    features = [[attendance, assignment, previous]]
    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities)) * 100

    return prediction, confidence


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-icon">🎓</div>
            <div class="brand-title">AI Smart College</div>
            <div class="brand-caption">Management Portal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Navigate to",
        [
            "📊 Dashboard",
            "👨‍🎓 Register Student",
            "📋 Students",
            "📅 Attendance",
            "🤖 AI Prediction",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("**TECHNOLOGY STACK**")
    st.caption("🐍 Python")
    st.caption("🎨 Streamlit")
    st.caption("🗄️ SQLite")
    st.caption("🧠 Scikit-learn")
    st.caption("📈 Plotly")

    st.markdown("---")
    st.caption("AI/ML Internship Project")


# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown(
    """
    <div class="top-header">
        <h1>🎓 AI Smart College Management System</h1>
        <p>
            Intelligent student management, academic analytics,
            attendance monitoring and machine learning insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load existing database records.
df = students_dataframe()


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    section_heading(
        "Dashboard Overview",
        "Monitor student records, academic performance and attendance.",
    )

    total_students = len(df)

    avg_attendance = (
        float(df["attendance"].mean())
        if total_students
        else 0.0
    )

    avg_marks = (
        float(
            (
                df["assignment_marks"]
                + df["previous_marks"]
            ).mean() / 2
        )
        if total_students
        else 0.0
    )

    eligible_students = (
        int((df["attendance"] >= 75).sum())
        if total_students
        else 0
    )

    metric_columns = st.columns(4)

    with metric_columns[0]:
        st.metric(
            "👨‍🎓 Total Students",
            f"{total_students:,}",
        )

    with metric_columns[1]:
        st.metric(
            "📅 Avg. Attendance",
            f"{avg_attendance:.1f}%",
        )

    with metric_columns[2]:
        st.metric(
            "📚 Average Marks",
            f"{avg_marks:.1f}%",
        )

    with metric_columns[3]:
        st.metric(
            "🤖 ML Test Accuracy",
            f"{model_accuracy:.1f}%",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if total_students:

        left_chart, right_chart = st.columns(2)

        with left_chart:

            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">🏢 Department Distribution</div>
                    <div class="panel-subtitle">
                        Student count by academic department
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            department_counts = (
                df["department"]
                .value_counts()
                .rename_axis("Department")
                .reset_index(name="Students")
            )

            fig = px.pie(
                department_counts,
                names="Department",
                values="Students",
                hole=0.62,
                color_discrete_sequence=[
                    "#102A43",
                    "#0D9488",
                    "#38BDF8",
                    "#818CF8",
                    "#F59E0B",
                    "#F472B6",
                ],
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                hole=0.62,
            )

            fig = plot_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

        with right_chart:

            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">📅 Attendance Overview</div>
                    <div class="panel-subtitle">
                        Eligible and low-attendance students
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            attendance_counts = pd.DataFrame(
                {
                    "Attendance Status": [
                        "Eligible (75%+)",
                        "Low Attendance",
                    ],
                    "Students": [
                        eligible_students,
                        total_students - eligible_students,
                    ],
                }
            )

            fig = px.bar(
                attendance_counts,
                x="Attendance Status",
                y="Students",
                color="Attendance Status",
                text="Students",
                color_discrete_map={
                    "Eligible (75%+)": "#0D9488",
                    "Low Attendance": "#F59E0B",
                },
            )

            fig.update_traces(
                textposition="outside",
                marker_line_width=0,
            )

            fig.update_layout(
                showlegend=False,
                yaxis_title="Number of Students",
                xaxis_title="",
                yaxis=dict(rangemode="tozero"),
            )

            fig = plot_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📈 Student Performance Analysis</div>
                <div class="panel-subtitle">
                    Assignment marks compared with previous marks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = px.scatter(
            df,
            x="assignment_marks",
            y="previous_marks",
            color="department",
            size="attendance",
            hover_name="name",
            hover_data={
                "email": True,
                "attendance": True,
            },
            labels={
                "assignment_marks": "Assignment Marks",
                "previous_marks": "Previous Marks",
                "department": "Department",
                "attendance": "Attendance (%)",
            },
        )

        fig.update_layout(
            xaxis=dict(range=[0, 100], title="Assignment Marks"),
            yaxis=dict(range=[0, 100], title="Previous Marks"),
        )

        fig = plot_layout(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🆕 Recently Registered Students</div>
                <div class="panel-subtitle">
                    Latest student records from your SQLite database
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        recent = df.head(5)[
            ["id", "name", "email", "department", "attendance"]
        ].copy()

        recent.columns = [
            "Student ID",
            "Student Name",
            "Email Address",
            "Department",
            "Attendance (%)",
        ]

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "Your database currently contains no student records. "
            "Register a student to populate the dashboard."
        )

    st.caption(
        f"Model accuracy is measured on the existing held-out sample test set: "
        f"{model_accuracy:.1f}%. It is not a guarantee of real-world performance."
    )


# ============================================================
# STUDENT REGISTRATION
# ============================================================

elif page == "👨‍🎓 Register Student":

    section_heading(
        "Student Registration",
        "Add a student record to your existing college database.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">📝 Student Information</div>
            <div class="panel-subtitle">
                Enter accurate academic and contact information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    departments = [
        "Computer Science",
        "Information Technology",
        "Mechanical Engineering",
        "Electrical Engineering",
        "Civil Engineering",
        "Electronics Engineering",
    ]

    with st.form("student_registration", clear_on_submit=True):

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Student Full Name *",
                placeholder="Enter full name",
                max_chars=100,
            )

            email = st.text_input(
                "Email Address *",
                placeholder="student@example.com",
                max_chars=254,
            )

            department = st.selectbox(
                "Department *",
                departments,
                index=None,
                placeholder="Select department",
            )

        with col2:
            attendance = st.number_input(
                "Attendance (%) *",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=0.5,
            )

            assignment = st.number_input(
                "Assignment Marks *",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
            )

            previous = st.number_input(
                "Previous Marks *",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
            )

        st.caption("* Required fields")

        submitted = st.form_submit_button(
            "➕ Register Student",
            use_container_width=True,
        )

    if submitted:

        name = name.strip()
        email = email.strip()

        if not name or not email or not department:
            st.error("Please complete all required fields.")

        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            st.error("Enter a valid email address.")

        elif not any(character.isalpha() for character in name):
            st.error("Enter a valid student name.")

        else:
            try:
                with get_db() as connection:
                    connection.execute(
                        """
                        INSERT INTO students
                        (
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
                            assignment,
                            previous,
                        ),
                    )

                st.success(
                    f"Student '{name}' registered successfully!"
                )

                st.rerun()

            except sqlite3.IntegrityError:
                st.error(
                    "This email address already exists in the database."
                )

            except sqlite3.Error as exc:
                st.error(f"Database error: {exc}")


# ============================================================
# STUDENT RECORDS AND DELETE
# ============================================================

elif page == "📋 Students":

    section_heading(
        "Student Records",
        "Search, review and manage student records.",
    )

    if not df.empty:

        search = st.text_input(
            "🔍 Search students",
            placeholder="Search by name, email or department",
        )

        filtered_df = df.copy()

        if search.strip():
            mask = (
                filtered_df["name"].astype(str).str.contains(
                    search.strip(), case=False, na=False
                )
                | filtered_df["email"].astype(str).str.contains(
                    search.strip(), case=False, na=False
                )
                | filtered_df["department"].astype(str).str.contains(
                    search.strip(), case=False, na=False
                )
            )

            filtered_df = filtered_df[mask]

        display_df = filtered_df.rename(
            columns={
                "id": "Student ID",
                "name": "Student Name",
                "email": "Email Address",
                "department": "Department",
                "attendance": "Attendance (%)",
                "assignment_marks": "Assignment Marks",
                "previous_marks": "Previous Marks",
            }
        )

        st.markdown(
            f"**Showing {len(filtered_df)} of {len(df)} students**"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Export Visible Students to CSV",
            data=csv_data,
            file_name="student_records.csv",
            mime="text/csv",
        )

        st.markdown("---")

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🗑️ Delete Student</div>
                <div class="panel-subtitle">
                    Deletion permanently removes the selected record.
                    Check the student details before confirming.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        options = {
            f"{row['id']} — {row['name']} ({row['email']})": row["id"]
            for _, row in df.iterrows()
        }

        selected_label = st.selectbox(
            "Select a student",
            list(options.keys()),
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to permanently delete this student."
        )

        if st.button(
            "🗑️ Delete Selected Student",
            disabled=not confirm_delete,
        ):

            try:
                with get_db() as connection:
                    connection.execute(
                        "DELETE FROM students WHERE id = ?",
                        (options[selected_label],),
                    )

                st.success("Student deleted successfully.")
                st.rerun()

            except sqlite3.Error as exc:
                st.error(f"Unable to delete student: {exc}")

    else:
        st.info("No student records found. Register a student first.")


# ============================================================
# ATTENDANCE MANAGEMENT
# ============================================================

elif page == "📅 Attendance":

    section_heading(
        "Attendance Management",
        "Monitor attendance percentages and the 75% eligibility threshold.",
    )

    if not df.empty:

        eligible_count = int((df["attendance"] >= 75).sum())
        low_count = len(df) - eligible_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", len(df))
        col2.metric("Eligible Students", eligible_count)
        col3.metric("Low Attendance", low_count)

        st.markdown("<br>", unsafe_allow_html=True)

        attendance_filter = st.selectbox(
            "Filter by attendance",
            [
                "All Students",
                "Eligible (75% and above)",
                "Low Attendance (below 75%)",
            ],
        )

        attendance_df = df[
            ["id", "name", "department", "attendance"]
        ].copy()

        attendance_df["Eligibility"] = attendance_df[
            "attendance"
        ].apply(attendance_status)

        if attendance_filter == "Eligible (75% and above)":
            attendance_df = attendance_df[
                attendance_df["attendance"] >= 75
            ]

        elif attendance_filter == "Low Attendance (below 75%)":
            attendance_df = attendance_df[
                attendance_df["attendance"] < 75
            ]

        attendance_df = attendance_df.rename(
            columns={
                "id": "Student ID",
                "name": "Student Name",
                "department": "Department",
                "attendance": "Attendance (%)",
            }
        )

        st.dataframe(
            attendance_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Attendance (%)": st.column_config.ProgressColumn(
                    "Attendance (%)",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "Eligibility": st.column_config.TextColumn(
                    "Eligibility",
                ),
            },
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📌 Attendance Policy Indicator</div>
                <div class="panel-subtitle">
                    This application uses 75% as the attendance threshold.
                    Verify your institution's official rules before making
                    academic eligibility decisions.
                </div>
                <span class="status-good">Eligible: 75% or above</span>
                &nbsp;
                <span class="status-danger">Low: Below 75%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=df["attendance"],
                nbinsx=10,
                marker_color="#0D9488",
                opacity=0.9,
                name="Students",
            )
        )

        fig.add_vline(
            x=75,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="75% Threshold",
        )

        fig.update_layout(
            xaxis_title="Attendance (%)",
            yaxis_title="Number of Students",
            bargap=0.12,
        )

        fig = plot_layout(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No student attendance records are available yet.")


# ============================================================
# AI / MACHINE LEARNING PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    section_heading(
        "AI Student Performance Prediction",
        "Estimate academic performance using the existing Random Forest model.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🧠 Prediction Engine</div>
            <div class="panel-subtitle">
                Enter attendance and marks to estimate the student's
                performance category.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 0.8])

    with left:

        with st.form("prediction_form"):

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
            )

            assignment = st.number_input(
                "Assignment Marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
            )

            previous = st.number_input(
                "Previous Marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
            )

            predict_clicked = st.form_submit_button(
                "🚀 Predict Performance",
                use_container_width=True,
            )

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📊 Model Information</div>
                <div class="panel-subtitle">
                    Current machine learning configuration
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Held-out Test Accuracy",
            f"{model_accuracy:.1f}%",
        )

        st.write("**Algorithm:** Random Forest Classifier")
        st.write("**Estimators:** 100")
        st.write("**Input Features:** 3")
        st.write("**Output Categories:** 3")

        st.caption(
            "Accuracy is calculated using the existing sample dataset "
            "and held-out test split."
        )

    if predict_clicked:

        prediction, confidence = performance_prediction(
            attendance,
            assignment,
            previous,
        )

        st.markdown("---")

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">✨ Prediction Results</div>
                <div class="panel-subtitle">
                    The estimated performance category for the supplied inputs
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_col, confidence_col = st.columns(2)

        with result_col:

            if prediction == "Good":
                status_class = "status-good"
                status_message = "Positive performance category"

            elif prediction == "Average":
                status_class = "status-warning"
                status_message = "Moderate performance category"

            else:
                status_class = "status-danger"
                status_message = "Additional academic support may help"

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted Category</div>
                    <div class="prediction-value">{prediction}</div>
                    <br>
                    <span class="{status_class}">{prediction}</span>
                    <p style="color:#627D98;margin-top:18px;">
                        {status_message}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with confidence_col:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Model Class Probability
                    </div>
                    <div class="prediction-value">{confidence:.1f}%</div>
                    <p style="color:#627D98;margin-top:12px;">
                        Highest probability among the model's output classes.
                        This is not a guarantee of prediction correctness.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        probabilities = model.predict_proba(
            [[attendance, assignment, previous]]
        )[0]

        probability_df = pd.DataFrame(
            {
                "Performance Category": model.classes_,
                "Model Probability (%)": probabilities * 100,
            }
        ).sort_values(
            "Model Probability (%)",
            ascending=True,
        )

        fig = px.bar(
            probability_df,
            x="Model Probability (%)",
            y="Performance Category",
            orientation="h",
            text=probability_df["Model Probability (%)"].map(
                lambda value: f"{value:.1f}%"
            ),
            color="Performance Category",
            color_discrete_map={
                "Good": "#0D9488",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
        )

        fig.update_layout(
            showlegend=False,
            xaxis=dict(range=[0, 100], title="Model Probability (%)"),
            yaxis_title="",
        )

        fig = plot_layout(fig, 280)
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "Educational estimate only. The model uses a small, predefined "
            "sample dataset, not a validated model trained on the complete "
            "student database. Its predictions and probabilities should not "
            "be treated as verified academic outcomes."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <p><strong>AI Smart College Management System</strong></p>
        <p>Developed for AI/ML Internship Project</p>
        <p>Python · Streamlit · SQLite · Scikit-learn · Plotly</p>
    </div>
    """,
    unsafe_allow_html=True,
)
