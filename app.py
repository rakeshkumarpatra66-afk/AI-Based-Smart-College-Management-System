```python
# ============================================================
# AI SMART COLLEGE MANAGEMENT SYSTEM
# Python + Streamlit + SQLite + Machine Learning
# ============================================================

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


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"


# ============================================================
# PROFESSIONAL DARK DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --background: #0B1120;
        --surface: #151F32;
        --surface2: #1C2A40;
        --primary: #14B8A6;
        --text: #F1F5F9;
        --muted: #A7B5C8;
        --border: #2A3A52;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101B30, #0B1120);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] p {
        color: var(--muted);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 10px;
        border-radius: 10px;
        transition: 0.2s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #1C2A40;
    }

    .brand-box {
        background: linear-gradient(135deg, #17263E, #0F766E);
        border: 1px solid #2B5964;
        border-radius: 18px;
        padding: 22px 14px;
        margin-bottom: 24px;
        text-align: center;
    }

    .brand-icon {
        font-size: 42px;
    }

    .brand-title {
        color: #FFFFFF;
        font-size: 19px;
        font-weight: 800;
    }

    .brand-caption {
        color: #99F6E4;
        font-size: 11px;
        letter-spacing: 1.5px;
    }

    .top-header {
        background: linear-gradient(120deg, #17263E, #0F766E);
        padding: 30px;
        border: 1px solid #2B5964;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
    }

    .top-header h1 {
        color: #FFFFFF !important;
        font-size: clamp(23px, 3vw, 35px);
        font-weight: 800;
        margin-bottom: 10px;
    }

    .top-header p {
        color: #D1FAE5 !important;
        font-size: 14px;
    }

    .section-heading {
        color: #F8FAFC;
        font-size: 25px;
        font-weight: 800;
        margin: 15px 0 8px;
    }

    .section-description {
        color: #A7B5C8;
        font-size: 13px;
        margin-bottom: 22px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #17263E, #111C2F);
        border: 1px solid #2A3A52;
        border-top: 4px solid #14B8A6;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {
        color: #B7C5D8 !important;
        font-size: 13px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 800;
    }

    .panel {
        background: #151F32;
        border: 1px solid #2A3A52;
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0 18px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }

    .panel-title {
        color: #F8FAFC;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .panel-subtitle {
        color: #A7B5C8;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .stApp p {
        color: #D5DEEA;
    }

    .stApp label {
        color: #E2E8F0 !important;
    }

    div.stButton > button,
    div.stFormSubmitButton > button,
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0D9488, #14B8A6);
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        padding: 11px 18px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #0F766E;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: #1C2A40 !important;
        border: 1px solid #40516A !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background: transparent !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }

    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background: #17263E !important;
    }

    li[role="option"] {
        background: #17263E !important;
        color: #FFFFFF !important;
    }

    li[role="option"]:hover {
        background: #0F766E !important;
    }

    [data-testid="stForm"] {
        background: #151F32;
        border: 1px solid #2A3A52;
        border-radius: 18px;
        padding: 24px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2A3A52;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .prediction-card {
        background: linear-gradient(145deg, #17263E, #111C2F);
        border: 1px solid #2A3A52;
        border-radius: 18px;
        padding: 25px;
        margin: 15px 0;
    }

    .prediction-label {
        color: #A7B5C8;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .prediction-value {
        color: #5EEAD4;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .status-good {
        display: inline-block;
        padding: 6px 12px;
        background: #064E3B;
        color: #A7F3D0;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-warning {
        display: inline-block;
        padding: 6px 12px;
        background: #78350F;
        color: #FDE68A;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-danger {
        display: inline-block;
        padding: 6px 12px;
        background: #7F1D1D;
        color: #FECACA;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .footer {
        background: #151F32;
        border: 1px solid #2A3A52;
        border-radius: 16px;
        padding: 22px;
        margin-top: 40px;
        text-align: center;
    }

    .footer p {
        color: #A7B5C8;
        font-size: 12px;
    }

    .footer strong {
        color: #5EEAD4;
    }

    @media (max-width: 768px) {
        .top-header {
            padding: 20px;
        }

        .panel {
            padding: 15px;
        }

        div[data-testid="stMetric"] {
            padding: 12px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    connection = sqlite3.connect(
        str(DB_PATH),
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


def get_students():
    with get_db() as connection:
        return connection.execute(
            "SELECT * FROM students ORDER BY id DESC"
        ).fetchall()


def students_dataframe():
    records = get_students()

    columns = [
        "id",
        "name",
        "email",
        "department",
        "attendance",
        "assignment_marks",
        "previous_marks",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame([dict(row) for row in records])


init_db()


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

X = [
    [95, 90, 92], [90, 85, 88], [85, 88, 85],
    [80, 80, 82], [75, 75, 78], [70, 72, 74],
    [65, 68, 70], [60, 65, 63], [55, 55, 58],
    [50, 45, 48], [92, 95, 96], [88, 89, 90],
    [82, 84, 86], [78, 79, 80], [72, 73, 75],
    [68, 69, 68], [62, 61, 65], [58, 58, 55],
    [45, 50, 45], [40, 35, 38], [98, 98, 97],
    [93, 92, 94], [87, 86, 89], [81, 83, 84],
    [76, 77, 79], [71, 70, 72], [66, 64, 66],
    [59, 57, 60], [52, 49, 50], [48, 40, 42],
]

y = [
    "Good", "Good", "Good", "Good", "Good", "Good",
    "Average", "Average", "Average", "Needs Improvement",
    "Good", "Good", "Good", "Good", "Good", "Average",
    "Average", "Average", "Needs Improvement",
    "Needs Improvement", "Good", "Good", "Good", "Good",
    "Good", "Good", "Average", "Average",
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
        paper_bgcolor="#151F32",
        plot_bgcolor="#151F32",
        font=dict(
            family="Inter, sans-serif",
            color="#D5DEEA",
            size=12,
        ),
        margin=dict(l=15, r=15, t=40, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        gridcolor="#2A3A52",
        zerolinecolor="#2A3A52",
    )

    fig.update_yaxes(
        gridcolor="#2A3A52",
        zerolinecolor="#2A3A52",
    )

    return fig


def attendance_status(value):
    return "Eligible" if value >= 75 else "Low Attendance"


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
        float(df["attendance"].mean()) if total_students else 0.0
    )

    avg_marks = (
        float(
            (df["assignment_marks"] + df["previous_marks"]).mean() / 2
        )
        if total_students else 0.0
    )

    eligible_students = (
        int((df["attendance"] >= 75).sum()) if total_students else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👨‍🎓 Total Students", total_students)

    with col2:
        st.metric("📅 Avg. Attendance", f"{avg_attendance:.1f}%")

    with col3:
        st.metric("📚 Average Marks", f"{avg_marks:.1f}%")

    with col4:
        st.metric("🤖 ML Test Accuracy", f"{model_accuracy:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    if total_students:

        left, right = st.columns(2)

        with left:

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
                hole=0.60,
                color_discrete_sequence=[
                    "#14B8A6",
                    "#38BDF8",
                    "#818CF8",
                    "#F59E0B",
                    "#F472B6",
                    "#A78BFA",
                ],
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True,
            )

        with right:

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
                    "Eligible (75%+)": "#14B8A6",
                    "Low Attendance": "#F59E0B",
                },
            )

            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Number of Students",
                yaxis=dict(rangemode="tozero"),
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True,
            )

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
            hover_data=["email", "attendance"],
            labels={
                "assignment_marks": "Assignment Marks",
                "previous_marks": "Previous Marks",
                "department": "Department",
                "attendance": "Attendance (%)",
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        fig.update_layout(
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
        )

        st.plotly_chart(
            plot_layout(fig, 420),
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🆕 Recently Registered Students</div>
                <div class="panel-subtitle">
                    Latest records saved in the SQLite database
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
            "No student records found. Open Register Student "
            "to add your first student."
        )

        if st.button("➕ Register Your First Student"):
            st.session_state["navigate_register"] = True
            st.rerun()

    st.caption(
        f"ML test accuracy: {model_accuracy:.1f}%. "
        "This is based on the predefined sample dataset and is not "
        "a guarantee of real-world prediction accuracy."
    )


# ============================================================
# STUDENT REGISTRATION
# ============================================================

elif page == "👨‍🎓 Register Student":

    section_heading(
        "Student Registration",
        "Add student information and academic records.",
    )

    departments = [
        "Computer Science",
        "Information Technology",
        "Mechanical Engineering",
        "Electrical Engineering",
        "Civil Engineering",
        "Electronics Engineering",
    ]

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">👤 Personal Information</div>
            <div class="panel-subtitle">
                Enter the student's name and contact details.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("student_registration", clear_on_submit=True):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            name = st.text_input(
                "Full Name *",
                placeholder="e.g. Rahul Sharma",
                max_chars=100,
            )

        with col2:
            email = st.text_input(
                "Email Address *",
                placeholder="e.g. rahul@gmail.com",
                max_chars=254,
            )

        department = st.selectbox(
            "Academic Department *",
            departments,
            index=None,
            placeholder="Choose a department",
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📚 Academic Information</div>
                <div class="panel-subtitle">
                    Enter attendance and academic marks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col3, col4, col5 = st.columns(3, gap="medium")

        with col3:
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
            )

        with col4:
            assignment = st.number_input(
                "Assignment Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
            )

        with col5:
            previous = st.number_input(
                "Previous Marks",
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

        elif not any(character.isalpha() for character in name):
            st.error("Enter a valid student name.")

        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            st.error("Enter a valid email address.")

        else:

            try:

                with get_db() as connection:
                    connection.execute(
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
                            assignment,
                            previous,
                        ),
                    )

                st.success(
                    f"Student {name} registered successfully!"
                )

                st.rerun()

            except sqlite3.IntegrityError:
                st.error(
                    "This email already exists. Use a different email."
                )

            except sqlite3.Error as exc:
                st.error(f"Database error: {exc}")


# ============================================================
# STUDENT RECORDS
# ============================================================

elif page == "📋 Students":

    section_heading(
        "Student Records",
        "Search, review, export and manage student records.",
    )

    if not df.empty:

        search = st.text_input(
            "🔍 Search Students",
            placeholder="Enter name, email or department",
        )

        filtered_df = df.copy()

        if search.strip():

            search_text = search.strip()

            mask = (
                filtered_df["name"].astype(str).str.contains(
                    search_text, case=False, na=False, regex=False
                )
                |
                filtered_df["email"].astype(str).str.contains(
                    search_text, case=False, na=False, regex=False
                )
                |
                filtered_df["department"].astype(str).str.contains(
                    search_text, case=False, na=False, regex=False
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

        st.write(
            f"Showing {len(filtered_df)} of {len(df)} students"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Export Students to CSV",
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
                    Permanently delete a selected student record.
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
            "Select Student",
            list(options.keys()),
        )

        confirm_delete = st.checkbox(
            "I confirm permanent deletion of this student."
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
        "Monitor attendance and eligibility status.",
    )

    if not df.empty:

        eligible_count = int((df["attendance"] >= 75).sum())
        low_count = len(df) - eligible_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", len(df))
        col2.metric("Eligible Students", eligible_count)
        col3.metric("Low Attendance", low_count)

        attendance_filter = st.selectbox(
            "Filter Attendance",
            [
                "All Students",
                "Eligible (75% and above)",
                "Low Attendance (below 75%)",
            ],
        )

        attendance_df = df[
            ["id", "name", "department", "attendance"]
        ].copy()

        attendance_df["Eligibility"] = (
            attendance_df["attendance"].apply(attendance_status)
        )

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
            },
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📌 Attendance Policy</div>
                <div class="panel-subtitle">
                    This application uses 75% as the threshold.
                    Confirm the official college policy before
                    making eligibility decisions.
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
                marker_color="#14B8A6",
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

        st.plotly_chart(
            plot_layout(fig, 350),
            use_container_width=True,
        )

    else:
        st.info("No attendance records are available yet.")


# ============================================================
# AI / MACHINE LEARNING PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    section_heading(
        "AI Student Performance Prediction",
        "Estimate academic performance using a Random Forest classifier.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🧠 Prediction Engine</div>
            <div class="panel-subtitle">
                Enter attendance and marks to estimate a performance category.
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
                    Machine learning configuration
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
                    Estimated performance category for the supplied inputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_col, confidence_col = st.columns(2)

        if prediction == "Good":
            status_class = "status-good"
            status_message = "Positive performance category"

        elif prediction == "Average":
            status_class = "status-warning"
            status_message = "Moderate performance category"

        else:
            status_class = "status-danger"
            status_message = "Additional academic support may help"

        with result_col:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted Category</div>
                    <div class="prediction-value">{prediction}</div>
                    <br>
                    <span class="{status_class}">{prediction}</span>
                    <p>{status_message}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with confidence_col:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Highest Class Probability
                    </div>
                    <div class="prediction-value">{confidence:.1f}%</div>
                    <p>
                        This represents the model's highest class probability,
                        not a guarantee that the prediction is correct.
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
                "Good": "#14B8A6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
        )

        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                range=[0, 100],
                title="Model Probability (%)",
            ),
            yaxis_title="",
        )

        st.plotly_chart(
            plot_layout(fig, 280),
            use_container_width=True,
        )

        st.warning(
            "Educational estimate only. This model uses a small, predefined "
            "sample dataset. It has not been validated on the complete "
            "student database and should not be treated as a verified "
            "academic outcome."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <p><strong>AI Smart College Management System</strong></p>
        <p>AI/ML Internship Project</p>
        <p>Python · Streamlit · SQLite · Scikit-learn · Plotly</p>
    </div>
    """,
    unsafe_allow_html=True,
)
```
