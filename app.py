import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIGURATION / DESIGN
# =====================================================
st.set_page_config(
    page_title="Smart College Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"

# Keep the same navy, light background, green buttons, and college theme.
st.markdown(
    """
    <style>
    .stApp { background: #f1f5f9; color: #1e293b; }
    [data-testid="stHeader"] { background: #173b68; }
    .hero {
        background: #173b68; color: white; padding: 22px 26px;
        border-radius: 12px; margin-bottom: 18px; text-align: center;
    }
    .hero h1 { color: white; margin: 0; }
    .hero p { color: white; margin: 8px 0 0; }
    div[data-testid="stMetric"] {
        background: white; padding: 22px; border-radius: 12px;
        box-shadow: 0 3px 12px #00000012; border-top: 4px solid #173b68;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background: #176b45; color: white; border: 0;
        border-radius: 6px; padding: 0.55rem 1rem;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #125637; color: white; border: 0;
    }
    .section-title {
        color: #173b68; font-weight: 700; padding-top: 6px;
    }
    .footer {
        background: #173b68; color: white; text-align: center;
        padding: 16px; border-radius: 8px; margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# DATABASE
# =====================================================
def get_db():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
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


# =====================================================
# MACHINE LEARNING MODEL (same sample training data)
# =====================================================
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
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
model_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown(
    """
    <div class="hero">
      <h2>🎓 Smart College</h2>
      <p>Management System</p>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "👨‍🎓 Register Student",
        "📋 Students",
        "📅 Attendance",
        "🤖 AI Prediction",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Python | Streamlit | SQLite | Machine Learning")

st.markdown(
    """
    <div class="hero">
      <h1>🎓 Smart College Management System</h1>
      <p>AI-Based Smart College Management System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# DASHBOARD
# =====================================================
if page == "📊 Dashboard":
    records = get_students()
    total = len(records)
    avg_attendance = (
        sum(float(row["attendance"]) for row in records) / total if total else 0
    )
    avg_marks = (
        sum(
            (float(row["assignment_marks"]) + float(row["previous_marks"])) / 2
            for row in records
        ) / total
        if total else 0
    )

    st.subheader("📊 Dashboard")
    st.write("Welcome to your college management dashboard.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total)
    c2.metric("Average Attendance", f"{avg_attendance:.1f}%")
    c3.metric("Average Marks", f"{avg_marks:.1f}%")
    c4.metric("ML Test Accuracy", f"{model_accuracy:.1f}%")

    st.markdown('<h3 class="section-title">Recently Registered Students</h3>',
                unsafe_allow_html=True)
    if records:
        recent_df = pd.DataFrame([dict(row) for row in records[:5]])
        recent_df = recent_df[["id", "name", "email", "department"]]
        recent_df.columns = ["ID", "Name", "Email", "Department"]
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No students registered yet.")

# =====================================================
# STUDENT REGISTRATION
# =====================================================
elif page == "👨‍🎓 Register Student":
    st.subheader("👨‍🎓 Student Registration")
    with st.form("student_registration", clear_on_submit=True):
        name = st.text_input("Student Name")
        email = st.text_input("Email Address")
        department = st.selectbox(
            "Department",
            [
                "Computer Science",
                "Information Technology",
                "Mechanical Engineering",
                "Electrical Engineering",
                "Civil Engineering",
                "Electronics Engineering",
            ],
            index=None,
            placeholder="Select Department",
        )
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5
        )
        assignment = st.number_input(
            "Assignment Marks", min_value=0.0, max_value=100.0, value=0.0, step=1.0
        )
        previous = st.number_input(
            "Previous Marks", min_value=0.0, max_value=100.0, value=0.0, step=1.0
        )
        submitted = st.form_submit_button("Register Student")

    if submitted:
        name, email = name.strip(), email.strip()
        if not name or not email or not department:
            st.error("All fields are required.")
        elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            st.error("Please enter a valid email address.")
        else:
            try:
                with get_db() as connection:
                    connection.execute(
                        """
                        INSERT INTO students
                        (name, email, department, attendance,
                         assignment_marks, previous_marks)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (name, email, department, attendance, assignment, previous),
                    )
                st.success("Student registered successfully!")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("This email is already registered.")

# =====================================================
# STUDENT RECORDS / DELETE
# =====================================================
elif page == "📋 Students":
    st.subheader("📋 Student Records")
    records = get_students()
    if records:
        df = pd.DataFrame([dict(row) for row in records])
        display_df = df.rename(
            columns={
                "id": "ID",
                "name": "Name",
                "email": "Email",
                "department": "Department",
                "attendance": "Attendance (%)",
                "assignment_marks": "Assignment Marks",
                "previous_marks": "Previous Marks",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### Delete Student")
        options = {
            f"{row['id']} — {row['name']} ({row['email']})": row["id"]
            for row in records
        }
        selected = st.selectbox("Select student to delete", list(options.keys()))
        if st.button("🗑️ Delete Selected Student"):
            with get_db() as connection:
                connection.execute(
                    "DELETE FROM students WHERE id = ?", (options[selected],)
                )
            st.success("Student deleted successfully.")
            st.rerun()
    else:
        st.info("No students found.")

# =====================================================
# ATTENDANCE
# =====================================================
elif page == "📅 Attendance":
    st.subheader("📅 Attendance Management")
    st.write(
        "Students with attendance below 75% are marked as having low attendance."
    )
    records = get_students()
    if records:
        attendance_df = pd.DataFrame(
            [
                {
                    "ID": row["id"],
                    "Name": row["name"],
                    "Department": row["department"],
                    "Attendance": f"{float(row['attendance']):.1f}%",
                    "Status": (
                        "Eligible by percentage"
                        if float(row["attendance"]) >= 75
                        else "Low Attendance"
                    ),
                }
                for row in records
            ]
        )
        st.dataframe(attendance_df, use_container_width=True, hide_index=True)
    else:
        st.info("No students registered yet.")

# =====================================================
# MACHINE LEARNING PREDICTION
# =====================================================
elif page == "🤖 AI Prediction":
    st.subheader("🤖 AI Student Performance Prediction")
    with st.form("prediction_form"):
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0
        )
        assignment = st.number_input(
            "Assignment Marks", min_value=0.0, max_value=100.0, value=75.0, step=1.0
        )
        previous = st.number_input(
            "Previous Marks", min_value=0.0, max_value=100.0, value=75.0, step=1.0
        )
        predict_clicked = st.form_submit_button("Predict Performance")

    if predict_clicked:
        prediction = model.predict([[attendance, assignment, previous]])[0]
        st.markdown("### Prediction Result")
        if prediction == "Good":
            st.success(f"**{prediction}**")
        elif prediction == "Average":
            st.warning(f"**{prediction}**")
        else:
            st.error(f"**{prediction}**")
        st.info(
            "This is a model estimate based on the sample training data, "
            "not a guaranteed academic outcome."
        )

st.markdown(
    """
    <div class="footer">
      <p>AI-Based Smart College Management System</p>
      <p>Python | Streamlit | SQLite | Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)
