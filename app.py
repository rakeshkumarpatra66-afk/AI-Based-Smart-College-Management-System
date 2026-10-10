import os
import sqlite3

from flask import Flask, request, redirect, url_for, flash

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)
app.secret_key = "smart_college_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "students.db")


# ==========================================
# DATABASE
# ==========================================

def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            department TEXT NOT NULL,
            attendance REAL NOT NULL,
            assignment_marks REAL NOT NULL,
            previous_marks REAL NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ==========================================
# MACHINE LEARNING TRAINING DATA
# ==========================================

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
    "Good", "Good", "Average", "Average", "Average",
    "Needs Improvement", "Needs Improvement", "Good",
    "Good", "Good", "Good", "Good", "Good",
    "Average", "Average", "Needs Improvement",
    "Needs Improvement"
]

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

model_accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
) * 100


# ==========================================
# CSS AND HTML IN ONE PYTHON FILE
# ==========================================

def page(title, content):

    return """
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>""" + title + """ | Smart College</title>

        <style>

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f1f5f9;
                color: #1e293b;
            }

            header {
                background: #173b68;
                color: white;
                padding: 25px;
                text-align: center;
            }

            header h1 {
                margin: 0 0 20px;
            }

            nav {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 10px;
            }

            nav a {
                color: white;
                text-decoration: none;
                padding: 11px 15px;
                border-radius: 6px;
            }

            nav a:hover {
                background: #315b8c;
            }

            main {
                max-width: 1200px;
                width: 94%;
                margin: 30px auto;
                min-height: 65vh;
            }

            .cards {
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }

            .card {
                background: white;
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 3px 12px #00000012;
            }

            .card h3 {
                color: #475569;
            }

            .card p {
                font-size: 30px;
                font-weight: bold;
                color: #173b68;
            }

            form {
                max-width: 650px;
                margin: 20px auto;
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 3px 12px #00000012;
            }

            label {
                display: block;
                margin: 15px 0 7px;
                font-weight: bold;
            }

            input, select {
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-size: 15px;
            }

            button, .button {
                display: inline-block;
                background: #176b45;
                color: white;
                padding: 12px 18px;
                margin-top: 18px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                text-decoration: none;
            }

            .delete {
                background: #c62828;
                margin-top: 0;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin: 20px 0;
            }

            th, td {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }

            th {
                background: #173b68;
                color: white;
            }

            tr:nth-child(even) {
                background: #f8fafc;
            }

            .alert {
                padding: 14px;
                margin-bottom: 15px;
                border-radius: 6px;
                background: #d8f3dc;
                color: #155724;
            }

            .error {
                background: #ffe0e0;
                color: #8a1111;
            }

            .result {
                background: white;
                padding: 25px;
                border-left: 5px solid #176b45;
                border-radius: 8px;
                margin-top: 20px;
            }

            footer {
                background: #173b68;
                color: white;
                text-align: center;
                padding: 20px;
            }

            @media(max-width: 650px) {
                table {
                    font-size: 12px;
                }

                th, td {
                    padding: 7px;
                }

                nav {
                    flex-direction: column;
                }
            }

        </style>
    </head>

    <body>

        <header>
            <h1>🎓 Smart College Management System</h1>

            <nav>
                <a href="/">Dashboard</a>
                <a href="/register">Register Student</a>
                <a href="/students">Students</a>
                <a href="/attendance">Attendance</a>
                <a href="/predict">AI Prediction</a>
            </nav>
        </header>

        <main>
            """ + content + """
        </main>

        <footer>
            <p>AI-Based Smart College Management System</p>
            <p>Python | Flask | SQLite | Machine Learning</p>
        </footer>

    </body>
    </html>
    """


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    connection = get_db()

    total = connection.execute(
        "SELECT COUNT(*) FROM students"
    ).fetchone()[0]

    attendance = connection.execute(
        "SELECT AVG(attendance) FROM students"
    ).fetchone()[0] or 0

    marks = connection.execute(
        """
        SELECT AVG((assignment_marks + previous_marks) / 2.0)
        FROM students
        """
    ).fetchone()[0] or 0

    recent = connection.execute(
        "SELECT * FROM students ORDER BY id DESC LIMIT 5"
    ).fetchall()

    connection.close()

    rows = ""

    for student in recent:
        rows += f"""
        <tr>
            <td>{student['id']}</td>
            <td>{student['name']}</td>
            <td>{student['email']}</td>
            <td>{student['department']}</td>
        </tr>
        """

    if not rows:
        rows = """
        <tr>
            <td colspan="4">No students registered yet.</td>
        </tr>
        """

    content = f"""
        <h2>📊 Dashboard</h2>

        <p>Welcome to your college management dashboard.</p>

        <div class="cards">

            <div class="card">
                <h3>Total Students</h3>
                <p>{total}</p>
            </div>

            <div class="card">
                <h3>Average Attendance</h3>
                <p>{attendance:.1f}%</p>
            </div>

            <div class="card">
                <h3>Average Marks</h3>
                <p>{marks:.1f}%</p>
            </div>

            <div class="card">
                <h3>ML Test Accuracy</h3>
                <p>{model_accuracy:.1f}%</p>
            </div>

        </div>

        <h2>Recently Registered Students</h2>

        <table>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Department</th>
            </tr>
            {rows}
        </table>
    """

    return page("Dashboard", content)


# ==========================================
# STUDENT REGISTRATION
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        department = request.form.get("department", "").strip()

        try:
            attendance = float(request.form.get("attendance", ""))
            assignment = float(
                request.form.get("assignment_marks", "")
            )
            previous = float(
                request.form.get("previous_marks", "")
            )

        except ValueError:
            return page(
                "Registration Error",
                '<h2 class="error">Enter valid numeric values.</h2>'
            ), 400

        if not name or not email or not department:
            return page(
                "Registration Error",
                '<h2 class="error">All fields are required.</h2>'
            ), 400

        if not all(
            0 <= value <= 100
            for value in (attendance, assignment, previous)
        ):
            return page(
                "Registration Error",
                '<h2 class="error">Values must be between 0 and 100.</h2>'
            ), 400

        connection = get_db()

        try:
            connection.execute(
                """
                INSERT INTO students
                (name, email, department, attendance,
                 assignment_marks, previous_marks)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    email,
                    department,
                    attendance,
                    assignment,
                    previous
                )
            )

            connection.commit()

        except sqlite3.IntegrityError:
            connection.close()

            return page(
                "Registration Error",
                """
                <h2 class="error">
                    This email is already registered.
                </h2>
                <a class="button" href="/register">Go Back</a>
                """
            ), 400

        connection.close()

        return redirect(url_for("students"))

    content = """
        <h2>👨‍🎓 Student Registration</h2>

        <form method="POST">

            <label>Student Name</label>
            <input type="text" name="name" required>

            <label>Email Address</label>
            <input type="email" name="email" required>

            <label>Department</label>
            <select name="department" required>
                <option value="">Select Department</option>
                <option>Computer Science</option>
                <option>Information Technology</option>
                <option>Mechanical Engineering</option>
                <option>Electrical Engineering</option>
                <option>Civil Engineering</option>
                <option>Electronics Engineering</option>
            </select>

            <label>Attendance (%)</label>
            <input type="number" name="attendance"
                   min="0" max="100" step="0.01" required>

            <label>Assignment Marks</label>
            <input type="number" name="assignment_marks"
                   min="0" max="100" step="0.01" required>

            <label>Previous Marks</label>
            <input type="number" name="previous_marks"
                   min="0" max="100" step="0.01" required>

            <button type="submit">Register Student</button>

        </form>
    """

    return page("Register Student", content)


# ==========================================
# VIEW AND DELETE STUDENTS
# ==========================================

@app.route("/students")
def students():

    connection = get_db()

    records = connection.execute(
        "SELECT * FROM students ORDER BY id DESC"
    ).fetchall()

    connection.close()

    rows = ""

    for student in records:

        rows += f"""
        <tr>
            <td>{student['id']}</td>
            <td>{student['name']}</td>
            <td>{student['email']}</td>
            <td>{student['department']}</td>
            <td>{student['attendance']}%</td>
            <td>{student['assignment_marks']}</td>
            <td>{student['previous_marks']}</td>
            <td>
                <form method="POST"
                      action="/delete/{student['id']}"
                      style="padding:0;box-shadow:none"
                      onsubmit="return confirm('Delete this student?')">
                    <button class="delete" type="submit">
                        Delete
                    </button>
                </form>
            </td>
        </tr>
        """

    if not rows:
        rows = """
        <tr>
            <td colspan="8">No students found.</td>
        </tr>
        """

    content = f"""
        <h2>📋 Student Records</h2>

        <table>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Department</th>
                <th>Attendance</th>
                <th>Assignment</th>
                <th>Previous Marks</th>
                <th>Action</th>
            </tr>

            {rows}
        </table>
    """

    return page("Students", content)


@app.route("/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id):

    connection = get_db()

    connection.execute(
        "DELETE FROM students WHERE id = ?",
        (student_id,)
    )

    connection.commit()
    connection.close()

    return redirect(url_for("students"))


# ==========================================
# ATTENDANCE
# ==========================================

@app.route("/attendance")
def attendance_page():

    connection = get_db()

    records = connection.execute(
        "SELECT * FROM students ORDER BY name"
    ).fetchall()

    connection.close()

    rows = ""

    for student in records:

        status = (
            "Eligible by percentage"
            if student["attendance"] >= 75
            else "Low Attendance"
        )

        rows += f"""
        <tr>
            <td>{student['id']}</td>
            <td>{student['name']}</td>
            <td>{student['department']}</td>
            <td>{student['attendance']}%</td>
            <td>{status}</td>
        </tr>
        """

    if not rows:
        rows = """
        <tr>
            <td colspan="5">No students registered yet.</td>
        </tr>
        """

    content = f"""
        <h2>📅 Attendance Management</h2>

        <p>
            Students with attendance below 75% are marked
            as having low attendance.
        </p>

        <table>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Department</th>
                <th>Attendance</th>
                <th>Status</th>
            </tr>

            {rows}
        </table>
    """

    return page("Attendance", content)


# ==========================================
# MACHINE LEARNING PREDICTION
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = ""

    if request.method == "POST":

        try:
            attendance = float(request.form.get("attendance", ""))
            assignment = float(
                request.form.get("assignment_marks", "")
            )
            previous = float(
                request.form.get("previous_marks", "")
            )

            if not all(
                0 <= value <= 100
                for value in (attendance, assignment, previous)
            ):
                raise ValueError

            prediction = model.predict([[
                attendance,
                assignment,
                previous
            ]])[0]

            result = f"""
                <div class="result">
                    <h3>Prediction Result</h3>
                    <h2>{prediction}</h2>
                    <p>
                        This is a model estimate based on the
                        sample training data, not a guaranteed
                        academic outcome.
                    </p>
                </div>
            """

        except ValueError:
            result = """
                <div class="alert error">
                    Please enter valid numbers between 0 and 100.
                </div>
            """

    content = f"""
        <h2>🤖 AI Student Performance Prediction</h2>

        <form method="POST">

            <label>Attendance (%)</label>
            <input type="number" name="attendance"
                   min="0" max="100" required>

            <label>Assignment Marks</label>
            <input type="number" name="assignment_marks"
                   min="0" max="100" required>

            <label>Previous Marks</label>
            <input type="number" name="previous_marks"
                   min="0" max="100" required>

            <button type="submit">Predict Performance</button>

        </form>

        {result}
    """

    return page("AI Prediction", content)


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":

    init_db()

    print("Smart College Management System Started!")
    print("Open http://127.0.0.1:5000 in your browser.")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )