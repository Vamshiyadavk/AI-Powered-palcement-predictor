import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import jwt
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import check_password_hash, generate_password_hash

# Optional Google GenAI SDK for live industry tech stack analysis
try:
    from google import genai
except ImportError:
    genai = None

# -----------------------------------------------------------------------------
# Project Path Configuration & Environment Setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
ARFF_PATH = DATA_DIR / "placement_data.arff"
INSTANCE_DIR = BASE_DIR / "instance"

# Initialize Flask with explicit static and instance paths so the database
# and HTML assets remain consistent regardless of working directory.
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="",
    instance_path=str(INSTANCE_DIR),
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "sqlite:///placement.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET"] = os.getenv(
    "JWT_SECRET", "placement-predictor-jwt-secret-key-32-chars-min"
)

db = SQLAlchemy(app)

# Global holder for our trained Random Forest Classifier
model = None


# -----------------------------------------------------------------------------
# Database Models
# -----------------------------------------------------------------------------
class User(db.Model):
    """User account model for authentication (students and recruiters)."""
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(40), nullable=False, default="STUDENT")


class Student(db.Model):
    """Academic record evaluated by placement algorithms."""
    __tablename__ = "students"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    cgpa = db.Column(db.Float, nullable=False)
    internships = db.Column(db.Integer, nullable=False)
    projects = db.Column(db.Integer, nullable=False)
    backlogs = db.Column(db.Integer, nullable=False)


class Skill(db.Model):
    """Individual technical skill acquired by a student."""
    __tablename__ = "skills"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    skill_name = db.Column(db.String(120), nullable=False)
    proficiency = db.Column(db.String(40), nullable=False, default="Intermediate")


class RequiredSkill(db.Model):
    """Industry benchmark skills mapped by company category."""
    __tablename__ = "required_skills"
    id = db.Column(db.Integer, primary_key=True)
    company_tier = db.Column(db.String(80), nullable=False, index=True)
    skill_name = db.Column(db.String(120), nullable=False)


class Course(db.Model):
    """Recommended learning resources for bridging skill deficiencies."""
    __tablename__ = "courses"
    id = db.Column(db.Integer, primary_key=True)
    skill_name = db.Column(db.String(120), nullable=False, index=True)
    course_name = db.Column(db.String(255), nullable=False)
    platform = db.Column(db.String(120), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.String(80), nullable=False)

    def as_dict(self):
        return {
            "id": self.id,
            "skillName": self.skill_name,
            "courseName": self.course_name,
            "platform": self.platform,
            "url": self.url,
            "duration": self.duration,
        }


# -----------------------------------------------------------------------------
# Machine Learning: Training the Placement Classifier
# -----------------------------------------------------------------------------
def train_model():
    """
    Parses the ARFF recruitment dataset and trains a 100-tree Random Forest.
    Features: [cgpa, internships, projects, backlogs, skills_count] -> placed ('yes'/'no')
    """
    global model
    rows = []
    if not ARFF_PATH.exists():
        app.logger.warning(f"Training dataset not found at {ARFF_PATH}")
        return

    # Parse ARFF data records, ignoring header and comment annotations
    for line in ARFF_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("@") or line.startswith("%"):
            continue
        values = [val.strip() for val in line.split(",")]
        if len(values) == 6:
            feature_vector = [float(val) for val in values[:5]]
            target_class = values[5].lower()
            rows.append((feature_vector, target_class))

    if rows:
        features, labels = zip(*rows)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features, labels)
        model = clf


# -----------------------------------------------------------------------------
# Database Seed Data: Company Tiers & Course Catalog
# -----------------------------------------------------------------------------
def seed_data():
    """Populates baseline company skills and course catalog on first launch."""
    if RequiredSkill.query.first() is not None:
        return

    # Standard industry benchmarks across hiring categories
    tier_skills = {
        "PRODUCT": [
            "Data Structures & Algorithms",
            "System Design",
            "Python",
            "SQL",
            "Operating Systems",
            "Computer Networks",
        ],
        "SERVICE": [
            "Java",
            "Python",
            "SQL",
            "Web Development",
            "Software Engineering",
            "Communication Skills",
        ],
        "STARTUP": [
            "React.js",
            "Node.js",
            "Python",
            "Docker",
            "Git",
            "REST APIs",
        ],
    }
    for tier, skills in tier_skills.items():
        for skill_name in skills:
            db.session.add(RequiredSkill(company_tier=tier, skill_name=skill_name))

    # Curated online courses with verifiable platforms and durations
    courses_data = [
        ("Data Structures & Algorithms", "Mastering Data Structures & Algorithms", "Udemy", "https://www.udemy.com/course/datastructurescncpp/", "58 hours"),
        ("Data Structures & Algorithms", "Algorithms Specialization", "Coursera", "https://www.coursera.org/specializations/algorithms", "4 months"),
        ("System Design", "Grokking System Design Fundamentals", "Educative", "https://www.educative.io/courses/grokking-system-design-fundamentals", "20 hours"),
        ("Python", "Python for Everybody", "Coursera", "https://www.coursera.org/specializations/python", "3 months"),
        ("Python", "Complete Python Bootcamp", "Udemy", "https://www.udemy.com/course/complete-python-bootcamp/", "22 hours"),
        ("SQL", "The Complete SQL Bootcamp", "Udemy", "https://www.udemy.com/course/the-complete-sql-bootcamp/", "9 hours"),
        ("SQL", "SQL for Data Science", "Coursera", "https://www.coursera.org/learn/sql-for-data-science", "14 hours"),
        ("Operating Systems", "Operating Systems and You: Becoming a Power User", "Coursera", "https://www.coursera.org/learn/os-power-user", "30 hours"),
        ("Computer Networks", "The Bits and Bytes of Computer Networking", "Coursera", "https://www.coursera.org/learn/computer-networking", "25 hours"),
        ("Java", "Java Programming Masterclass", "Udemy", "https://www.udemy.com/course/java-the-complete-java-developer-course/", "80 hours"),
        ("Web Development", "The Complete 2026 Web Development Bootcamp", "Udemy", "https://www.udemy.com/course/the-complete-web-development-bootcamp/", "65 hours"),
        ("Software Engineering", "Software Development Lifecycle Specialization", "Coursera", "https://www.coursera.org/specializations/software-development-lifecycle", "4 months"),
        ("Communication Skills", "Effective Communication Specialization", "Coursera", "https://www.coursera.org/specializations/effective-communication", "5 months"),
        ("React.js", "React - The Complete Guide", "Udemy", "https://www.udemy.com/course/react-the-complete-guide-incl-redux/", "48 hours"),
        ("Node.js", "The Complete Node.js Developer Course", "Udemy", "https://www.udemy.com/course/the-complete-nodejs-developer-course-2/", "35 hours"),
        ("Docker", "Docker & Kubernetes: The Practical Guide", "Udemy", "https://www.udemy.com/course/docker-kubernetes-the-practical-guide/", "23 hours"),
        ("Git", "Version Control with Git", "Coursera", "https://www.coursera.org/learn/version-control-with-git", "13 hours"),
        ("REST APIs", "REST APIs with Flask and Python", "Udemy", "https://www.udemy.com/course/rest-api-flask-and-python/", "17 hours"),
    ]
    for skill, cname, platform, url, duration in courses_data:
        db.session.add(Course(
            skill_name=skill,
            course_name=cname,
            platform=platform,
            url=url,
            duration=duration,
        ))
    db.session.commit()


# -----------------------------------------------------------------------------
# Security & Helper Functions
# -----------------------------------------------------------------------------
def token_for(email):
    """Signs a 24-hour JSON Web Token (JWT) for authenticated requests."""
    payload = {
        "sub": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=1),
    }
    return jwt.encode(payload, app.config["JWT_SECRET"], algorithm="HS256")


def user_response(user):
    """Standardized login response payload."""
    return {"token": token_for(user.email), "name": user.name, "email": user.email}


def student_dict(student):
    """Serializes student model along with their saved skill names."""
    if student is None:
        return None
    skills = [s.skill_name for s in Skill.query.filter_by(email=student.email).all()]
    return {
        "id": student.id,
        "email": student.email,
        "cgpa": student.cgpa,
        "internships": student.internships,
        "projects": student.projects,
        "backlogs": student.backlogs,
        "skills": skills,
    }


# -----------------------------------------------------------------------------
# Authentication Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/auth/register")
def register():
    """Registers a new student account with password hashing."""
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    name = str(data.get("name", "")).strip()
    password = str(data.get("password", "")).strip()

    if not email or not name or not password:
        return jsonify({"error": "Name, email and password are required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered!"}), 409

    db.session.add(User(
        name=name,
        email=email,
        password=generate_password_hash(password),
        role="STUDENT",
    ))
    db.session.commit()
    return jsonify({"message": "User registered successfully!"}), 201


@app.post("/api/auth/login")
def login():
    """Verifies credentials and returns a valid session token."""
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email or password!"}), 401

    return jsonify(user_response(user))


# -----------------------------------------------------------------------------
# Student Academic Records & Details Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/student/save")
def save_student():
    """
    Saves student academic indicators (CGPA, internships, projects, backlogs, skills).
    Performs boundary checks to prevent runtime errors and invalid entries.
    """
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()

    required = ("cgpa", "internships", "projects", "backlogs")
    if not email or any(data.get(field) is None for field in required):
        return jsonify({"error": "All fields (CGPA, internships, projects, backlogs) are required"}), 400

    try:
        cgpa = float(data["cgpa"])
        internships = int(data["internships"])
        projects = int(data["projects"])
        backlogs = int(data["backlogs"])
    except (ValueError, TypeError):
        return jsonify({"error": "CGPA must be a valid number, and internships, projects, and backlogs must be integers"}), 400

    # Sanity checks on academic boundaries
    if not (0.0 <= cgpa <= 10.0):
        return jsonify({"error": "CGPA must be between 0.0 and 10.0"}), 400
    if internships < 0 or projects < 0 or backlogs < 0:
        return jsonify({"error": "Internships, projects, and backlogs cannot be negative"}), 400

    # Overwrite previous entry for this student
    Student.query.filter_by(email=email).delete()
    Skill.query.filter_by(email=email).delete()

    db.session.add(Student(
        email=email,
        cgpa=cgpa,
        internships=internships,
        projects=projects,
        backlogs=backlogs,
    ))

    # Clean, trim, and deduplicate technical skills list
    raw_skills = data.get("skills") or []
    seen = set()
    for skill_name in raw_skills:
        clean_skill = str(skill_name).strip()
        if clean_skill and clean_skill.lower() not in seen:
            seen.add(clean_skill.lower())
            db.session.add(Skill(email=email, skill_name=clean_skill))

    db.session.commit()
    return jsonify({"message": "Student data updated successfully!"})


@app.get("/api/student/get/<email>")
def get_student(email):
    """Retrieves saved student profile and skills for form pre-population."""
    email = str(email).strip().lower()
    return jsonify(student_dict(Student.query.filter_by(email=email).first()))


# -----------------------------------------------------------------------------
# Placement Prediction Endpoint
# -----------------------------------------------------------------------------
@app.get("/api/prediction/<email>")
def predict(email):
    """
    Feeds student attributes into the trained Random Forest classifier.
    Calculates placement status ('PLACED' / 'NOT PLACED') and percentage chance.
    """
    email = str(email).strip().lower()
    student = Student.query.filter_by(email=email).first()

    if not student:
        return jsonify({
            "status": "ERROR",
            "probability": 0,
            "message": "Student details not found. Please enter your academic details first."
        }), 404

    if model is None:
        return jsonify({
            "status": "ERROR",
            "probability": 0,
            "message": "Prediction model is currently unavailable."
        }), 503

    skill_count = Skill.query.filter_by(email=email).count()
    feature_vector = [[student.cgpa, student.internships, student.projects, student.backlogs, skill_count]]

    probabilities = dict(zip(model.classes_, model.predict_proba(feature_vector)[0]))
    chance = round(probabilities.get("yes", 0) * 100, 1)
    placed = model.predict(feature_vector)[0] == "yes"

    return jsonify({
        "status": "PLACED" if placed else "NOT PLACED",
        "probability": chance,
        "message": "Congratulations! High chance of placement!" if placed
        else "Keep improving your skills!",
    })


# -----------------------------------------------------------------------------
# Baseline Skill Gap & Course Recommendations
# -----------------------------------------------------------------------------
def skill_gap(email, company_tier):
    """Compares student skills against target company tier requirements."""
    email = str(email).strip().lower()
    skills = Skill.query.filter_by(email=email).all()
    required = RequiredSkill.query.filter_by(company_tier=company_tier).all()

    yours_lower = {s.skill_name.strip().lower() for s in skills if s.skill_name.strip()}
    missing = [s.skill_name for s in required if s.skill_name.strip().lower() not in yours_lower]

    percentage = 0 if not required else int((len(required) - len(missing)) / len(required) * 100)
    if percentage >= 80:
        readiness = f"READY ✅ ({percentage}%)"
    elif percentage >= 50:
        readiness = f"PARTIALLY READY ⚠️ ({percentage}%)"
    else:
        readiness = f"NOT READY ❌ ({percentage}%)"

    return {
        "companyTier": company_tier,
        "yourSkills": [s.skill_name for s in skills],
        "requiredSkills": [s.skill_name for s in required],
        "missingSkills": missing,
        "readinessLevel": readiness,
    }


@app.get("/api/skillgap/<email>/<company_tier>")
def get_skill_gap(email, company_tier):
    return jsonify(skill_gap(email, company_tier))


@app.get("/api/learning/<email>/<company_tier>")
def learning_path(email, company_tier):
    """Finds courses from catalog to address missing skills."""
    email = str(email).strip().lower()
    gap = skill_gap(email, company_tier)
    courses = {}

    for missing_skill in gap["missingSkills"]:
        matches = Course.query.filter(db.func.lower(Course.skill_name) == missing_skill.strip().lower()).all()
        if matches:
            courses[missing_skill] = [c.as_dict() for c in matches]

    return jsonify({
        "email": email,
        "companyTier": company_tier,
        "missingSkills": gap["missingSkills"],
        "recommendedCourses": courses,
    })


# -----------------------------------------------------------------------------
# Google Gemini AI: Real-Time Dynamic Skill Gap & Course Discovery
# -----------------------------------------------------------------------------
def analyze_skill_gap_with_gemini(student_skills, company_name, target_role="Software Engineer"):
    """
    Calls Google Gemini AI to analyze up-to-date industry requirements
    for any company in the world and recommend tailored learning links.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not genai:
        return None

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
        You are an expert technical recruiter and software engineering career advisor.
        Target Company: {company_name}
        Target Role: {target_role}
        Candidate's Current Skills: {json.dumps(student_skills)}

        Analyze the current 2026 industry technical requirements for {company_name} for the role of {target_role}.
        Return ONLY a valid JSON object matching this schema:
        {{
            "company": "{company_name}",
            "role": "{target_role}",
            "requiredSkills": ["list of 6-8 core technical skills expected by {company_name} for this role"],
            "missingSkills": ["list of skills from requiredSkills that the candidate does not have"],
            "readinessScore": <integer between 0 and 100 representing percentage match>,
            "readinessLevel": "<READY ✅ (X%) | PARTIALLY READY ⚠️ (X%) | NOT READY ❌ (X%)>",
            "aiSummary": "<1-2 sentence recruiter summary assessing the candidate for this specific company>",
            "recommendedCourses": {{
                "<MissingSkillName>": [
                    {{
                        "skillName": "<MissingSkillName>",
                        "courseName": "<real, high-quality course or tutorial title>",
                        "platform": "<Coursera / Udemy / YouTube / Official Docs / LeetCode>",
                        "url": "<valid search or direct URL>",
                        "duration": "<e.g. 20 hours, 4 weeks>"
                    }}
                ]
            }}
        }}
        Do not wrap in markdown or backticks. Return strictly raw JSON.
        """

        response = None
        for model_name in ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.8-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            return None

        clean_text = response.text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.strip("`").removeprefix("json").strip()

        return json.loads(clean_text)

    except Exception as e:
        app.logger.warning(f"Gemini AI query encountered an issue: {e}")
        return None


@app.route("/api/ai/skillgap/<email>", methods=["GET", "POST"])
def ai_skill_gap(email):
    """
    AI-driven skill gap analyzer supporting any company and role.
    Uses Gemini AI if configured, with graceful fallback to catalog profiles.
    """
    email = str(email).strip().lower()
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        company = str(data.get("company", "Google")).strip()
        role = str(data.get("role", "Software Engineer")).strip()
    else:
        company = str(request.args.get("company", "Google")).strip()
        role = str(request.args.get("role", "Software Engineer")).strip()

    skills = [s.skill_name for s in Skill.query.filter_by(email=email).all()]

    # 1. Real-time Gemini AI analysis
    ai_result = analyze_skill_gap_with_gemini(skills, company, role)
    if ai_result:
        ai_result["isLiveAi"] = True
        ai_result["yourSkills"] = skills
        return jsonify(ai_result)

    # 2. Resilient fallback mapping for offline usage
    comp_upper = company.upper()
    if any(p in comp_upper for p in ["GOOGLE", "MICROSOFT", "AMAZON", "META", "APPLE", "NETFLIX", "PRODUCT"]):
        tier = "PRODUCT"
    elif any(s in comp_upper for s in ["TCS", "INFOSYS", "WIPRO", "ACCENTURE", "COGNIZANT", "SERVICE"]):
        tier = "SERVICE"
    else:
        tier = "STARTUP"

    fallback_gap = skill_gap(email, tier)
    fallback_learning = learning_path(email, tier).get_json()

    req_len = max(1, len(fallback_gap["requiredSkills"]))
    miss_len = len(fallback_gap["missingSkills"])
    readiness_pct = max(0, 100 - int((miss_len / req_len) * 100))

    return jsonify({
        "company": company,
        "role": role,
        "isLiveAi": False,
        "note": "Using baseline skill profiles. Add your free GEMINI_API_KEY in .env to unlock real-time Gemini AI analysis for any company!",
        "requiredSkills": fallback_gap["requiredSkills"],
        "yourSkills": fallback_gap["yourSkills"],
        "missingSkills": fallback_gap["missingSkills"],
        "readinessScore": readiness_pct,
        "readinessLevel": fallback_gap["readinessLevel"],
        "aiSummary": f"Baseline career path evaluation for {company} ({tier} tier). Add your free GEMINI_API_KEY in .env for custom AI insights!",
        "recommendedCourses": fallback_learning.get("recommendedCourses", {})
    })


# -----------------------------------------------------------------------------
# Static Web App Routing
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    """Serves the login & landing portal."""
    return send_from_directory(STATIC_DIR, "login.html")


# -----------------------------------------------------------------------------
# Application Startup Lifecycle
# -----------------------------------------------------------------------------
with app.app_context():
    db.create_all()
    seed_data()
    train_model()


if __name__ == "__main__":
    port_number = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port_number, debug=False)