# AI-Powered Student Placement Predictor 🎓

A modern, full-stack Python Flask web application that stores student academic records, predicts placement outcomes using Machine Learning (Random Forest), and performs **real-time AI skill gap analysis and course recommendations powered by Google Gemini AI**.

---

## 📁 Clean Project Structure

```text
placement-predictor-main/
│
├── app.py                     # 🚀 Main Flask application, routes, ML model & Gemini AI
├── requirements.txt           # 📦 Python library dependencies
├── .env                       # 🔑 Active local environment variables & GEMINI_API_KEY
├── .env.example               # 📋 Template configuration
├── .gitignore                 # 🙈 Git ignore rules
├── README.md                  # 📖 Project guide and documentation
│
├── static/                    # 🌐 Frontend Web Interface (HTML & CSS)
│   ├── login.html             #    Student registration & login page
│   ├── dashboard.html         #    Student home portal
│   ├── marks.html             #    Academic details & skills input form
│   ├── result.html            #    Placement outcome & Gemini AI live skill gap analyzer
│   └── style.css              #    UI styling & animations
│
├── data/                      # 📊 Machine Learning Data
│   └── placement_data.arff    #    600-sample realistic student recruitment training dataset
│
├── instance/                  # 💾 Local Database
│   └── placement.db           #    SQLite database (users, marks, skills, seed courses)
│
└── tests/                     # 🧪 Automated Testing
    └── test_app.py            #    Comprehensive API & validation test suite
```

---

## ✨ Features

- **Student Authentication**: Secure registration and login with hashed passwords and JWT.
- **Academic Profile Management**: Tracks CGPA, internships, projects, backlogs, and technical skills.
- **Machine Learning Placement Predictor**: Random Forest model (82.5% accuracy) evaluating placement probability based on historical patterns.
- **🤖 Live Google Gemini AI Skill Gap Analysis**:
  - Enter **any company in the world** (Google, Amazon, NVIDIA, Microsoft, TCS, Startups) and role.
  - Dynamically fetches real-time required industry skills.
  - Identifies student's missing skill gaps and calculates personalized readiness.
  - Recommends direct course links from Coursera, Udemy, YouTube, and LeetCode.
- **Auto-Prefill & Defensive Validation**: Forms automatically reload previously saved student records with full error prevention.

---

## 🚀 Quick Setup & Run

### 1. Requirements
* Python 3.10 or newer

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
python tests/test_app.py
```

### 4. Start the Application
```powershell
python app.py
```

### 5. Open in Your Browser
Navigate to **`http://127.0.0.1:8080/`**

---

## 🔑 AI Key Configuration

The application includes an active Google Gemini API key configured in `.env`.
To use your own key, generate one for free at **[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)** and update `.env`:

```env
GEMINI_API_KEY=your_key_here
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new student |
| `POST` | `/api/auth/login` | Login and receive JWT token |
| `POST` | `/api/student/save` | Save or update student marks & skills |
| `GET` | `/api/student/get/<email>` | Fetch saved student profile & skills |
| `GET` | `/api/prediction/<email>` | Run Random Forest placement prediction |
| `GET`/`POST` | `/api/ai/skillgap/<email>` | Real-time Gemini AI skill gap & courses for any company |
| `GET` | `/api/skillgap/<email>/<tier>` | Baseline skill gap analysis by tier |
| `GET` | `/api/learning/<email>/<tier>` | Baseline recommended courses by tier |