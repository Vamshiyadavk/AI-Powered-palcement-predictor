import os
import sys
import unittest
from pathlib import Path

# Use an in-memory database for fast, isolated test execution
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"

# Ensure app module can be discovered from the tests folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app, db, seed_data


class PlacementPredictorTestSuite(unittest.TestCase):
    """
    Automated regression and validation test suite for student placement API.
    Covers authentication, student profile CRUD, ML predictions, and AI skill gaps.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def setUp(self):
        """Recreates tables and re-seeds baseline data before each test."""
        with app.app_context():
            db.drop_all()
            db.create_all()
            seed_data()

    def test_registration_and_duplicate_prevention(self):
        """Verifies successful registration and rejection of duplicate emails."""
        # Valid registration
        res = self.client.post("/api/auth/register", json={
            "name": "Vamshi",
            "email": "vamshi@example.com",
            "password": "strongPassword123"
        })
        self.assertIn(res.status_code, (200, 201))

        # Duplicate email should return 409 Conflict
        dup_res = self.client.post("/api/auth/register", json={
            "name": "Vamshi",
            "email": "vamshi@example.com",
            "password": "strongPassword123"
        })
        self.assertEqual(dup_res.status_code, 409)

    def test_user_authentication(self):
        """Ensures valid users receive a JWT token and bad passwords are rejected."""
        self.client.post("/api/auth/register", json={
            "name": "Alice",
            "email": "alice@example.com",
            "password": "correctPassword"
        })

        # Correct password returns 200 + token
        login_res = self.client.post("/api/auth/login", json={
            "email": "alice@example.com",
            "password": "correctPassword"
        })
        self.assertEqual(login_res.status_code, 200)
        self.assertIn("token", login_res.json)

        # Wrong password returns 401 Unauthorized
        bad_login = self.client.post("/api/auth/login", json={
            "email": "alice@example.com",
            "password": "wrongPassword"
        })
        self.assertEqual(bad_login.status_code, 401)

    def test_student_profile_persistence_and_skills(self):
        """Tests that academic parameters and technical skills save and retrieve correctly."""
        save_res = self.client.post("/api/student/save", json={
            "email": "student@example.com",
            "cgpa": 8.75,
            "internships": 2,
            "projects": 3,
            "backlogs": 0,
            "skills": ["Python", "SQL", "React.js"]
        })
        self.assertEqual(save_res.status_code, 200)

        # Verify saved data via GET
        get_res = self.client.get("/api/student/get/student@example.com")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json
        self.assertEqual(data["cgpa"], 8.75)
        self.assertEqual(data["internships"], 2)
        self.assertIn("Python", data["skills"])
        self.assertIn("React.js", data["skills"])

    def test_placement_prediction_model(self):
        """Verifies the Random Forest classifier generates predictions for registered profiles."""
        self.client.post("/api/student/save", json={
            "email": "highachiever@example.com",
            "cgpa": 9.2,
            "internships": 2,
            "projects": 4,
            "backlogs": 0,
            "skills": ["Python", "Java", "SQL", "Docker"]
        })

        pred_res = self.client.get("/api/prediction/highachiever@example.com")
        self.assertEqual(pred_res.status_code, 200)
        self.assertIn(pred_res.json["status"], {"PLACED", "NOT PLACED"})
        self.assertGreaterEqual(pred_res.json["probability"], 0.0)

    def test_company_tier_skill_gap_analysis(self):
        """Tests that skill gap computation identifies missing competencies and suggests courses."""
        self.client.post("/api/student/save", json={
            "email": "student@example.com",
            "cgpa": 8.0,
            "internships": 1,
            "projects": 2,
            "backlogs": 0,
            "skills": ["Python", "SQL"]
        })

        gap_res = self.client.get("/api/skillgap/student@example.com/PRODUCT")
        self.assertEqual(gap_res.status_code, 200)
        gap = gap_res.json
        self.assertIn("System Design", gap["missingSkills"])
        self.assertNotIn("Python", gap["missingSkills"])

        learn_res = self.client.get("/api/learning/student@example.com/PRODUCT")
        self.assertEqual(learn_res.status_code, 200)
        self.assertIn("System Design", learn_res.json["recommendedCourses"])

    def test_ai_live_skill_gap_endpoint(self):
        """Tests the unified AI skill gap endpoint with company parameter."""
        self.client.post("/api/student/save", json={
            "email": "student@example.com",
            "cgpa": 8.0,
            "internships": 1,
            "projects": 2,
            "backlogs": 0,
            "skills": ["Python", "SQL"]
        })

        res = self.client.get("/api/ai/skillgap/student@example.com?company=Google&role=Software+Engineer")
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["company"], "Google")
        self.assertIn("requiredSkills", data)
        self.assertIn("recommendedCourses", data)

    def test_defensive_input_validation(self):
        """Validates that malformed inputs and boundary violations are caught with 400 errors."""
        # Missing required registration fields
        res1 = self.client.post("/api/auth/register", json={"name": "No Password"})
        self.assertEqual(res1.status_code, 400)

        # Null CGPA (from blank form field)
        res2 = self.client.post("/api/student/save", json={
            "email": "test@example.com",
            "cgpa": None,
            "internships": 1,
            "projects": 1,
            "backlogs": 0
        })
        self.assertEqual(res2.status_code, 400)

        # Out-of-bounds CGPA (> 10.0)
        res3 = self.client.post("/api/student/save", json={
            "email": "test@example.com",
            "cgpa": 14.5,
            "internships": 1,
            "projects": 1,
            "backlogs": 0
        })
        self.assertEqual(res3.status_code, 400)

        # Prediction requested before details are entered
        res4 = self.client.get("/api/prediction/unknown_student@example.com")
        self.assertEqual(res4.status_code, 404)


if __name__ == "__main__":
    unittest.main()