from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

# Attempt to remove the database if it exists to start fresh, but don't fail if it's locked
try:
    if os.path.exists("./students.db"):
        os.remove("./students.db")
except PermissionError:
    print("Warning: Could not remove students.db, it may be in use.")

def test_create_student():
    response = client.post(
        "/create-student/",
        json={"name": "Faiq", "email": "faiq@example.com", "age": 20, "course": "Python"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Faiq"
    assert data["email"] == "faiq@example.com"
    assert "id" in data

def test_create_student_invalid_age():
    response = client.post(
        "/create-student/",
        json={"name": "Faiq", "email": "faiq2@example.com", "age": -5, "course": "Python"}
    )
    assert response.status_code == 422 # Validation error

def test_create_student_duplicate_email():
    client.post(
        "/create-student/",
        json={"name": "Faiq", "email": "duplicate@example.com", "age": 20, "course": "Python"}
    )
    response = client.post(
        "/create-student/",
        json={"name": "Another", "email": "duplicate@example.com", "age": 21, "course": "JS"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"

def test_get_all_students():
    response = client.get("/students-all/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_student_by_id():
    # Create one first
    create_res = client.post(
        "/create-student/",
        json={"name": "Search Me", "email": "search@example.com", "age": 25, "course": "AI"}
    )
    student_id = create_res.json()["id"]
    
    response = client.get(f"/student/{student_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Search Me"

def test_update_student():
    create_res = client.post(
        "/create-student/",
        json={"name": "Update Me", "email": "update@example.com", "age": 30, "course": "Cloud"}
    )
    student_id = create_res.json()["id"]
    
    response = client.put(
        f"/update-students/{student_id}",
        json={"name": "Updated Name", "age": 31}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Name"
    assert response.json()["age"] == 31
    assert response.json()["email"] == "update@example.com" # Should remain unchanged

def test_delete_student():
    create_res = client.post(
        "/create-student/",
        json={"name": "Delete Me", "email": "delete@example.com", "age": 40, "course": "DevOps"}
    )
    student_id = create_res.json()["id"]
    
    response = client.delete(f"/delete-students/{student_id}")
    assert response.status_code == 204
    
    # Verify it's gone
    get_res = client.get(f"/student/{student_id}")
    assert get_res.status_code == 404

if __name__ == "__main__":
    # If running directly, run all tests
    try:
        test_create_student()
        print("test_create_student passed")
        test_create_student_invalid_age()
        print("test_create_student_invalid_age passed")
        test_create_student_duplicate_email()
        print("test_create_student_duplicate_email passed")
        test_get_all_students()
        print("test_get_all_students passed")
        test_get_student_by_id()
        print("test_get_student_by_id passed")
        test_update_student()
        print("test_update_student passed")
        test_delete_student()
        print("test_delete_student passed")
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
