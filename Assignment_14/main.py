from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List

from database import engine, get_db, Base
import models, schemas

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Student Management System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend Files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def read_root():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")

# 1. Create Students
@app.post("/create-student/", response_model=schemas.StudentResponse, status_code=status.HTTP_201_CREATED)
def create_student(student: schemas.StudentCreate, db: Session = Depends(get_db)):
    # Check if email exists
    db_student = db.query(models.Student).filter(models.Student.email == student.email).first()
    if db_student:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    new_student = models.Student(**student.model_dump())
    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    return new_student

# 2. Get All Students
@app.get("/students-all/", response_model=List[schemas.StudentResponse])
def get_all_students(db: Session = Depends(get_db)):
    students = db.query(models.Student).all()
    return students

# 3. Set Single student with id (The user wrote "Set Single student with id" but the endpoint is GET /student/{id})
@app.get("/student/{id}", response_model=schemas.StudentResponse)
def get_student(id: int, db: Session = Depends(get_db)):
    student = db.query(models.Student).filter(models.Student.id == id).first()
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with id {id} not found"
        )
    return student

# 4. Update Student by id
@app.put("/update-students/{id}", response_model=schemas.StudentResponse)
def update_student(id: int, student_update: schemas.StudentUpdate, db: Session = Depends(get_db)):
    db_student = db.query(models.Student).filter(models.Student.id == id).first()
    if db_student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with id {id} not found"
        )
    
    update_data = student_update.model_dump(exclude_unset=True)
    
    # If email is being updated, check if it's already taken
    if "email" in update_data and update_data["email"] != db_student.email:
        existing_email = db.query(models.Student).filter(models.Student.email == update_data["email"]).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    for key, value in update_data.items():
        setattr(db_student, key, value)
    
    db.commit()
    db.refresh(db_student)
    return db_student

# 5. Delete Student by id
@app.delete("/delete-students/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(id: int, db: Session = Depends(get_db)):
    db_student = db.query(models.Student).filter(models.Student.id == id).first()
    if db_student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with id {id} not found"
        )
    
    db.delete(db_student)
    db.commit()
    return None
