from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="Student API", version="1.0")
 # Same in-memory list as Flask example
students = [
 {"id": 1, "name": "Ali", "grade": "A"},
 {"id": 2, "name": "Sara", "grade": "B"},
 ]


# Used when CREATING a student (POST)
class StudentCreate(BaseModel):
    name: str # required — must be a string
    grade: Optional[str] = "N/A" # optional — defaults to "N/A"
 # Used when UPDATING a student (PUT)
class StudentUpdate(BaseModel):
    name: Optional[str] = None # both fields are optional
    grade: Optional[str] = None # only send what you want to change



@app.get("/students")
def get_all_students():
 return students # FastAPI converts to JSON automatically



@app.post("/students", status_code=201)
def add_student(student: StudentCreate): # Pydantic does all parsing
    new_student = {
    "id": len(students) + 1,
    "name": student.name, # dot notation — not student['name']
    "grade": student.grade
    }
    students.append(new_student)
    return new_student
