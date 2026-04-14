from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional

class StudentBase(BaseModel):
    name: str
    email: EmailStr
    age: int = Field(gt=0, description="Age must be positive")
    course: str

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, gt=0)
    course: Optional[str] = None

class StudentResponse(StudentBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
