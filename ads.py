from pydantic import BaseModel

class ADs(BaseModel):
    age: int
    EstimatedSalary: int 
    gender: int 
