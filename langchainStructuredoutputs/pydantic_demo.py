from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10,default=7.0)

new_student={'name':'Sumit','age':'23','email':'sumit@gmail.com','cgpa':9}
# coerce
student = Student(**new_student)

print(student)

student_dict = student.model_dump()
print(student_dict['age'])
print(student_dict['email'])


student_json = student.model_dump_json()
print(student_json)