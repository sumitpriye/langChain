from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int

new_person={'name':'Sumit','age':'30'}
person=Person(**new_person)
print(person)