# Print the mystery employee's name
print(mystery.name)

# Print the mystery employee's salary
print(mystery.salary)

help(mystery)
# Give the mystery employee a raise of $2500
mystery.give_raise(2500)

# Print the salary again
print(mystery.salary)


## First Class
class Employee:
    def set_name(self, new_name):
        self.name = new_name

    def set_salary(self, new_salary):
        self.salary = new_salary 

    def give_raise(self, amount):
        self.salary = self.salary + amount

    # Add monthly_salary method that returns 1/12th of salary attribute
    def monthly_salary(self):
        return self.salary/12

    
emp = Employee()
emp.set_name('Korel Rossi')
emp.set_salary(50000)
# Get monthly salary of emp and assign to mon_sal
mon_sal = emp.monthly_salary()

# Print mon_sal
print(mon_sal)


## Use Constructor
# Import datetime from datetime
from datetime import datetime

class Employee:
    
    def __init__(self, name, salary=0):
        self.name = name
        if salary > 0:
          self.salary = salary
        else:
          self.salary = 0
          print("Invalid salary!")
          
        # Add the hire_date attribute and set it to today's date
        self.hire_date = datetime.today()
        
   # ...Other methods omitted for brevity ...
      
emp = Employee("Korel Rossi", -1000)
print(emp.name)
print(emp.salary)
print(emp.hire_date)


## First OOP
# Write the class Point as outlined in the instructions
import numpy as np
class Point:
    def __init__(self, x=0.0,y=0.0):
        self.x = x
        self.y = y
    def distance_to_origin(self):
        return np.sqrt(self.x**2+self.y**2)
    def reflect(self, axis):
        if axis == "x":
            self.y = -self.y
        if axis == "y":
            self.x = -self.x
        else:
            print('There has been an error in the system')
  