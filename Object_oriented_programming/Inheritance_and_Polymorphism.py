# Class-level attributes
# Create a Player class
class Player():
    MAX_POSITION = 10 # this is class attribute
    def __init__(self): #this is method
        self.position = 0
    

# Print Player.MAX_POSITION       
print(Player.MAX_POSITION)

# Create a player p and print its MAX_POSITITON
p = Player()
print(p.MAX_POSITION)


#Class attributes like MAX_POSITION are accessed using class name, 
# but instance attributes like position are accessed using self.

class Player:    
    MAX_POSITION = 10
    def __init__(self):
        self.position = 0
    # Add a move() method with steps parameter
    def move(self, steps):
        if self.position + steps < Player.MAX_POSITION:
            self.position+=steps
        else:
            self.position = Player.MAX_POSITION
    def draw(self):
        drawing = "-" * self.position + "|" +"-"*(Player.MAX_POSITION - self.position)
        print(drawing)


    

       
# This method provides a rudimentary visualization in the console    

p = Player(); p.draw()
p.move(4); p.draw()
p.move(5); p.draw()
p.move(3); p.draw()


# Changing class attributes
# Create Players p1 and p2
p1, p2 = Player(), Player()

print("MAX_SPEED of p1 and p2 before assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

# ---MODIFY THIS LINE--- 
Player.MAX_SPEED = 7

print("MAX_SPEED of p1 and p2 after assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

print("MAX_SPEED of Player:")
# Print Player.MAX_SPEED
print(Player.MAX_SPEED)



# import datetime from datetime
from datetime import datetime

class BetterDate:
    def __init__(self, year, month, day):
      self.year, self.month, self.day = year, month, day
      
    @classmethod  #Use Decorator
    def from_str(cls, datestr):
        year, month, day = map(int, datestr.split("-"))
        return cls(year, month, day)
    
    # Define a class method from_datetime accepting a datetime object
    @classmethod  #Use Decorator
    def from_datetime(cls, datetime):
        year, month, day = datetime.year, datetime.month, datetime.day
        return cls(year, month, day)




# You should be able to run the code below with no errors: 
today = datetime.today()     
bd = BetterDate.from_datetime(today)   
print(bd.year)
print(bd.month)
print(bd.day)

#Create a subclass and learn inheritance
class Employee:
  MIN_SALARY = 30000    

  def __init__(self, name, salary=MIN_SALARY):
      self.name = name
      if salary >= Employee.MIN_SALARY:
        self.salary = salary
      else:
        self.salary = Employee.MIN_SALARY
  def give_raise(self, amount):
    self.salary += amount      
        
# MODIFY Manager class and add a display method
class Manager(Employee):
  def display(self):
    print('Manager' + self.name)

mng = Manager("Debbie Lashko", 86500)
print(mng.name)

# Call mng.display()
mng.display()

# Polymorphism
class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

        
class Manager(Employee):
    def display(self):
        print("Manager ", self.name)

    def __init__(self, name, salary=50000, project=None):
        Employee.__init__(self, name, salary)
        self.project = project

    # Add a give_raise method
    def give_raise(self, amount, bonus = 1.05):
        Employee.give_raise(self, amount*bonus)

    
mngr = Manager("Ashta Dunbar", 78500)
mngr.give_raise(1000)
print(mngr.salary)
mngr.give_raise(2000, bonus=1.03)
print(mngr.salary)


# Customizing a DataFrame
# Import pandas as pd
import pandas as pd

# Define LoggedDF inherited from pd.DataFrame and add the constructor
class LoggedDF(pd.DataFrame):
  
  def __init__(self, *args, **kwargs):
    pd.DataFrame.__init__(self, *args, **kwargs)
    self.created_at = datetime.today()
    
  def to_csv(self, *args, **kwargs):
    # Copy self to a temporary DataFrame
    temp = self.copy()
    
    # Create a new column filled with self.created at
    temp["created_at"] = self.created_at
    
    # Call pd.DataFrame.to_csv on temp with *args and **kwargs
    pd.DataFrame.to_csv(temp, *args, **kwargs)
    

    