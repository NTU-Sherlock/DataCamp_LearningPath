#LSP Liskov Substitutes Principles
# The classic example of a problem that violates the Liskov Substitution Principle 
# is the Circle-Ellipse problem, Square-Rectangle problem.

# Define a Rectangle class
class Rectangle:
    def __init__(self, h, w):
      self.h, self.w = h, w

# Define a Square class
class Square(Rectangle):
    def __init__(self, w):
      self.h, self.w = w, w  
# Create a Square object with side length 4.
# Then try assigning 7 to the h attribute.
# The 4x4 Square object would no longer be a square if we assign 7 to h.



class Rectangle:
    def __init__(self, w,h):
      self.w, self.h = w,h
      
# Define set_h to set h       
    def set_h(self, h):
      self.h = h

# Define set_w to set w
    def set_w(self, w):
      self.w = w   
      
class Square(Rectangle):
    def __init__(self, w):
      self.w, self.h = w, w 
      
# Define set_h to set w and h 
    def set_h(self, h):
      self.h = h
      self.w = h
      
# Define set_w to set w and h 
    def set_w(self, w):
      self.w = w   
      self.h = w 

# The single leading underscore is a convention for internal details of implementation.
# Double leading underscores are used for attributes that should not be inherited to aviod name clashes in child classes.
# Finally, leading and trailing double underscores are reserved for built-in methods.