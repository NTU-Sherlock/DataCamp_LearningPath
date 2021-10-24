#The square brackets around imag in the documentation showed us that the imag argument is optional
?complex()
# Init signature: complex(self, /, *args, **kwargs)
# Docstring:
# complex(real[, imag]) -> complex number

# Create a complex number from a real part and an optional imaginary part.
# This is equivalent to (real + imag*1j) where imag defaults to 0.
# Type:           type


# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place,place_up)

# Print out the number of o's in place
print(place.count('o'))

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20))

# Print out how often 9.50 appears in areas
print(areas.count(9.5))



# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5); areas.append(15.45)


# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)



###Import certain function
# import math, make all functionality from the math package available to you. 
# However, if you decide to only use a specific part of a package, you can always make your import more selective:

from math import pi



# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2*math.pi*r

# Calculate A
A = math.pi*r**2

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))


# Definition of radius
r = 192500

# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
dist = radians(12) * r

# Print out dist
print(dist)