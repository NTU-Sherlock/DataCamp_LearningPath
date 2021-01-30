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
        Player.MAX_POSITION
        if self.position + steps < Player.MAX_POSITION:
            self.position+=steps
        else:
            self.position = Player.MAX_POSITION


    

       
    # This method provides a rudimentary visualization in the console    
    def draw(self):
        drawing = "-" * self.position + "|" +"-"*(Player.MAX_POSITION - self.position)
        print(drawing)

p = Player(); p.draw()
p.move(4); p.draw()
p.move(5); p.draw()
p.move(3); p.draw()