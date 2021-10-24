# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
upstairs = areas[6:]

# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ['poolhouse',24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ['garage',15.45]

#Delete the data in a list 
areas_3  = del(areas[-3])


#Inner workings of lists
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = areas 

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)

#vs.
# Create areas_copy with [:] or list()
areas_copy = areas[:]
areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)