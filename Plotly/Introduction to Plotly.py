# Create the box plot (with specific columns hovering)
fig = px.box(
  			# Set the data
  			data_frame= revenues, 
  			# Set the y variable
            y= 'Revenue', 
            # Add in hover data to see outliers
            hover_data=['Company'])

# Show the plot
fig.show()

# Create a simple histogram with specific bins(nbins)
fig = px.histogram(
  			data_frame=revenues, 
            # Set up the x-axis
           	x = 'Revenue',
            # Set the number of bins
            nbins = 5
            )

# Show the plot
fig.show()