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

# Create your own continuous color scale
my_scale = ['rgb(255,0,0)', 'rgb(3,252,40)']

# Create the bar plot
fig = px.bar(data_frame= student_scores, 
             x='student_name', y='score', title='Student Scores by Student',
             # Set the color variable and scale
             color='score',
             color_continuous_scale=my_scale
             )

# Show the plot
fig.show()

# Create the industry-color map (Box plot)
ind_color_map = {'Tech': 'rgb(124,250,120)', 'Oil': 'rgb(112,128,144)', 
                 'Pharmaceuticals': 'rgb(137, 109, 247)', 'Professional Services' : 'rgb(255, 0, 0)'}

# Create the basic box plot
fig = px.box(
  			# Set the data and y variable
  			data_frame=revenues, y='Revenue',
  			# Set the color map and variable
			color_discrete_map=ind_color_map,
			color='Industry')

# Show the plot
fig.show()

# Create the industry-color map (Histogram)
ind_color_map = {'Tech': 'rgb(124,250,120)', 'Oil': 'rgb(112,128,144)', 
                 'Pharmaceuticals': 'rgb(137, 109, 247)', 'Professional Services' : 'rgb(255, 0, 0)'}

# Create a simple histogram
fig = px.histogram(
  			# Set the data and x variable
  			data_frame=revenues, x= 'Revenue', nbins=5,
    		# Set the color map and variable
			color_discrete_map=ind_col	or_map,
			color='Industry')

# Show the plot
fig.show()