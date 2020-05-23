# Create a distplot of the Award Amount
sns.distplot(df['Award_Amount'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})

# Plot the results
plt.show()


# Simpe regplot vs. Advanced lmplot


# Create an lmplot of premiums vs. insurance_losses

# Create an lmplot of premiums vs. insurance_losses
sns.lmplot(data = df, x = 'insurance_losses', y= 'premiums')
# Display the second plot
plt.show()

sns.lmplot(data = df, x = 'insurance_losses', y= 'premiums')
# Display the second plot
plt.show()


## Hue can add one more dimension
# Create a regression plot using hue
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           hue="Region")

# Show the results
plt.show()

## Row can split frame
# Create a regression plot with multiple rows
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           row="Region")

# Show the plot
plt.show()