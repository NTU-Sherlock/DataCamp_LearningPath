# double forward-slashes to navigate to all future generations. 
# navigating to all paragraph p elements within any HTML code.
xpath = '//p'
# select all span elements whose class attribute equals "span-class"
xpath = '//span[@class="span-class"]'

# Import a scrapy Selector
from scrapy import Selector

# Import requests
import requests

# Create the string html containing the HTML source
html = requests.get( url ).content

# Create the Selector object sel from html
sel = Selector( text = html)

# Print out the number of elements in the HTML document
print( "There are 1020 elements in the HTML document.")
print( "You have found: ", len( sel.xpath('//*') ) )