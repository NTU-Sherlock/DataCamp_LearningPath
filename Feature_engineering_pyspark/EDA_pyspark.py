###Check the version first
# Return spark version
print(spark.version)

# Return python version
import sys
print(sys.version_info)

# Read the file into a dataframe
df = spark.read.parquet('Real_Estate.parq')
# Print columns in dataframe
print(df.columns)