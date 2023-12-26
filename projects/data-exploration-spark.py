# Databricks notebook source
# MAGIC %md ## Data Exploration - Spark / SQL
# MAGIC - How to Process IoT Device JSON Data Using Dataset

# COMMAND ----------

# MAGIC %md ####Reading JSON as a Dataset

# COMMAND ----------

# MAGIC %md Use the Scala case class *DeviceIoTData* to convert the JSON device data into a dataframe. There is GeoIP information for each device entry:
# MAGIC * IP address
# MAGIC * ISO-3166-1 two and three letter codes
# MAGIC * Country Name
# MAGIC * Latitude and longitude
# MAGIC
# MAGIC With these attributes as part of the device data, we can map and visualize them as needed. For each IP associated with a *device_id*, I optained the above attributes from a webservice at http://freegeoip.net/csv/ip
# MAGIC
# MAGIC *{"device_id": 198164, "device_name": "sensor-pad-198164owomcJZ", "ip": "80.55.20.25", "cca2": "PL", "cca3": "POL", "cn": "Poland", "latitude": 53.080000, "longitude": 18.620000, "scale": "Celsius", "temp": 21, "humidity": 65, "battery_level": 8, "c02_level": 1408, "lcd": "red", "timestamp" :1458081226051 }*
# MAGIC
# MAGIC This dataset is avaialbe from Public S3 bucket //databricks-public-datasets/data/iot or https://github.com/dmatrix/examples/blob/master/spark/databricks/notebooks/py/data/iot_devices.json
# MAGIC

# COMMAND ----------

# MAGIC %fs ls "FileStore/tables/"

# COMMAND ----------

# read the json file and create the dataframe


file_location = "/FileStore/tables/iot_devices.json"
file_type = "json"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)



# COMMAND ----------

# Create a view or table

temp_table_name = "iot_devices_json"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md Displaying your Dataset

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md #### Data exploration

# COMMAND ----------

# MAGIC %md Top five entries

# COMMAND ----------

df.take(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC For all relational expressions, the [Catalyst Optimizer](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html) will formulate an optimized logical and physical plan for execution, and [Tungsten](https://databricks.com/blog/2015/04/28/project-tungsten-bringing-spark-closer-to-bare-metal.html) engine will optimize the generated code. For our *DeviceIoTData*, it will use its standard encoders to optimize its binary internal representation, hence decrease the size of generated code, minimize the bytes transfered over the networks between nodes, and execute faster.
# MAGIC
# MAGIC For instance, let's first filter the device dataset on *temp* and *humidity* attributes with a predicate and display the first 10 items.
# MAGIC
# MAGIC

# COMMAND ----------

# issue select, map, filter operations on the dataframes

from pyspark.sql.functions import col, asc
from pyspark.sql.functions import *
TempFilter = df.filter(col("temp") > 30).filter(col("humidity") > 70)
display(TempFilter)


# COMMAND ----------

# MAGIC %md Use filter to filter out dataframe rows that met the temperature and humidity predicate

# COMMAND ----------

# filter out rows that meet the temperature and humimdity predicate
TempFilter10 = df.filter(col("temp") > 30).filter(col("humidity") > 70).take(10)
TempFilter10

# COMMAND ----------

# Mapping four fields- temp, device_name, device_id, cca3 
dfTempMap = df.where((col("temp") > 25)).rdd.map(lambda d: (d.temp, d.device_name, d.device_id, d.cca3))
dfTempMap

# COMMAND ----------

display(dfTempMap.toDF())

# COMMAND ----------

# MAGIC %md Now use the filter() method that is equivalent as the where() method used above.

# COMMAND ----------

dfTemp25 = df.filter(col("temp") > 25).rdd.map(lambda d: (d.temp, d.device_name, d.device_id, d.cca3))

display(dfTemp25.toDF())

# COMMAND ----------

# MAGIC %md select() where battery_level is greater than 6, sort in asceding order on C02_level.

# COMMAND ----------

display(df.select("battery_level", "c02_level", "device_name").where(col("battery_level") > 6).sort(col("c02_level")))

# COMMAND ----------

# MAGIC %md Let's see how to use groupBy() and avg(). 
# MAGIC Let's take all temperatures readings > 25, along with their corresponding devices' humidity, groupBy ccca3 country code, and compute averages. Plot the resulting Dataset.

# COMMAND ----------

from pyspark.sql.functions import avg

dfAvgTmp = df.filter(col("temp") > 25).rdd.map(lambda d: (d.temp, d.humidity, d.cca3)).toDF().groupBy("_3").agg(avg("_1"), avg("_2"))


display(dfAvgTmp)


# COMMAND ----------

# MAGIC %md #### Visualizing datasets

# COMMAND ----------

# MAGIC %md **Finally, the fun bit!**
# MAGIC
# MAGIC Data without visualization without a narrative arc, to infer insights or to see a trend, is useless. We always desire to make sense of the results.
# MAGIC
# MAGIC By saving our Dataset, as a temporary table, I can issue complex SQL queries against it and visualize the results, using notebook's myriad plotting options.

# COMMAND ----------

df.createOrReplaceTempView("iot_device_data")

# COMMAND ----------

# MAGIC %md Count all devices for a partiular country and map them

# COMMAND ----------

# MAGIC %sql select cca3, count(distinct device_id) as device_id from iot_device_data group by cca3 order by device_id desc limit 100

# COMMAND ----------

# MAGIC %md Let's visualize the results as a pie chart and distribution for devices in the country where C02 are high.

# COMMAND ----------

# MAGIC %sql select cca3, c02_level from iot_device_data where c02_level > 1400 order by c02_level desc

# COMMAND ----------

# MAGIC %md Select all countries' devices with high-levels of C02 and group by cca3 and order by device_ids 

# COMMAND ----------

# MAGIC %sql select cca3, count(distinct device_id) as device_id from iot_device_data where lcd == 'red' group by cca3 order by device_id desc limit 100

# COMMAND ----------

# MAGIC %md find out all devices in countries whose batteries need replacements 

# COMMAND ----------

# MAGIC %sql select cca3, count(distinct device_id) as device_id from iot_device_data where battery_level == 0 group by cca3 order by device_id desc limit 100

# COMMAND ----------

# MAGIC %md Converting a Dataset to RDDs.
# MAGIC
# MAGIC

# COMMAND ----------

deviceEvents = df.select("device_name","cca3","c02_level").where(col("c02_level") > 1300)

eventsRDD = deviceEvents.take(10)


# COMMAND ----------

display(deviceEvents)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Explain the main differences between RDDs, Dataframes and Datasets
# MAGIC
# MAGIC - RDD is an immutable collection of elements of data, and it is possible to use parallel operations including transformations and actions. It is to be used when the data is unstructured, actions can be taken without a fixed schema, there is no need for optimization and performance like we have with DataFrames and Datasets and we would like to use low-level transformation and actions.
# MAGIC DataFrames are also immutable collection of elements of data, but the data is structured. It is organized like a relational database and the actions can be taken with a schema, like retrieving data attributes by column names. It has the benefit of optimization and performance, since it is easier to process large data sets. Additionally, the commands to manipulate are more accessible and easier to understand. It is benefitial for R and Python users.
# MAGIC Datasets have two types of API, strongly-typed and untyped. It has many benefits and optimizations when compared to other data. For example, syntax errors and analysis errors are catched duting compile time. With DataFrames, the analysis errors are only observed during runtime. It also works well with semi-structured data and it is easier to code operations when compared to RDD. Additionaly, the memory usage is more efficient.

# COMMAND ----------

# MAGIC %md 2.1 How many sensor pads are reported to be from Poland

# COMMAND ----------

# MAGIC %sql select cca3, count(distinct device_id) as count_sensor from iot_device_data where device_name LIKE '%sensor-pad%' AND cca3 == 'POL' group by cca3

# COMMAND ----------

# MAGIC %md
# MAGIC - There are 1413 sensors from Poland

# COMMAND ----------

# MAGIC %md 2.2 How many different LCDs (distinct colors) are present in the dataset
# MAGIC

# COMMAND ----------

# MAGIC %sql select lcd, count(distinct device_id) as count_lcd from iot_device_data group by lcd sort by count_lcd DESC

# COMMAND ----------

# MAGIC %md
# MAGIC - There are 99051 yellow, 49699 green and 49414 red lcds.

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Find 5 countries that have the largest number of MAC devices used

# COMMAND ----------

# MAGIC %sql select cca3, count(distinct device_id) as count_mac from iot_device_data where device_name LIKE '%device-mac%' group by cca3 sort by count_mac DESC limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC - The five countries with largest number of MAC devices are: United States, China, Japan, Korea and Germany.

# COMMAND ----------

# MAGIC %md 2.4 Propose and try an interesting statistical test or machine learning model you could use to gain insight from this dataset. Note, you don't have to use Machine Learning for this question. You can apply any analysis to the data even using SparkSQL, Python visualization libraries to analyze the data. Another example cloud be to apply correlation functions or other Spark functions to analyze the data. 

# COMMAND ----------

#Checking info about the data elements
df.cache()
df.printSchema()

# COMMAND ----------

#Checking more info about the continuous variables
df.describe().toPandas().transpose()

# COMMAND ----------

#We can analyze the continuous variables and plot a scatter matrix to visualize relations between data elements
import pandas as pd
from pandas.plotting._misc import scatter_matrix

numeric_features = ["battery_level","c02_level","humidity","temp"]
sampled_data = df.select(numeric_features).sample(False, 0.8).toPandas()
axs = scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# COMMAND ----------

#Analyzing the correlation between the c02 level and the other numerical variables
import six
for i in df.columns:
    if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to c02_level for ", i, df.stat.corr('c02_level',i))

# COMMAND ----------

#Analyzing the correlation between the battery level and the other numerical variables
import six
for i in df.columns:
    if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to battery_level for ", i, df.stat.corr('battery_level',i))

# COMMAND ----------

# MAGIC %md
# MAGIC - From the scatter matrix plot, it was not clear that there was any relation between the numerical variables. When analyzing the correlation factors between c02 and battery level with respect to the other variables, we can see that the factors are very small (of the order of 0.001) which means that there is no significant linear correlation between the variables.
