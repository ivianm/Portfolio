# Databricks notebook source
# MAGIC %md
# MAGIC ## Diamond Price Prediction - Spark ML

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Using the Apache Spark ML pipeline, we will build a model to predict the price of a diamond based on the available features.
# MAGIC
# MAGIC Read from the following notebook for details about dataset.
# MAGIC
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5915990090493625/4396972618536508/6085673883631125/latest.html

# COMMAND ----------

#Reading the datafile
diamonds = (spark.read
  .option("header", "true")
  .csv("dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"))

# COMMAND ----------

display(diamonds)

# COMMAND ----------

#Checking for rows with null values
import pyspark.sql.functions as f
from functools import reduce

diamonds.where(reduce(lambda x, y: x | y, (f.col(x).isNull() for x in diamonds.columns))).show()

# COMMAND ----------

#Checking more info about the features
display(diamonds.describe())

# COMMAND ----------

#For our analysis, we do not need the id column _c0, so we can drop it
diamonds_clean1 = diamonds.drop("_c0")
display(diamonds_clean1)

# COMMAND ----------

#Treating the categorical values
#cut - quality of the cut (Fair, Good, Very Good, Premium, Ideal)
dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

user_func =  udf (lambda x: dict.get(x), IntegerType())
diamonds_prep1 = diamonds_clean1.withColumn('cut',user_func(diamonds_clean1.cut))
display(diamonds_prep1)

# COMMAND ----------

# color - diamond colour, from D (best) to J (worst)
dict2 = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
from pyspark.sql.functions import udf
user_func =  udf (lambda x: dict2.get(x), IntegerType())
diamonds_prep2 = diamonds_prep1.withColumn('color',user_func(diamonds_prep1.color))
display(diamonds_prep2)

# COMMAND ----------

# clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
dict3 = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

user_func =  udf (lambda x: dict3.get(x), IntegerType())
diamonds_prep3 = diamonds_prep2.withColumn('clarity',user_func(diamonds_prep1.clarity))
display(diamonds_prep3)

# COMMAND ----------

display(diamonds_prep3.describe())

# COMMAND ----------

#Changing the parameters from string to double
from pyspark.sql.types import DoubleType

ftr_list = ['carat','depth','table','price','x','y','z']
diamonds_prep4=diamonds_prep3
for ftr in ftr_list:     
    diamonds_prep4 = diamonds_prep4.withColumn(ftr, diamonds_prep4[ftr].cast(DoubleType()))
display(diamonds_prep4)

# COMMAND ----------

#Checking null values before running the model
diamonds_prep4.where(reduce(lambda x, y: x | y, (f.col(x).isNull() for x in diamonds_prep4.columns))).show()

# COMMAND ----------

nonFeatureCols = ["price"]
featureCols = [item for item in diamonds_prep4.columns if item not in nonFeatureCols]

# COMMAND ----------

# VectorAssembler Assembles all of these columns into one single vector. To do this, set the input columns and output column. Then that assembler will be used to transform the prepped data to the final dataset.
from pyspark.ml.feature import VectorAssembler

assembler = (VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features"))

finalPrep = assembler.transform(diamonds_prep4)

# COMMAND ----------

finalPrep.printSchema()

# COMMAND ----------

training, test = finalPrep.randomSplit([0.7, 0.3])

#  Going to cache the data to make sure things stay snappy!
training.cache()
test.cache()

print(training.count()) # Why execute count here??
print(test.count())

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lrModel = (LinearRegression()
  .setLabelCol("price")
  .setFeaturesCol("features")
  .setElasticNetParam(0.5))

print("Printing out the model Parameters:")
print("-"*20)
print(lrModel.explainParams())
print("-"*20)

# COMMAND ----------

from pyspark.mllib.evaluation import RegressionMetrics
lrFitted = lrModel.fit(training)

# COMMAND ----------

# MAGIC %md
# MAGIC - We will create the holdout considering a price between +/- 200, which is roughly 5% of the average diamond price.

# COMMAND ----------

holdout = (lrFitted
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "price", 
    """CASE double(round(prediction)) BETWEEN price - 200 AND price + 200 
  WHEN true then 1
  ELSE 0
END as price_within_range"""))

display(holdout)

# COMMAND ----------

display(holdout.selectExpr("sum(price_within_range)/sum(1)*100"))

# COMMAND ----------

# have to do a type conversion for RegressionMetrics
rm = RegressionMetrics(holdout.select("prediction", "price").rdd.map(lambda x:  (x[0], x[1])))

print("MSE: ", rm.meanSquaredError)
print("MAE: ", rm.meanAbsoluteError)
print("RMSE Squared: ", rm.rootMeanSquaredError)
print("R Squared: ", rm.r2)
print("Explained Variance: ", rm.explainedVariance, "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC - The model did not perform so well with a holdout below 20%. We will see if it improves after the pipeline.

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline

rfModel = (RandomForestRegressor()
  .setLabelCol("price")
  .setFeaturesCol("features"))

paramGrid = (ParamGridBuilder()
  .addGrid(rfModel.maxDepth, [5, 10])
  #.addGrid(rfModel.numTrees, [20, 60])
  .build())
# Note, that this parameter grid will take a long time
# to run in the community edition due to limited number
# of workers available! Be patient for it to run!
# If you want it to run faster, remove some of
# the above parameters and it'll speed right up!

stages = [rfModel]

pipeline = Pipeline().setStages(stages)

cv = (CrossValidator() # you can feel free to change the number of folds used in cross validation as well
  .setEstimator(pipeline) # the estimator can also just be an individual model rather than a pipeline
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(RegressionEvaluator().setLabelCol("price")))

pipelineFitted = cv.fit(training)

# COMMAND ----------

print("The Best Parameters:\n--------------------")
print(pipelineFitted.bestModel.stages[0])
pipelineFitted.bestModel.stages[0].extractParamMap()

# COMMAND ----------

pipelineFitted.bestModel

# COMMAND ----------

holdout2 = (pipelineFitted.bestModel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "price", 
    """CASE double(round(prediction)) BETWEEN price - 200 AND price + 200 
  WHEN true then 1
  ELSE 0
END as price_within_range"""))
  
display(holdout2)

# COMMAND ----------

display(holdout2.selectExpr("sum(price_within_range)/sum(1)*100"))

# COMMAND ----------

# MAGIC %md
# MAGIC - We can see that the model improved a good amount after pipeline, with the holdout at 60% and better metrics.

# COMMAND ----------

rm2 = RegressionMetrics(holdout2.select("prediction", "price").rdd.map(lambda x:  (x[0], x[1])))

print("MSE: ", rm2.meanSquaredError)
print("MAE: ", rm2.meanAbsoluteError)
print("RMSE Squared: ", rm2.rootMeanSquaredError)
print("R Squared: ", rm2.r2)
print("Explained Variance: ", rm2.explainedVariance, "\n")
