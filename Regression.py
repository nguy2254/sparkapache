#Import and create a Spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, skewness, kurtosis
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor,RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Create session
spark = SparkSession.builder.appName("Python Spark Regression example")\
		.config("spark.some.config.option", "some-value")\
		.getOrCreate()

#Set path
path = "E:\\ML\\Spark_data\\"

#Import data
df = spark.read.format('com.databricks.spark.csv')\
					.options(header="true", inferschema ='true')\
					.load(path+"Advertising.csv",header=True)

df.show(2)
df.printSchema()
df.columns

#Describe data
def describe_pd(df_in, columns, deciles = False):
	if deciles:
		percentiles = np.array(range(0,110,10))
	else:
		percentiles = [25,50,75]
	percs = np.transpose([np.percentile(df_in.select(x).collect(),percentiles)\
					   for x in columns])
	percs = pd.DataFrame(percs, columns = columns)
	percs['summary'] = [str(p) + '%' for p in percentiles]
	spark_describe = df_in.describe().toPandas()
	new_df = pd.concat([spark_describe, percs],ignore_index = True)
	new_df = new_df.round(2)
	return new_df[['summary']+columns]

describe_pd(df,df.columns)
df.describe().show()

#Correlation matrix
features = df.select(df.columns).rdd.map(lambda row: row[0:])
corr_mat = Statistics.corr(features,method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = df.columns, df.columns
print(corr_df.to_string())

#Pairplot
sns.pairplot(df.toPandas())
plt.show()

#Convert the data to dense vector
#method 1
def transData1(row):
	return Row(label=row["Sales"], features=Vectors.dense([row["TV"],
													   row["Radio"],
													   row["Newspaper"]]))
#method 2
def transData2(data):
	return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

#Get Dummy function
def get_dummy(df,indexCol, categoricalCols,continousCols, labelCol):
	indexers = [StringIndexer(inputCol=c, outputCol = "{0}_indexed".format(c)) \
			 for c in categoricalCols]
	#default setting: droplast = True
	encoders = [OneHotEncoder(inputCol = indexer.getOutputCol(),
						   outputCol="{0}_encoded".format(indexer.getOutputCol()))\
						   for indexer in indexers]
	assembler = VectorAssembler(inputCols = [encoder.getOutputCol()\
										  for encoder in encoders] + \
				continousCols, outputCol = "features")
	pipeline = Pipeline(stages = indexers + encoders + [assembler])
	model = pipeline.fit(df)
	data  = model.transform(df)
	data = data.withColumn('label',col(labelCol))
	return data.select(indexCol, 'features', 'label')


#Get Dummy function - unsupervisoed
def get_dummy(df,indexCol,categoricalCols,continousCols):
	indexers = [StringIndexer(inputCol=c, outputCol = "{0}_indexed".format(c))\
						   for c in categoricalCols]
	encoders = [OneHotEncoder(inputCol= indexer.getOutputCol(),
						   outputCol = "{0}_encoded".format(indexer.getOutputCol()))
						   for indexer in indexers]
	assembler = VectorAssembler(inputCols = [encoder.getOutputCol()
							for encoder in encoders] + continousCols,\
	outputCol = "features")
	pipeline = Pipeline(stages = indexers + encoders + [assembler])
	model = pipeline.fit(df)
	data  = model.transform(df)
	return data.select(indexCol, 'feature')


#Transform dataset
transformed = transData2(df)
transformed.show(5)

#Index feature
featureIndexer = VectorIndexer(inputCol="features", \
							   outputCol="indexedFeatures",\
							   maxCategories =4).fit(transformed)

data = featureIndexer.transform(transformed)
data.show(5)

#Split data into training and testing set
(trainingData,testData) = transformed.randomSplit([0.6,0.4])

trainingData.show(5)
testData.show(5)

##########################################################################
#Linear Regression
#Fit regression
lr = LinearRegression()

#Chain index
pipeline = Pipeline(stages=[featureIndexer, lr])
model = pipeline.fit(trainingData)


#Model Summary
def modelsummary(model, printMSE=True):
	print ("Note: the last rows are the information for Intercept")
	print ("##------------------------------------------- ")
	print ("##   Estimate | Std.Error |t Values| P-Values ")
	coef = np.append(list(model.coefficients), model.intercept)
	Summary=model.summary
	for i in range(len(Summary.pValues)):
		print ("##",'{:10.6f}'.format(coef[i]),\
		 '{:10.6f}'.format(Summary.coefficientStandardErrors[i]),\
		 '{:8.3f}'.format(Summary.tValues[i]),\
		 '{:10.6f}'.format(Summary.pValues[i]))
	print("##---------------------------------------------------")
	if printMSE==True:
		print("##Mean Squared Error: % .6f" \
			% Summary.meanSquaredError, ", RMSE: % .6f"\
			% Summary.rootMeanSquaredError )
		print("##Muliple R-squared: %f" % Summary.r2, \
			   "total iterations: %i" % Summary.totalIterations)

modelsummary(model.stages[-1])


#Make prediction
predictions = model.transform(testData)
predictions.show(5)
predictions.select("features","label","prediction").show(5)


#Evaluation
evaluator = RegressionEvaluator(labelCol = "label",
								predictionCol = "prediction",
								metricName = "r2")

r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)


#Plot
predictions_df = predictions.toPandas()
predictions_df["error"] =predictions_df["prediction"] -predictions_df["label"]
print(predictions_df.head(2))
sns.scatterplot(predictions_df.label,predictions_df.prediction)
plt.show()
sns.boxplot(predictions_df.error)
plt.show()


##########################################################################
#Generalized Linear Regression
#Fit regression
glr = GeneralizedLinearRegression(family = "gaussian", link="identity",\
								  maxIter=10, regParam=0.3)

#Chain index
pipeline = Pipeline(stages=[featureIndexer, glr])
model = pipeline.fit(trainingData)

modelsummary(model.stages[-1],printMSE=False)


#Make prediction
predictions = model.transform(testData)
predictions.show(5)
predictions.select("features","label","prediction").show(5)


#Evaluation
evaluator = RegressionEvaluator(labelCol = "label",
								predictionCol = "prediction",
								metricName = "r2")

r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)


#Plot
predictions_df = predictions.toPandas()
predictions_df["error"] =predictions_df["prediction"] -predictions_df["label"]
print(predictions_df.head(2))
sns.scatterplot(predictions_df.label,predictions_df.prediction)
plt.show()
sns.boxplot(predictions_df.error)
plt.show()

##########################################################################
#Eecision Tree Regression
#Fit regression
dt = DecisionTreeRegressor(featuresCol = "indexedFeatures")

#Chain index
pipeline = Pipeline(stages=[featureIndexer, dt])
model = pipeline.fit(trainingData)

#Make prediction
predictions = model.transform(testData)
predictions.show(5)
predictions.select("features","label","prediction").show(5)

#Evaluation
evaluator = RegressionEvaluator(labelCol = "label",
								predictionCol = "prediction",
								metricName = "r2")

r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)

#Plot
predictions_df = predictions.toPandas()
predictions_df["error"] =predictions_df["prediction"] -predictions_df["label"]
print(predictions_df.head(2))
sns.scatterplot(predictions_df.label,predictions_df.prediction)
plt.show()
sns.boxplot(predictions_df.error)
plt.show()


##########################################################################
#Random Forest Regression
#Fit regression
rf = RandomForestRegressor(featuresCol = "indexedFeatures")

#Chain index
pipeline = Pipeline(stages=[featureIndexer, rf])
model = pipeline.fit(trainingData)

#Make prediction
predictions = model.transform(testData)
predictions.show(5)
predictions.select("features","label","prediction").show(5)

#Evaluation
evaluator = RegressionEvaluator(labelCol = "label",
								predictionCol = "prediction",
								metricName = "r2")

r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)

#Plot
predictions_df = predictions.toPandas()
predictions_df["error"] =predictions_df["prediction"] -predictions_df["label"]
print(predictions_df.head(2))
sns.scatterplot(predictions_df.label,predictions_df.prediction)
plt.show()
sns.boxplot(predictions_df.error)
plt.show()

#Feature importances
model.stages[-1].featureImportances
model.stages[-1].trees


##########################################################################
#Gradient Boosted Regression
#Fit regression
gb = GBTRegressor()

#Chain index
pipeline = Pipeline(stages=[featureIndexer, gb])
model = pipeline.fit(trainingData)

#Make prediction
predictions = model.transform(testData)
predictions.show(5)
predictions.select("features","label","prediction").show(5)

#Evaluation
evaluator = RegressionEvaluator(labelCol = "label",
								predictionCol = "prediction",
								metricName = "r2")

r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)

#Plot
predictions_df = predictions.toPandas()
predictions_df["error"] =predictions_df["prediction"] -predictions_df["label"]
print(predictions_df.head(2))
sns.scatterplot(predictions_df.label,predictions_df.prediction)
plt.show()
sns.boxplot(predictions_df.error)
plt.show()

#Feature importances
model.stages[-1].featureImportances
model.stages[-1].trees
