#Import and create a Spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, skewness, kurtosis
from pyspark.mllib.stat import Statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
spark = SparkSession.builder.appName("Python Spark create RDD")\
		.config("spark.some.config.option", "some-value")\
		.getOrCreate()

#Set path
path = "E:\\ML\\Santander\\"

#Import data
df = spark.read.format('com.databricks.spark.csv')\
					.options(header="true", inferschema ='true')\
					.load(path+"train.csv",header=True)

df.show(2)
df.printSchema()
df.columns


#Defintion
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

#Describe
selected = [s for s in df.columns if 'var_' in s]
print(selected)
df.select(selected).describe().show()
selected = ['var_0','var_1','var_2','var_3','var_4', 'var_5']
describe_pd(df,selected)
describe_pd(df,selected,deciles=True)

#Skewness and kurtosis
df.select(skewness('var_0'),kurtosis('var_0')).show()


#Plot histogram
var = [ 'var_108']
bins = np.arange(0,105,5.0)
df[var].describe().show()
plt.figure(figsize=(10,8))
plt.hist(df_new[var].astype(float),alpha=0.8,histtype='bar',ec='black')

df.dtypes
df[df.var_100 <14.0]

#Correlation matrix
selected = ['target','var_0','var_1','var_2','var_3','var_4', 'var_5']
features = df.select(selected).rdd.map(lambda row: row[0:])
corr_mat = Statistics.corr(features,method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = selected, selected
print(corr_df.to_string())

#Pair plot
sns.pairplot(df.select(selected).toPandas())
plt.show()

df.select("target").describe().show()

