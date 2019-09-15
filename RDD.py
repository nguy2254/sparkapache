#Import and create a Spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import pandas as pd

spark = SparkSession.builder.appName("Python Spark create RDD")\
		.config("spark.some.config.option", "some-value")\
		.getOrCreate()

#Set path
path = "E:\\ML\\IEEE-Fraud\\"

#Create a RDD
df = spark.sparkContext.parallelize([(1,2,3,'a b c'),\
									 (4,5,6,'d e f'),\
									 (7,8,9,'g h i')])\
						.toDF(['col1','col2','col3','col4'])

df.show()
#Rename column
df= df.toDF('a','b','c', 'd')
mapping = {'c':"Newspaper",'d':"Sales"}
new_names = [mapping.get(col,col) for col in df.columns]
df= df.toDF(*new_names)
#Rename one column
df.withColumnRenamed('Newspaper','Paper').show()

#Create another RDD
myData = spark.sparkContext.parallelize([(1,2),(3,4),(5,6),(7,8),(9,10)])

myData.collect()

#Use createDataFrame() function
Employee = spark.createDataFrame([('1','Joe',"70000",'1'),\
								  ('2','Henry',"80000","2"),\
								  ('3','Sam',"60000","2"),\
								  ('4','Max',"90000","1")],\
									["ID","Name","Salary","DepartmentID"])

Employee.show()

#Load CSV
Fraud = spark.read.format('com.databricks.spark.csv')\
					.options(header="true", inferschema ='true')\
					.load(path+"train_transaction.csv",header=True)

Fraud.show(5)
Fraud.columns
Fraud.dtypes
Fraud.fillna(0).show()
Fraud.printSchema()

#Create a new column
Fraud.withColumn( 'V338_norm', Fraud.V338/Fraud.groupBy()\
					 .agg(F.sum('V338')).collect()[0][0])

#Create DataFrame
my_list = [['a',1,2],['b',2,3],['c',3,4]]
col_name = ['A',"B","C"]

spark.createDataFrame(my_list,col_name).show()

#From Dict
dict = {'A': [0,1,0], 'B':[1,0,1], 'C': [1,0,0]}

spark.createDataFrame(np.array(list(dict.values())).T.tolist(),list(dict.keys())).show()


#Join
leftp = pd.DataFrame({'A':['A0',"A1","A2","A3"],\
					  'B':["B0","B1","B2","B3"],\
					  'C':["C0","C1","C2","C3"],\
					  'D':["D0","D1","D2","D3"]},\
					index = [0,1,2,3])


rightp = pd.DataFrame({'A': ['A0', 'A1', 'A6', 'A7'],\
					   'F': ['B4', 'B5', 'B6', 'B7'],\
					   'G': ['C4', 'C5', 'C6', 'C7'],\
					   'H': ['D4', 'D5', 'D6', 'D7']},\
					index=[4, 5, 6, 7])

lefts = spark.createDataFrame(leftp)
rights = spark.createDataFrame(rightp)

lefts.join(rights,on='A', how='left').orderBy('A',ascending=True).show()
lefts.join(rights,on='A',how = 'right').orderBy('A', ascending=False).show()
lefts.join(rights,on='A', how='full').orderBy('A', ascending=True).show()

#Concat columns
my_list = [('a',2,3),('b',5,6),('c',8,9),('a',2,4),('b',5,6),('c',8,9)]
col_name = ['col1','col2','col3']
ds = spark.createDataFrame(my_list, schema= col_name)
ds.withColumn('concat',F.concat('col1','col2')).show()
