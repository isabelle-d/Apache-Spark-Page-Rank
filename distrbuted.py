from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame
import os
import time

PATH = "data/soc-LiveJournal1-250mb.txt"
NUM_PARTITIONS = 15
CLUSTER = "spark://Isabelles-MacBook-Air.local:7077"

spark = SparkSession.builder.appName("PageRankWithGraphFrames")\
    .master(CLUSTER)\
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.11.0")\
    .config("spark.executor.memory", "512mb")\
    .config("spark.sql.shuffle.partitions", str(NUM_PARTITIONS))\
    .config("spark.default.parallelism", str(NUM_PARTITIONS))\
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .getOrCreate()

time_start = time.time()
df = spark.read.option("delimiter", "\t")\
.option("comment", "#")\
.csv(PATH)

edges = df.select(col("_c0").alias("src"),col("_c1").alias("dst")).repartition(NUM_PARTITIONS)

verts_src = edges.select(col("src").alias("id"))
verts_dst = edges.select(col("dst").alias("id"))
verts = verts_src.union(verts_dst).distinct()

edges.cache()
verts.cache()
graph = GraphFrame(verts, edges)


results = graph.pageRank(resetProbability=0.15, maxIter=2)
print("--------_------------ DATA ----------------")

edges_num = edges.count()
verts_num = verts.count()

print(f"Number of edges (unique): {edges_num}")
print(f"Number of vertices (unique): {verts_num}")

if(os.path.exists(PATH)):
    print(f"size of {PATH}: {os.path.getsize(PATH)} bytes or \n {os.path.getsize(PATH) / (1024 * 1024):.2f} MB")

num_partitions = edges.rdd.getNumPartitions()
print(f"Number of Spark Partitions:    {num_partitions}")


print(f"Total time: {time.time() - time_start:.2f} seconds")
print("--------_------------ DONE ----------------")