from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split
from graphframes import GraphFrame
import os
import time

# Update this path to where your Reddit TSV file is stored
PATH = "data/soc-redditHyperlinks-body.tsv"

spark = SparkSession.builder.appName("PageRankWithRedditHyperlinks")\
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.11.0")\
    .config("spark.driver.memory", "4g")\
    .config("spark.executor.memory", "4g")\
    .getOrCreate()

# Create a checkpoint directory to relieve JVM memory pressure during iterations
spark.sparkContext.setCheckpointDir("data/spark-checkpoints")

time_start = time.time()

# 1. Read TSV with header enabled
df = spark.read \
    .option("delimiter", "\t") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(PATH)

# 2. Extract directed edges and map attributes (including signed sentiment and split properties array)
edges = df.select(
    col("SOURCE_SUBREDDIT").alias("src"),
    col("TARGET_SUBREDDIT").alias("dst"),
    col("TIMESTAMP").alias("timestamp"),
    col("LINK_SENTIMENT").alias("sentiment"),
    split(col("PROPERTIES"), ",").alias("properties_vector")
)

# 3. Extract unique subreddit names for vertices
verts_src = edges.select(col("src").alias("id"))
verts_dst = edges.select(col("dst").alias("id"))
verts = verts_src.union(verts_dst).distinct()

# 4. Cache DataFrames before running iterative PageRank loops
verts.cache()
edges.cache()

# 5. Construct Graph and run PageRank
graph = GraphFrame(verts, edges)
results = graph.pageRank(resetProbability=0.15, maxIter=10)

print("--------_------------ DATA ----------------")

edges_num = edges.count()
verts_num = verts.count()

print(f"Number of edges: {edges_num}")
print(f"Number of vertices: {verts_num}")

if os.path.exists(PATH):
    print(f"Size of {PATH}: {os.path.getsize(PATH)} bytes or \n {os.path.getsize(PATH) / (1024 * 1024):.2f} MB")

num_partitions = edges.rdd.getNumPartitions()
print(f"Number of Spark Partitions:    {num_partitions}")
rows_per_partition = edges.rdd.glom().map(len).collect()

print("Rows per partition:")
for i, rows in enumerate(rows_per_partition):
    print(f"  Partition {i}: {rows}")

print(f"Total time: {time.time() - time_start:.2f} seconds")
print("--------_------------ DONE ----------------")

# Display the top 10 most influential subreddits by score
print("\nTop 10 Subreddits by PageRank:")
results.vertices.orderBy(col("pagerank").desc()).show(10, truncate=False)