import pyspark
import numpy as np
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry
from pyspark.sql.types import *
from operator import  add
from pyspark.mllib.linalg import Vectors, ArrayType

minimal_fields = [ 
          StructField("author", StringType(), True),
          StructField("score", LongType(), True),
          StructField("subreddit", StringType(), True)]

def df_most_active_subreddits(num_subreddit = 1000, npartitions = 18):
    most_active = sqlContext.sql("""
    select * from
        (select *, dense_rank() over (order by activity desc) as ordered_id
        from (select rid, subreddit, sum(activity) as activity
            from occurrences
            group by rid, subreddit))
        where ordered_id<=%d
""" % num_subreddit).persist(StorageLevel.MEMORY_AND_DISK_SER)
    most_active.registerTempTable('most_active')
    return most_active.repartition(npartitions)

def df_valid_users(min_subreddits = 2, max_subreddits = 20):
    """
    filter users by the number of subreddits they've posted in, among the above-defined most active subreddits
    """
    most_active_users = sqlContext.sql("""
    select * from
        (select author, count(subreddit) as count
        from 
            (select * from occurrences
            where subreddit in (select subreddit from most_active))
        group by author
        order by count desc)
    where count>=%d and count<=%d""" % (min_subreddits, max_subreddits)).persist(StorageLevel.MEMORY_AND_DISK_SER)
    most_active_users.registerTempTable('most_active_users')
    return most_active_users

def load_and_preprocess(json_uri, num_subreddit, user_min_active_subreddits = 4, user_max_active_subreddits = 20):
    """
    Load json and do preprocessing via some SQL queries
    """
    sj = sqlContext.read.json(json_uri, StructType(minimal_fields))
    sj.registerTempTable('test')
    
    occurrences = sqlContext.sql("""
    select *, dense_rank() over (order by subreddit desc) as rid 
    from  (SELECT subreddit, author, sum(sign(score)) as tally,\
        count(score) as activity, dense_rank() over (order by author desc) as uid
    from test
    group by subreddit, author)
    where tally!=0
    """).persist(StorageLevel.MEMORY_AND_DISK_SER)
    occurrences.registerTempTable('occurrences')
        
    df_most_active_subreddits(num_subreddit)
    df_valid_users(user_min_active_subreddits, user_max_active_subreddits)
    
    test2 = sqlContext.sql("""
    select test.author, test.score, test.subreddit, most_active.ordered_id as ordered_id
    from test
    inner join most_active on most_active.subreddit=test.subreddit""")
    test2.registerTempTable('test2')
    
    occurrences_pruned = sqlContext.sql("""
    select *
    from  (SELECT test2.subreddit, author, test2.ordered_id, sum(score) as tally,\
        sum(abs(score)) as activity, dense_rank() over (order by author desc) as uid
        from test2
        where author in (select author from most_active_users)
        group by test2.subreddit, test2.ordered_id, author)
    where tally!=0
    """).persist(StorageLevel.MEMORY_AND_DISK_SER)
    occurrences_pruned.registerTempTable('occurrences_pruned')
    
    bare_occurrences = sqlContext.sql("""
    select ordered_id, uid, tally
    from occurrences_pruned
    """).persist(StorageLevel.MEMORY_AND_DISK_SER)
    bare_occurrences.registerTempTable('bare_occurrences')
    
def gen_frequency_matrix(npartitions = 18, bias_correction = False):
    # subreddit-activity matrix
    bare_occurrences = sqlContext.sql("""select * from bare_occurrences""").repartition(npartitions)
    tf_ij = CoordinateMatrix(bare_occurrences.rdd.map(
            lambda row: (row.ordered_id, (row.uid, row.tally)))\
        .sortByKey().map(lambda entry: (entry[0] - 1, entry[1][0], entry[1][1])))
    
    return tf_ij
