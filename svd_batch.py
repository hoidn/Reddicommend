import linalg
from batch_common import *
import svd
import sqlite3
sc = pyspark.SparkContext.getOrCreate()
sqlContext= pyspark.sql.SQLContext.getOrCreate(sc)


#json_uri = "s3a://insight-ohoidn/sample3.json"
#json_uri = "s3a://insight-ohoidn/sample10M.json"
json_uri = "s3a://insight-ohoidn/RC_2017-08"
hdfs_prefix = "hdfs://ip-10-0-0-4:9000"
numreddits = 50000
user_min_active_subreddits = 5
user_max_active_subreddits = 100
default_partitions = 6
dbname='redicommend10.db'

connection = sqlite3.connect(dbname)
cursor = connection.cursor()

def df_most_active_subreddits(num_subreddit = 50000, npartitions = 18):
    most_active = sqlContext.sql("""
    select * from
        (select *, dense_rank() over (order by activity desc) as ordered_id
        from (select rid, subreddit, sum(activity) as activity
            from occurrences
            group by rid, subreddit))
        where ordered_id<=%d
""" % num_subreddit).persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
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
    where count>=%d and count<=%d""" % (min_subreddits, max_subreddits)).persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
    most_active_users.registerTempTable('most_active_users')
    return most_active_users

def load_and_preprocess(json_uri = json_uri, num_subreddit = numreddits,
        user_min_active_subreddits = user_min_active_subreddits,
        user_max_active_subreddits = user_max_active_subreddits):
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
    """).persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
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
    """).persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
    occurrences_pruned.registerTempTable('occurrences_pruned')
    
    bare_occurrences = sqlContext.sql("""
    select ordered_id, uid, tally
    from occurrences_pruned
    """).persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
    bare_occurrences.registerTempTable('bare_occurrences')
    
def gen_frequency_matrix(npartitions = 18, bias_correction = False):
    # subreddit-activity matrix
    bare_occurrences = sqlContext.sql("""select * from bare_occurrences""").repartition(npartitions)
    rows = bare_occurrences.rdd.map(
            lambda row: (row.ordered_id, (row.uid, row.tally))).sortByKey()
    tf_ij = CoordinateMatrix(rows.map(lambda entry: (entry[0] - 1, entry[1][0], entry[1][1])))
    
    return rows, tf_ij

def get_sparse_subreddit_user_array():
    def idx_to_subreddit(idx):
        return subreddit_mapper[idx + 1]
    
    def subreddit_to_idx(sub):
        return idx_mapper[sub] - 1
    load_and_preprocess(json_uri, numreddits,
                        user_min_active_subreddits = user_min_active_subreddits,
                        user_max_active_subreddits = user_max_active_subreddits)
    
    subreddit_mapper = dict(df_most_active_subreddits(numreddits)\
                            .rdd.map(lambda entry: (entry.ordered_id, entry.subreddit)).sortByKey().collect())
    idx_mapper = {v: k for k, v in subreddit_mapper.iteritems()}
    
    
    subreddit_to_idx.inverse = idx_to_subreddit
    idx_to_subreddit.inverse = subreddit_to_idx
    
    rows, tf_ij = gen_frequency_matrix()
    
    bare_occurrences = sqlContext.sql("""select * from bare_occurrences""").repartition(default_partitions)
    i_sumtally_tuples = bare_occurrences.rdd.map(lambda row: (row.ordered_id, abs(row.tally))).sortByKey()\
    .reduceByKey(add)
    gf_i = CoordinateMatrix(i_sumtally_tuples.map(lambda entry: (entry[0] - 1, 0, entry[1])))
    
    
    nusers = tf_ij.numCols()
    
    p_ij = linalg.coordinate_matrix_elementwise_vector_division(tf_ij, gf_i)
    
    weight_i = linalg.coordinateMatrixElementwise(
        linalg.coordinate_matrix_sumj(
            linalg.coordinateMatrixElementwise(p_ij, lambda elt: -abs(elt) * np.log(abs(elt)) / np.log(nusers))),
        lambda elt: 1. + elt)
    
    log_tf_ij = linalg.coordinateMatrixElementwise(tf_ij, lambda elt: np.log(1 + abs(elt)))
    a_ij = linalg.coordinatematrix_multiply_vector_elementwise(log_tf_ij, weight_i)
    
    rows = tf_ij.entries.map(lambda entry: (entry.i, [(entry.j, entry.value)])).reduceByKey(add).collect()
    flat_coords = np.array(a_ij.entries.map(lambda row: (row.i, row.j, row.value)).collect())
    row_ind, col_ind, data = flat_coords.T
    sparse_arr = svd.csr_matrix((data.astype('float32'), (row_ind, col_ind)))
    return sparse_arr

def get_similarity_matrix(subreddit_user_array, singular_values = 300, normalize = True):
    vals, svc = svd.svd_correlation_matrix(subreddit_user_array, singular_values, normalize=normalize)
    return svc

def create_db():
    sql_command = """
    CREATE TABLE reddit ( 
    key VARCHAR(30), 
    activity INT,
    related VARCHAR(150),
    vector VARCHAR(8000));"""
    cursor.execute(sql_command)

def svd_output_to_db(sim, lowrank_subreddit_array):
    import cPickle
    activity_df = sqlContext.sql("""select activity from most_active""")
    activity = activity_df.select('activity').rdd.flatMap(lambda x: x).collect()

    # rows of the similarity matrix encoded as string
    lowrank_encoded = map(cPickle.dumps, lowrank_subreddit_array)
    
    def one_sub_to_db(sim, subreddit, k = 20):
        print subreddit
        related = [idx_to_subreddit(x) for x in np.argsort(sim[subreddit_to_idx(subreddit),:])[:-k-1:-1]]
        if related is None:
            return
        val = ', '.join(related[1:])
        act = activity[subreddit_to_idx(subreddit)]
        vec = lowrank_encoded[subreddit_to_idx(subreddit)]

        #print (subreddit, act, val, sim)
        sql_command = """INSERT INTO reddit (key, activity, related, vector)
        VALUES (?, ?, ?, ?);"""
        cursor.execute(sql_command, (subreddit, act, val, vec))
        
    for subreddit in idx_mapper.keys():
        one_sub_to_db(sim, subreddit)
    connection.commit()

def empty_table():
    cursor.execute("delete FROM reddit")
    connection.commit()

def get_author_subs():
    """
    Run load_and_preprocess first.
    """
    occurrences_pruned = sqlContext.sql("""
    select *
    from  (SELECT test2.subreddit, author, test2.ordered_id, sum(score) as tally,\
        sum(abs(score)) as activity, dense_rank() over (order by author desc) as uid
        from test2
        where author in (select author from most_active_users)
        group by test2.subreddit, test2.ordered_id, author)
    where tally!=0
    """).persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
    author_subs = occurrences_pruned.rdd.map(lambda row: (row.author, [row.subreddit])).reduceByKey(add)
    list_author_subs = author_subs.collect()
    return list_author_subs

def create_authors_table():
    sql_command = """
    CREATE TABLE authors ( 
    key VARCHAR(30), 
    author VARCHAR(150));"""
    cursor.execute(sql_command)
    connection.commit()

def author_subreddits_to_db(list_author_subs):
    for tup in list_author_subs:
        sql_command = """INSERT INTO authors (key, author)
        VALUES ("%s", "%s");""" % (tup[0], ','.join(tup[1]))
        cursor.execute(sql_command)
    connection.commit()
