import linalg
from batch_common import *
import svd

#json_uri = "s3a://insight-ohoidn/sample3.json"
#json_uri = "s3a://insight-ohoidn/sample10M.json"
json_uri = "s3a://insight-ohoidn/RC_2017-08"
hdfs_prefix = "hdfs://ip-10-0-0-4:9000"
numreddits = 5000
user_min_active_subreddits = 8
user_max_active_subreddits = 100
default_partitions = 6

load_and_preprocess(json_uri, numreddits,
                    user_min_active_subreddits = user_min_active_subreddits,
                    user_max_active_subreddits = user_max_active_subreddits)

subreddit_mapper = dict(df_most_active_subreddits(numreddits)\
                        .rdd.map(lambda entry: (entry.ordered_id, entry.subreddit)).sortByKey().collect())
idx_mapper = {v: k for k, v in subreddit_mapper.iteritems()}

def idx_to_subreddit(idx):
    return subreddit_mapper[idx + 1]

def subreddit_to_idx(sub):
    return idx_mapper[sub] - 1

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
