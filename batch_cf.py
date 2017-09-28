from batch_common import *
from flaskapp import db

#json_uri = "s3a://insight-ohoidn/sample3.json"
json_uri = "s3a://insight-ohoidn/sample10M.json"
#json_uri = "s3a://insight-ohoidn/RC_2017-08"

db_name = 'cf.db'

numreddits = 500
user_min_active_subreddits = 5
user_max_active_subreddits = 30

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

tf_ij = gen_frequency_matrix()

db.init(db_name)

def insert_one(subreddit):
    from cf_spark import spark_top_k_subs
    related = spark_top_k_subs(tf_ij, subreddit, subreddit_to_idx, k = 10)
    if related is None:
        return
    val = ', '.join(related[1:])
