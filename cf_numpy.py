import numpy as np

state = {'item_similarity': None}

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=3, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    print 'sim', sim
    diag = np.diagonal(sim)
    print 'diag', diag
    norms = np.array([np.sqrt(diag)])
    print norms
    return (sim / norms / norms.T)

test_activity_array = np.array([[ 0.47364486,  0.61937755,  0.00844025,  0.65533834,  0.77267817, 0.37602224], [ 0.13680721,  0.6778712 ,  0.47219011,  0.10933562,  0.66098062, 0.93689595], [ 0.85607774,  0.56905013,  0.92456913,  0.15854798,  0.84589546, 0.22489419], [ 0.64035207,  0.90656327,  0.42716786,  0.64355359,  0.4160582 , 0.30400444], [ 0.78897855,  0.06087785,  0.42894637,  0.99971801,  0.63067103, 0.19627554]])


##def test_fast_similarity():
#array([[ 1.        ,  0.75002058,  0.73386657,  0.89499209,  0.81681218],
#       [ 0.75002058,  1.        ,  0.73751518,  0.74443305,  0.49754955],
#       [ 0.73386657,  0.73751518,  1.        ,  0.83284528,  0.74795564],
#       [ 0.89499209,  0.74443305,  0.83284528,  1.        ,  0.78808545],
#       [ 0.81681218,  0.49754955,  0.74795564,  0.78808545,  1.        ]])


def init(ndmat, subreddit_mapper):
    epsilon = 1e-9
    ndmat = ndmat.T
    ndmat = ndmat[1:, 1:]

    # TODO: play with this
    ndmat /= (epsilon + np.std(ndmat, axis = 0))
    ndmat /= (epsilon + np.std(ndmat.T, axis = 1).T)

    #train, test = train_test_split(ndmat)
    state['item_similarity_full'] = fast_similarity(ndmat, kind = 'item')
    #state['item_similarity_sampled'] = fast_similarity(train, kind = 'item')
    state['subreddit_mapper'] = subreddit_mapper
    state['idx_mapper'] = {v: k for k, v in subreddit_mapper.iteritems()}


def idx_to_subreddit(idx):
    return state['subreddit_mapper'][idx + 1]

def subreddit_to_idx(sub):
    return state['idx_mapper'][sub] - 1

def top_k_subs(movie_idx, k=6):
    return [idx_to_subreddit(x) for x in np.argsort(state['item_similarity_full'][movie_idx,:])[:-k-1:-1]]

def related_subs(sub_name):
    return top_k_subs(subreddit_to_idx(sub_name), k = 10)
