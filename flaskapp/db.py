# TODO: it appears that importing a module that imports pyspark
# from jupyter is problematic. Find out what the underlying is.

import pdb
import numpy as np
import sqlite3

dbmap = {'connection': None, 'cursor': None}

def memo(f):
    """
    memoize, using object ids as cache keys
    """
    cache = {}
    def new_f(*args, **kwargs):
        ids = tuple(map(id, args)) + tuple(map(id, kwargs))
        if ids not in cache:
            cache[ids] = f(*args, **kwargs)
        return cache[ids]
    return new_f

def init(dbpath = "redicommend10.db"):
    connection = sqlite3.connect(dbpath)
    dbmap['connection'] = connection
    cursor = connection.cursor()
    dbmap['cursor'] = cursor


def insert_one(subreddit, string):
    sql_command = """INSERT INTO reddit (key, related)
    VALUES ("%s", "%s");""" % (subreddit, string)
    dbmap['cursor'].execute(sql_command)


@memo
def related_subs_from_sql():
    import cPickle
    import numpy as np
    sql_command = """select * from reddit """ 
    cursor = dbmap['cursor']
    cursor.execute(sql_command) 

    result = cursor.fetchall()
    related_subs_dict = {v[0]: v[1:] for v in result}
    for sub, data in related_subs_dict.iteritems():
        arr = cPickle.loads(str(data[2]))
        newdict = dict(zip(['activity', 'related', 'vector'],
            list(data[:2]) + [arr]))
        related_subs_dict[sub] = newdict
    return related_subs_dict

@memo
def get_author_visited_subs():
    sql_get = """select * from authors"""
    cursor = dbmap['cursor']
    cursor.execute(sql_get)
    result = cursor.fetchall()
    author_sub_recommendation_dict = {v[0]: v[1:] for v in result}
    return author_sub_recommendation_dict

def random_color(N):
    color = ['rgb(%d, %d, %d)' % (np.random.random_integers(255),
        np.random.random_integers(255),
        np.random.random_integers(255)) for _ in xrange(N)]
    return color

@memo
def get_svd_projection_plot_data():
    scale = 6e-2
    import numpy as np
    import copy
    d = copy.deepcopy(related_subs_from_sql())
    # TODO: temporary hack, figure out how to include AskReddit
    # without it screwing up the presentation
    try:
        del d['AskReddit']
    except:
        pass
    subs = d.keys()
    vectors = [d[s]['vector'] for s in subs]
    vector_norms = np.array([np.linalg.norm(v) for v in vectors])

    vectors_ndarr = np.array(vectors)
    vn1 = vectors_ndarr[:, -1] / vector_norms
    vn1 -= np.min(vn1)
    vn2 = vectors_ndarr[:, -2] / vector_norms
    vn2 -= np.min(vn2)
    x = map(float, vn1)
    y = map(float, vn2)
    size = [scale * float(np.sqrt(d[s]['activity'])) for s in subs]
    color = random_color(len(subs))

    data = dict(
        x=x,
        y=y,
        mode='markers',
        type='scatter',
        text = subs,
        marker=dict(
            color=color,
            size=size,
        )
    )
    return data


def get_author_recommended_subs(author):
    subs = []
    author_visited = get_author_visited_subs()[author][0]
    author_visited = author_visited.split(',')
    for sub in author_visited:
        try:
            subs += related_subs_from_sql()[sub]['related'].split(',')
        except KeyError:
            pass
    recommendations = list(np.random.choice(subs, 20, replace = False))
    return ', '.join(recommendations)
