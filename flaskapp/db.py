# TODO: it appears that importing a module that imports pyspark
# from jupyter is problematic. Find out what the underlying is.

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

#def init2(dbpath = "redicommend6.db"):
#    connection = sqlite3.connect(dbpath)
#    dbmap['connection2'] = connection
#    cursor = connection.cursor()
#    dbmap['cursor2'] = cursor

#create_command = """
#CREATE TABLE reddit ( 
#key VARCHAR(30), 
#related VARCHAR(150));"""


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

def get_author_recommended_subs(author):
    subs = []
    author_visited = get_author_visited_subs()[author][0]
    #print author_visited
    author_visited = author_visited.split(',')
    for sub in author_visited:
        try:
            subs += related_subs_from_sql()[sub]['related'].split(',')
        except IndexError, KeyError:
            pass
    recommendations = list(np.random.choice(subs, 20, replace = False))
    return ', '.join(recommendations)
