import numpy as np
import sqlite3

dbmap = {'connection': None, 'cursor': None}

def init(dbpath = "redicommend2.db"):
    connection = sqlite3.connect(dbpath)
    dbmap['connection'] = connection
    cursor = connection.cursor()
    dbmap['cursor'] = cursor

def init2(dbpath = "redicommend6.db"):
    connection = sqlite3.connect(dbpath)
    dbmap['connection2'] = connection
    cursor = connection.cursor()
    dbmap['cursor2'] = cursor

create_command = """
CREATE TABLE reddit ( 
key VARCHAR(30), 
related VARCHAR(150));"""


def insert_one(subreddit, string):
    sql_command = """INSERT INTO reddit (key, related)
    VALUES ("%s", "%s");""" % (subreddit, string)
    dbmap['cursor'].execute(sql_command)

def related_subs_from_sql(sub):
    sql_command = """select related from reddit
    where key='%s'"""  % sub
    cursor = dbmap['cursor']
    cursor.execute(sql_command) 
    return cursor.fetchall()[0][0]

def get_author_visited_subs(author):
    sql_get = """select related from authors
where key='%s'""" % author
    cursor = dbmap['cursor2']
    cursor.execute(sql_get)
    result = cursor.fetchall()
    return result[0][0]

def get_author_recommended_subs(author):
    subs = []
    author_visited = get_author_visited_subs(author)
    author_visited = author_visited.split(',')
    for sub in author_visited:
        try:
            subs += related_subs_from_sql(sub).split(',')
        except IndexError:
            pass
    recommendations = list(np.random.choice(subs, 20, replace = False))
    return ', '.join(recommendations)

