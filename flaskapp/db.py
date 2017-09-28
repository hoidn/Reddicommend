import sqlite3

dbmap = {'connection': None, 'cursor': None}

def init(dbpath = "redicommend2.db"):
    connection = sqlite3.connect(dbpath)
    dbmap['connection'] = connection
    cursor = connection.cursor()
    dbmap['cursor'] = cursor

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
