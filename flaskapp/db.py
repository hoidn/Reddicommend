import sqlite3

connection = sqlite3.connect("redicommend2.db")
cursor = connection.cursor()

create_command = """
CREATE TABLE reddit ( 
key VARCHAR(30), 
related VARCHAR(150));"""

#def insert_one(subreddit):
#    related = spark_top_k_subs(tf_ij, subreddit, subreddit_to_idx, k = 5)
#    if related is None:
#        return
#    val = ', '.join(related[1:])
#    sql_command = """INSERT INTO reddit (key, related)
#    VALUES ("%s", "%s");""" % (subreddit, val)
#    cursor.execute(sql_command)

def related_subs_from_sql(sub):
    sql_command = """select related from reddit
    where key='%s'"""  % sub
    cursor.execute(sql_command) 
    return cursor.fetchall()[0][0]
