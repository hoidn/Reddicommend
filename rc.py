#!/usr/bin/env python
import argparse
import sqlite3

connection = sqlite3.connect("redicommend2.db")
cursor = connection.cursor()

def related_subs_from_sql(sub):
    sql_command = """select related from reddit
    where key='%s'"""  % sub
    cursor.execute(sql_command) 
    return cursor.fetchall()[0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subreddit', type = str)

    args = parser.parse_args()
    sub  = args.subreddit
    print '----------------------------------'
    try:
        related = related_subs_from_sql(sub)
        print 'subs related to',  sub, 'include:'
        print related
    except IndexError:
        print 'sub not found:', sub
    print '----------------------------------'

