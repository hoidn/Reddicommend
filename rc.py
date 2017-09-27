#!/usr/bin/env python
import argparse
import sqlite3
import db

connection = sqlite3.connect("redicommend2.db")
cursor = connection.cursor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subreddit', type = str)

    args = parser.parse_args()
    sub  = args.subreddit
    print '----------------------------------'
    try:
        related = db.related_subs_from_sql(sub)
        print 'subs related to',  sub, 'include:'
        print related
    except IndexError:
        print 'sub not found:', sub
    print '----------------------------------'

