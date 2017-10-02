import db

db.init('redicommend5.db')
db.init2()

# -*- coding: utf-8 -*-
"""
    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

def process_query(search_term):
    # TODO: rewrite db.related_subs_from_sql
    try:
        related = db.related_subs_from_sql(search_term)
        header = "subs related to r/%s:" % search_term + '\n'
        rest = related.replace(', ', '\n')
    except IndexError:
        rest = '' 
        if search_term:
            header = "%s: no such subreddit" % search_term
        else:
            header = ''
    #return '\n' + header + rest
    return header + rest

def process_author_query(author):
    try:
        related = db.get_author_recommended_subs(author)
        header = "recommended subs of u/%s:" % author + '\n'
        rest = related.replace(', ', '\n')
    except IndexError:
        rest = ''
        if author:
            header = "%s: no such user" % author
        else:
            header = ''
    return header + rest

@app.route('/_query_related_subs')
def query_related_subs():
    sub = request.args.get('sub', 0, type=str)
    return jsonify(result=process_query(sub))

@app.route('/_query_author_related_subs')
def query_author_related_subs():
    author = request.args.get('author', 0, type=str)
    return jsonify(result=process_author_query(author))


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')

