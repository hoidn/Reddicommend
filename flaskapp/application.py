import db

# -*- coding: utf-8 -*-
"""
    jQuery Example
    ~~~~~~~~~~~~~~

    A simple application that shows how Flask and jQuery get along.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

def process_query(search_term):
    # TODO: rewrite db.related_subs_from_sql
    try:
        related = db.related_subs_from_sql(search_term)
        header = "subs related to %s:" % search_term + '\n'
        rest = related.replace(', ', '\n')
    except IndexError:
        rest = '' 
        if search_term:
            header = "%s: no such subreddit" % search_term
        else:
            header = ''
    return '\n' + header + rest

@app.route('/_query_related_subs')
def query_related_subs():
    sub = request.args.get('sub', 0, type=str)
    return jsonify(result=process_query(sub))


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
