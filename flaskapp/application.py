import db
import numpy as np
db.init('redicommend10.db')
db.related_subs_from_sql()
db.get_author_visited_subs()

N = 5000
scale = 10
x = list(np.random.random(N))
y = list(np.random.random(N))
size = list(scale * np.random.random(N))
text = N * ['aaaa']
color = ['rgb(%d, %d, %d)' % (np.random.random_integers(255),
    np.random.random_integers(255),
    np.random.random_integers(255)) for _ in xrange(N)]
data = dict(
    x=x,
    y=y,
    mode='markers',
    text = text,
    marker=dict(
        color=color,
        size=size,
    )
)


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
        related = db.related_subs_from_sql()[search_term]['related']
        header = "subs related to r/%s:" % search_term + '\n'
        rest = related.replace(', ', '\n')
    except KeyError:
        rest = '' 
        if search_term:
            header = "%s: no such subreddit" % search_term
        else:
            header = ''
    return header + rest

def process_author_query(author):
    try:
        related = db.get_author_recommended_subs(author)
        header = "recommended subs of u/%s:" % author + '\n'
        rest = related.replace(', ', '\n')
    except KeyError:
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

@app.route('/_get_scatter_data')
def _get_scatter_data():
    data = db.get_svd_projection_plot_data()
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')

