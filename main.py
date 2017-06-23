from flask import Flask, jsonify
from analyze import analyze
import requests
import sys
app = Flask(__name__)

DB_URL = sys.argv[1]

@app.route('/analyze/<int:uid>')
def analyze_endpoint(uid):
    r = requests.get(DB_URL + str(uid))
    weibo = r.json()
    batch = []
    for w in weibo:
        batch.append(w['content'])
        batch.extend([[c] for c in w['comments']])
    res = [float(r) for r in analyze(batch, 0)]
    for w in weibo:
        w['content_sentiment'] = res[0]
        res = res[1:]
        l = len(w['comments'])
        w['comments_sentiment'] = res[0:l]
        res = res[l:]
    response = jsonify(weibo)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
