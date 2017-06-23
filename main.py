from flask import Flask, jsonify
import requests
import sys
app = Flask(__name__)

DB_URL = sys.argv[1]

def sentiment_analyze(batch, mode):
    return [1] * len(batch)

@app.route('/analyze/<int:uid>')
def analyze(uid):
    r = requests.get(DB_URL + str(uid))
    weibo = r.json()
    batch = []
    for w in weibo:
        batch.append(w['content'])
        batch.extend([[c] for c in w['comments']])
    res = sentiment_analyze(batch, 1)
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
