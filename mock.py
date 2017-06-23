from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/<int:uid>')
def mock(uid):
    return jsonify([
        {
            'content': ["a"],
            'comments': ["a1", "a2"],
            'time': 0
        },
        {
            'content': ["b"],
            'comments': ["b1", "b2"],
            'time': 1
        }
    ])

if __name__ == '__main__':
    app.run(port=4000) 
