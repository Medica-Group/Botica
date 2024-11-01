from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, Looker!'

@app.route('/analyze', methods=['GET'])
def manifest():
    return jsonify({
        "name": "Simple Connector",
        "version": "1",
        "authType": "NONE",
        "schema": {
            "fields": [
                {
                    "name": "value",
                    "dataType": "NUMBER"
                }
            ]
        }
    })

if __name__ == '__main__':
    app.run()
