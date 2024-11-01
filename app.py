from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

@app.route('/')
def hello():
    return 'Hello, Looker!'

@app.route('/analyze', methods=['GET'])
def get_manifest():
    manifest = {
        "configParams": [],
        "dateRanges": [],
        "getAuthType": {
            "type": "NONE"
        },
        "getConfig": {
            "configParams": []
        },
        "getSchema": {
            "schema": [
                {
                    "name": "value",
                    "label": "Value",
                    "dataType": "NUMBER",
                    "semantics": {
                        "conceptType": "METRIC"
                    }
                }
            ]
        },
        "getData": {
            "data": [
                {
                    "values": [100]
                },
                {
                    "values": [200]
                }
            ]
        }
    }
    
    # Add CORS headers
    response = jsonify(manifest)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route('/analyze', methods=['POST'])
def analyze():
    return jsonify({
        "data": [
            {"value": 100},
            {"value": 200}
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
