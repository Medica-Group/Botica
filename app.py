from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, Looker!'

@app.route('/analyze')
def get_manifest():
    manifest = {
        "name": "Data Analysis Connector",
        "version": "1.0",
        "serviceType": "REST",
        "authType": "NONE",
        "endpoints": [{
            "name": "getData",
            "httpMethod": "POST",
            "url": "/data"
        }],
        "schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension1": {"type": "string"},
                            "metric1": {"type": "number"}
                        }
                    }
                }
            }
        }
    }
    return jsonify(manifest)

@app.route('/data', methods=['POST'])
def get_data():
    return jsonify({
        "data": [
            {"dimension1": "A", "metric1": 100},
            {"dimension1": "B", "metric1": 200}
        ]
    })

if __name__ == '__main__':
    app.run()
