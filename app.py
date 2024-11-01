from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, Looker!'

@app.route('/analyze')
def get_manifest():
    manifest = {
        "name": "Data Analysis API",
        "organizationId": "your-org",
        "apiVersion": "v1",
        "components": [
            {
                "name": "analyze",
                "label": "Data Analysis",
                "description": "Analyze data using Python",
                "authentication": {
                    "type": "NONE"
                },
                "endpoints": [
                    {
                        "name": "getData",
                        "label": "Get Data",
                        "description": "Get analyzed data",
                        "httpMethod": "POST",
                        "uri": "/analyze",
                        "response": {
                            "supportedTypes": ["JSON"]
                        }
                    }
                ],
                "schema": {
                    "fields": [
                        {
                            "name": "value",
                            "label": "Value",
                            "dataType": "NUMBER"
                        }
                    ]
                }
            }
        ]
    }
    return jsonify(manifest)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Simple response for testing
    return jsonify({
        "status": "success",
        "data": [{"value": 123}]
    })

if __name__ == '__main__':
    app.run()
