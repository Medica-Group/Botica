from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Looker Connector is Running"

@app.route('/manifest', methods=['GET'])
def manifest():
    return jsonify({
        "name": "Simple Data Connector",
        "version": "1",
        "authType": "NONE",
        "timeout": 10,
        "components": [
            {
                "name": "data",
                "label": "Sample Data",
                "description": "Sample numeric data",
                "schema": {
                    "fields": [
                        {
                            "name": "value",
                            "label": "Value",
                            "dataType": "NUMBER",
                            "semantics": {
                                "conceptType": "METRIC",
                                "semanticType": "NUMBER"
                            }
                        }
                    ]
                }
            }
        ]
    })

@app.route('/data', methods=['GET', 'POST'])
def data():
    return jsonify({
        "schema": [
            {
                "name": "value",
                "label": "Value",
                "dataType": "NUMBER"
            }
        ],
        "rows": [
            {"value": 100},
            {"value": 200},
            {"value": 300}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
