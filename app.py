from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Looker Connector is Running"

@app.route('/manifest')
def manifest():
    return jsonify({
        "name": "Simple Data Connector",
        "version": "1",
        "authType": "NONE",
        "schema": [
            {
                "name": "value",
                "dataType": "NUMBER",
                "semantics": {
                    "conceptType": "METRIC"
                }
            }
        ]
    })

@app.route('/data')
def data():
    return jsonify({
        "values": [100, 200, 300]
    })

if __name__ == '__main__':
    app.run(debug=True)
