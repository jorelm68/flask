from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/run_ml', methods=['POST'])
def run_ml():
    data = request.json
    index = data.get('index')
    variable = data.get('variable')
    direction = data.get('direction')

    # Your ML logic here
    # For now, we'll just return the input data
    result = {
        "index": index,
        "variable": variable,
        "direction": direction,
        "result": "ML operation not implemented yet"
    }

    return jsonify(result)

# This is important for Vercel deployment
if __name__ == '__main__':
    app.run()