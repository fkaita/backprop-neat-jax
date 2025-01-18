from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import jax.numpy as jnp
from neat import NEATModel

app = Flask(__name__)

# Initialize global model instance
neat_model = None

CORS(app, resources={r"/*": {"origins": "*"}})

# Serve the main index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Catch-all route to serve static files (e.g., CSS, JS, assets)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize NEAT model with a genome from the request"""
    global neat_model
    data = request.json  # Get JSON payload
    genome_json = data.get("genome")

    if not genome_json:
        return jsonify({"error": "Missing genome data"}), 400

    neat_model = NEATModel(genome_json)  # Initialize the model
    return jsonify({"message": "NEAT model initialized successfully"})


@app.route('/forward', methods=['POST'])
def forward():
    """Perform forward pass and return predictions"""
    global neat_model
    if neat_model is None:
        return jsonify({"error": "Model not initialized"}), 400

    data = request.json
    inputs = jnp.array(data.get("inputs"))

    try:
        outputs = neat_model.forward(inputs)
        return jsonify({"outputs": outputs.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/backward', methods=['POST'])
def backward():
    """Perform backward pass (training step) and return updated genome"""
    global neat_model
    if neat_model is None:
        return jsonify({"error": "Model not initialized"}), 400

    data = request.json
    inputs = jnp.array(data.get("inputs"))
    targets = jnp.array(data.get("targets"))

    try:
        updated_genome, avg_error = neat_model.backward(inputs, targets)
        return jsonify({
            "updated_genome": updated_genome,
            "avg_error": avg_error
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
