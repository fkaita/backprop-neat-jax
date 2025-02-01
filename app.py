from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import jax
import jax.numpy as jnp

import numpy as np
from neat import NEATModel
import json
import traceback

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
    # print("Received JSON:", data)

    genome_json = data.get("genome")
    learning_rate = data.get("learning_rate", 0.01)

    # Extract shared inputs and targets (same for all genomes)
    # inputs_w = list(data["inputs"]["w"].values())
    # targets_w = list(data["targets"]["w"].values())

    # inputs = jnp.array(inputs_w).reshape((data["inputs"]["n"], data["inputs"]["d"]))
    # targets = jnp.array(targets_w).reshape((data["targets"]["n"], data["targets"]["d"]))

    if isinstance(genome_json, str):
        try:
            genome_json = json.loads(genome_json)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid genome format, could not parse JSON"}), 400

    if not isinstance(genome_json, dict):
        return jsonify({"error": "Invalid genome format, expected a JSON object"}), 400

    
    neat_model = NEATModel(genome_json, inputs, targets, learning_rate=learning_rate)  # Initialize the model
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
    # """Perform backward pass (training step) and return updated genome"""
    global neat_model
    if neat_model is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    try:
        data = request.json
        # print("Received JSON:", data)
        inputs_w = list(data["inputs"]["w"].values())
        targets_w = list(data["targets"]["w"].values())
        inputs = jnp.array(inputs_w).reshape((data["inputs"]["n"], data["inputs"]["d"]))
        targets = jnp.array(targets_w).reshape((data["targets"]["n"], data["targets"]["d"]))
        nCycles = data.get("nCycles", 1)  # Get training cycles (default = 1)

        updated_genome, avg_error, predictions = neat_model.backward(inputs, targets, nCycles)
        return jsonify({
            "updated_genome": updated_genome,
            "avg_error": avg_error,
            "output": {
                "n": predictions["n"],  # Number of samples
                "d": predictions["d"],  # Number of output dimensions
                "w": predictions["w"],  # Predictions as Float32Array
                "dw": predictions["dw"]  # Gradients as Float32Array
            }
        })
    except Exception as e:
        print("Exception Traceback:", traceback.format_exc())  # Log full traceback
        return jsonify({"error": str(e)}), 500
    

@app.route('/batch_backward', methods=['POST'])
def batch_backward():
    """Perform backward pass on multiple genomes in parallel and return updated genomes."""
    global neat_model
    try:
        data = request.json
        # print(data)
        gene_list = data["genes"]  # List of genome JSON objects
        nCycles = data.get("nCycles", 1)  # Training cycles (default = 1)
        if not gene_list:
            return jsonify([]) # Return empty list
        
        # Clear old memory
        jax.clear_caches()
        
        print("got data")
        
        # Initialize the NEAT model using the first genome in the list
        json_gene_list = []
        for gene in gene_list:
            try:
                gene = json.loads(gene)
                json_gene_list.append(gene)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid genome format, could not parse JSON"}), 400
    
        if neat_model is None:
            neat_model = NEATModel(json_gene_list)

        # Extract shared inputs and targets (same for all genomes)
        inputs_w = np.array(list(data["inputs"]["w"].values()), dtype=np.float16)
        targets_w = np.array(list(data["targets"]["w"].values()), dtype=np.float16)

        # Reshape and add bias
        inputs = jnp.array(inputs_w).reshape((data["inputs"]["n"], data["inputs"]["d"]))
        bias = jnp.ones((data["inputs"]["n"], 1))
        inputs_with_bias = jnp.concatenate([inputs, bias], axis=1)
        targets = jnp.array(targets_w)

        print(f"number of gen is {len(gene_list)}")
        print(f"Input is {inputs_with_bias.shape}", f"target is {targets.shape}")
        print(f"first gene is {gene_list[0]}")

        # Use JAX for memory efficiency
        inputs = jax.device_put(inputs)
        targets = jax.device_put(targets)

        # @jax.jit
        # def batched_train(neat_model, inputs, targets, nCycles):
        #     return neat_modßel.train(inputs, targets, nCycles)

        # Call batch_backward function from NEATModel
        print("started model")
        results = neat_model.train(inputs_with_bias, targets, nCycles)
        # results = batched_train(neat_model, inputs, targets, nCycles)
        print("finished model")

        # Free up GPU memory
        jax.clear_caches()

        return jsonify(results)  # Convert JAX arrays to lists for JSON response

    except Exception as e:
        print("Exception Traceback:", traceback.format_exc())  # Log full traceback
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


# TODO 
# Circle when processing
# Fintess is not correct number
# Back prop doesnt change result