from flask import Flask, request, jsonify, send_file, send_from_directory
import jax.numpy as jnp
import jax

app = Flask(__name__)

@app.route("/")
def index():
    return send_file( "index.html")

# Serve CSS files
@app.route('/css/<path:filename>')
def css(filename):
    return send_from_directory('css', filename)

# Serve JS or other files from the lib directory
@app.route('/lib/<path:filename>')
def lib(filename):
    return send_from_directory('lib', filename)

# Serve JS files or other assets from the ml directory
@app.route('/ml/<path:filename>')
def ml(filename):
    return send_from_directory('ml', filename)

@app.route('/update-weights', methods=['POST'])
def update_weights():
    data = request.json
    weights = data['weights']
    inputs = jnp.array(data['inputs'])
    targets = jnp.array(data['targets'])

    updated_weights = update_weights(weights, inputs, targets)

    return jsonify({'updated_weights': updated_weights})

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

def update_weights(weights, inputs, targets, learning_rate=0.01):
    """
    Simulates weight optimization using JAX.
    Args:
        weights (list of floats): Current weights sent from the frontend.
        inputs (jax.numpy.array): Input data for training.
        targets (jax.numpy.array): Target data for training.
        learning_rate (float): Optimization step size.

    Returns:
        list of floats: Updated weights.
    """
    # Convert weights to JAX array
    weights_jax = jnp.array(weights)

    # Example loss function: Mean Squared Error
    def loss_fn(w):
        predictions = jnp.dot(inputs, w)  # Example linear prediction
        return jnp.mean((predictions - targets) ** 2)

    # Compute gradients
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss, gradients = loss_grad_fn(weights_jax)

    # Update weights
    updated_weights = weights_jax - learning_rate * gradients
    return updated_weights.tolist()


if __name__ == '__main__':
    app.run(debug=True)
