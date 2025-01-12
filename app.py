from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import jax.numpy as jnp
import jax

app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "*"}})

# Serve the main index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# Catch-all route to serve static files (e.g., CSS, JS, assets)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


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
    app.run(host='0.0.0.0', port=5001)
