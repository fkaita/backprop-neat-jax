import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.example_libraries import optimizers
import jax.random as jrandom

# Define activation functions
def tanh(x): return jax.nn.tanh(x)
def relu(x): return jax.nn.relu(x)
def sigmoid(x): return jax.nn.sigmoid(x)
def gaussian(x): return jnp.exp(-x**2)
def sin(x): return jnp.sin(x)
def cos(x): return jnp.cos(x)
def abs_fn(x): return jnp.abs(x)
def square(x): return x**2
def identity(x): return x  # Default function if node type is not recognized

# JAX-compatible activation function lookup using jax.lax.switch()
@jit
def activation_fn(x, node_type):
    activation_list = [
        sigmoid,   # 3
        tanh,      # 4
        relu,      # 5
        gaussian,  # 6
        sin,       # 7
        cos,       # 8
        abs_fn,    # 9
        square,    # 13
    ]
    index = jnp.clip(node_type - 3, 0, len(activation_list) - 1)  # Ensure index is valid
    return jax.vmap(lambda i, x: jax.lax.switch(i, activation_list, x))(index, x)

class NEATModel:
    def __init__(self, genome_json, learning_rate=0.01):
        self.num_inputs = genome_json["nInput"]
        self.num_outputs = genome_json["nOutput"]
        self.num_nodes = len(genome_json["nodes"])
        self.node_types = jnp.array(genome_json["nodes"])

        # Initialize weight matrix
        self.W = jnp.zeros((self.num_nodes, self.num_nodes))
        
        # Populate weight matrix from genome (now handled as a dict)
        connections = jnp.array(genome_json["connections"])
        genomes = genome_json["genome"]  # List of Dictionary with keys "0", "1", "2"

        for genome in genomes:
            idx = int(genome["0"])  # Connection index
            weight = genome["1"]    # Connection weight
            active = int(genome["2"])  # Connection active status
            if active:
                i, j = int(connections[idx][0]), int(connections[idx][1])
                self.W = self.W.at[i, j].set(weight)

        # Output node indices (last `num_outputs` nodes)
        self.output_nodes = jnp.arange(-self.num_outputs, 0)
        
        # Optimizer
        self.learning_rate = learning_rate
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.W)

        # Store original genome_json for updates
        self.genome_json = genome_json

    def forward(self, inputs):
        """Forward pass of NEAT, handling batch inputs."""
        batch_size = inputs.shape[0]

        # Initialize X_new for batch processing
        X_new = jnp.zeros((batch_size, self.num_nodes))
        X_new = X_new.at[:, :self.num_inputs].set(inputs)  # Set input nodes per batch
        X_new = jnp.dot(X_new, self.W.T)  # Matrix multiplication for batch
        X_new = jax.vmap(lambda x: activation_fn(x, self.node_types))(X_new)

        return jax.nn.sigmoid(X_new[:, self.output_nodes]) 

    def loss(self, W, inputs, targets):
        """ Binary Cross-Entropy Loss """
        preds = self.forward(inputs)
        return -jnp.mean(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))  # BCE loss

    def backward(self, inputs, targets, nCycles=1):
        """ Compute gradients and update weights (without `jit` on self) """
        # loss_grad_fn = jit(grad(self.loss))  # `jit` applied to grad function only
        # grads = loss_grad_fn(self.W, inputs, targets)  # Compute gradients
        nCycles = 1
        for _ in range(nCycles):
            # Random sample for batch training
            batch_indices = jrandom.choice(jrandom.PRNGKey(0), inputs.shape[0], shape=(10,), replace=False)
            batch_inputs = inputs #[batch_indices]
            batch_targets = targets #[batch_indices]
            preds = self.forward(batch_inputs)  # Compute forward pass explicitly
            loss_grad_fn = jit(grad(lambda W: self.loss(W, batch_inputs, batch_targets)))  # Use precomputed preds
            grads = loss_grad_fn(self.W)  # Compute gradients

            self.opt_state = self.opt_update(0, grads, self.opt_state)  # Update optimizer state
            self.W = self.get_params(self.opt_state)  # Apply updated weights

        # Convert updated W into genome format (dictionary keys "0", "1", "2")
        for genome in self.genome_json["genome"]:
            idx = int(genome["0"])  # Connection index
            if int(genome["2"]) == 1:  # If active
                i, j = int(self.genome_json["connections"][idx][0]), int(self.genome_json["connections"][idx][1])
                genome["1"] = float(self.W[i, j])  # Update weight

        # Compute average loss for batch
        avg_error = float(self.loss(self.W, inputs, targets))

        # Format predictions to match JS structure (arrays instead of dicts)
        pred_list = preds.flatten().tolist()  # Convert to a list
        grad_list = grads.flatten().tolist()  # Convert gradients to a list

        formatted_preds = {
            "n": preds.shape[0],  # Number of samples (batch size)
            "d": preds.shape[1],  # Number of outputs per sample
            "w": pred_list,  # Predictions as a flat list
            "dw": grad_list  # Gradients as a flat list (similar to how `dw` works in JS)
        }
        return self.genome_json, avg_error, formatted_preds  # Return updated weights
    
    def batch_backward(self, gene_list, inputs, targets, nCycles=1):
        """ Perform backward pass on multiple genomes in parallel """
        # def process_gene(gene):
        #     model = NEATModel(gene)  # Create NEAT model instance for each genome
        #     return model.backward(inputs, targets, nCycles)

        # batch_process_fn = jax.vmap(process_gene, in_axes=(0,))
        # return batch_process_fn(jnp.array(gene_list))
        results = []
        for gene in gene_list:
            # print(gene)
            model = NEATModel(gene)  # Create NEAT model instance for each genome
            updated_genome, avg_error, predictions = model.backward(inputs, targets, nCycles)
            results.append({
                "updated_genome": updated_genome,
                "avg_error": avg_error,
                "output": predictions
            })

        return results  # Return as a list instead of JAX array


if __name__ == "__main__":
    # Example Usage:
    genome_json = {
        "nodes": [3, 4, 5, 4],  # Node activations (sigmoid, tanh, relu, tanh)
        "connections": [[0, 1], [1, 2], [2, 3]],  # List of edges (from-to)
        "nInput": 2,
        "nOutput": 1,
        "genome": [
            {"0": 0, "1": 0.5, "2": 1},  # (index, weight, active)
            {"0": 1, "1": -0.8, "2": 1},
            {"0": 2, "1": 1.2, "2": 1},
        ],
    }

    # Example batch data
    inputs = jnp.array([[0.2, 0.5], [0.1, 0.4], [0.7, 0.3]])  # Shape: (batch_size, nInput)
    targets = jnp.array([[1.0], [0.0], [1.0]])  # True labels (binary)

    # Initialize NEAT model
    neat = NEATModel(genome_json, learning_rate=0.01)

    # Run forward and backward pass
    updated_genome, avg_error = neat.backward(inputs, targets)

    # Print results
    print("Updated Genome JSON:", updated_genome)
    print("Average Error:", avg_error)