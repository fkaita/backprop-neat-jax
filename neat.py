import jax
import jax.numpy as jnp
import jax.random as random
import optax
from collections import deque, defaultdict

import numpy as np


key = random.PRNGKey(0)

# Kahn's algorithm Sorting
def sort_graph(edges, num_nodes, start_nodes):
    adj_list = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes

    for u, v in edges:
        adj_list[u].append(v)
        in_degree[v] += 1

    queue = deque(start_nodes)
    steps = []
    while queue:
        layer_nodes = list(queue)
        queue.clear()  # empty the queue so we can fill it with next layer
        steps.append(layer_nodes)
        
        # For each node in this layer, reduce the in-degree of its children.
        for node in layer_nodes:
            for child in adj_list[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    return steps

def compress(genome):
    nodes = genome['nodes']
    conns = genome['connections']
    small_nodes = []
    small_conns = []
    max_conn_id = max(set(gen["0"] for gen in genome['genome']))
    small_conns = conns[:max_conn_id+1]
    max_node_id = max(set(x for sublist in small_conns for x in sublist))
    small_nodes = nodes[:max_node_id+1]

    res = {}
    res['genome'] = genome['genome']
    res['nodes'] = small_nodes
    res['connections'] = small_conns

    return res


# Define activation functions
def dummy(x): return x
def sigmoid(x): return 1 / (1 + jnp.exp(-x))
def tanh(x): return jnp.tanh(x)
def relu(x): return jnp.maximum(0, x)
def gaussian(x): return jnp.exp(-x**2)  # Standard Gaussian function
def sin(x): return jnp.sin(x)
def cos(x): return jnp.cos(x)
def abs(x): return jnp.sign(x)

# Manual parameters. Need to update when problem changed
START_NODES = [0, 1, 2] # Mannually set starting node HERE. node 2 is bias.
OUTPUT_NODE = 3 # Mannually set output node HERE.
ACTIVATION_MAP = (dummy, dummy, dummy, sigmoid, tanh, relu, gaussian, sin, cos, abs) # Mannually set activation fucntion  HERE.


# Parallel training for multiple genomes 
class NEATModel:
    def __init__(self, genome_list, lr=0.001):
        # Initialize population
        self.num_genomes = len(genome_list)
        self.original_genome_list = genome_list.copy()

        # Compress genomes 
        genome_list = [compress(gen) for gen in genome_list]
        
        # Find longest genome and step
        self.max_nodes = 0
        max_steps = 0
        max_step_length = 0
        steps_list = []
        for genome_json in genome_list:
            # Number of nodes
            nodes = genome_json['nodes']
            num_nodes = len(nodes)
            if num_nodes > self.max_nodes:
                self.max_nodes = num_nodes

            conns = genome_json['connections']

            steps = sort_graph(conns, num_nodes, START_NODES)
            steps_list.append(steps)
            if len(steps) > max_steps:
                max_steps = len(steps)

            if max(len(seq) for seq in steps) > max_step_length:
                max_step_length = max(len(seq) for seq in steps)
        
        # Process each genome separately
        self.nodes = []
        self.weight_matrix = []
        self.steps = []
        print('learning rate is ', lr)

        for idx, genome_json in enumerate(genome_list):
            # Get values from json
            nodes = jnp.array(genome_json['nodes'])
            padded_nodes = jnp.zeros(self.max_nodes, dtype=jnp.int16)
            padded_nodes = padded_nodes.at[:len(nodes)].set(jnp.array(nodes, dtype=jnp.int16))
            conns = genome_json['connections']

            weight_matrix = jnp.zeros((self.max_nodes, self.max_nodes), dtype=jnp.float16)
            genome_data = [(gen["0"], gen["1"], conns[gen["0"]]) for gen in genome_json["genome"] if gen["2"] == 1]

            if genome_data:
                indices = jnp.array([(conn[0], conn[1]) for _, _, conn in genome_data])
                values = jnp.array([gen[1] for gen in genome_data], dtype=jnp.float16)

                # Use JAX's functional update
                weight_matrix = weight_matrix.at[indices[:, 0], indices[:, 1]].set(values)

            # Create step by Kahn's algorithm Sorting
            padded_steps = []
            steps = steps_list[idx]
            for step in steps:
                if len(step) < max_step_length:
                    step += [0]*(max_step_length-len(step))
                padded_steps.append(step)

            for _ in range(max_steps - len(steps)):
                padded_steps.append([0]*max_step_length) 

            # Store genome-specific parameters
            self.nodes.append(padded_nodes)
            self.weight_matrix.append(weight_matrix)
            self.steps.append(padded_steps)

        print(f"Shape is {self.max_nodes}, {max_step_length}, {max_steps}")
        self.nodes = jnp.stack(self.nodes)
        self.weight_matrix = jnp.stack(self.weight_matrix)
        self.steps = jnp.stack(jnp.array(self.steps))

        self.optimizer = optax.adagrad(learning_rate=lr, initial_accumulator_value=1e-8)
        self.opt_state = jax.vmap(self.optimizer.init)(self.weight_matrix)
    
    def forward_single_genome(self, weight_matrix_i, step_i, node_i):
        # Get network 
        def _apply_activation(x, index):
            # for vmap function of activation function
            return jax.lax.switch(index, ACTIVATION_MAP, x)
        vmap_apply_activation = jax.vmap(jax.vmap(_apply_activation, in_axes=(0, 0)))

        # Initialize input vector, zeros except inputs
        batch_size = self.inputs.shape[0]
        input_vector = jnp.zeros((batch_size, self.max_nodes))
        input_vector = input_vector.at[:, jnp.array(START_NODES)].set(self.inputs)
        
        
        for idx in range(1, step_i.shape[0]):
            step = step_i[idx,:] # Select step for each genome
            
            # Create matrix with shape # of genome x # of max nodes x # of max nodes
            sub_matrix = jnp.zeros((self.max_nodes, self.max_nodes), dtype=jnp.float16) 
            sub_matrix = sub_matrix.at[:, step].set(weight_matrix_i[ :, step]) # Select weight in the step
            input_vector += jnp.dot(input_vector, sub_matrix)
            # Extract only functions for the node in step
            activation_functions = jnp.zeros((batch_size, self.max_nodes), dtype=int)
            activation_functions = activation_functions.at[:, step].set(node_i[step]) # Set activation function for each node
            
            # Apply activation function two vmaps (vmap for each node and vmap for each genome)
            input_vector = vmap_apply_activation(input_vector, activation_functions)
        return input_vector[:, OUTPUT_NODE] # Out put for each genome and input
    
    def forward_multiple_genome(self, inputs):
        self.inputs = inputs
        output = jax.vmap(self.forward_single_genome,in_axes=(0, 0, 0))(self.weight_matrix, self.steps, self.nodes)
        return jax.nn.sigmoid(output)
    
    def single_genome_loss(self, weight_matrix_i, step_i, node_i):
        assert jnp.all((self.target_values == 0) | (self.target_values == 1)), "Target values must be 0 or 1"
        # forward pass on just that one genome's data
        output = jax.nn.sigmoid(self.forward_single_genome(weight_matrix_i, step_i, node_i))
        bce_loss = - (self.target_values * jnp.log(output + 1e-8) + (1 - self.target_values) * jnp.log(1 - output + 1e-8))
        return jnp.mean(bce_loss)
    
    def multiple_genome_loss(self):
        # forward pass on just that one genome's data
        losses = jax.vmap(self.single_genome_loss,in_axes=(0, 0, 0))(self.weight_matrix, self.steps, self.nodes)
        return losses
    
    def update_genomes(self):
        """
        Updates each genome’s parameters concurrently.
        """
        # Define a per-genome update function.
        # It takes the parameters, its optimizer state, and the extra data for that genome.
        def update_fn(weight_matrix_i, opt_state_i, step_i, node_i):
            # Compute loss and gradients for one genome:
            loss, grads = jax.value_and_grad(self.single_genome_loss)(
                weight_matrix_i, step_i, node_i)
            # Get the updates and new optimizer state:
            updates, new_opt_state = self.optimizer.update(grads, opt_state_i)
            # Apply the updates:
            new_weight_matrix_i = optax.apply_updates(weight_matrix_i, updates)
            return new_weight_matrix_i, new_opt_state, loss

        # Vectorize the update function over all genomes:
        new_weight_matrix, new_opt_state, losses = jax.vmap(update_fn)(
            self.weight_matrix, self.opt_state, self.steps, self.nodes)
        
        # Update the model's parameters and optimizer states:
        self.weight_matrix = new_weight_matrix
        self.opt_state = new_opt_state
        return losses
    
    
    def train(self, inputs, target_values, nCycles=100, batch_size=16):
        jax.clear_caches()

        for cycle in range(nCycles):
            # Randomly select items
            indices = random.choice(key, inputs.shape[0], shape=(batch_size,))
            self.inputs = inputs[indices]
            self.target_values = target_values[indices]

            self.update_genomes()

            # Compute current loss and print
            if cycle % 50 == 0 or cycle == nCycles - 1:
                loss_value = self.multiple_genome_loss()
                print(f"Cycle {cycle}, Loss: {loss_value}")

        # Get final fitness
        fitness = - self.multiple_genome_loss()
        fitness = fitness.tolist() # Higher fitness means lower loss

        # Update weight
        new_genome_list = []
        for i, genome_json in enumerate(self.original_genome_list):
            # Get weight matrix for genome
            weight_matrix = self.weight_matrix[i]
            # Get connections
            conns = genome_json['connections']

            # Update weight
            for gen in genome_json['genome']:
                conn = conns[gen["0"]]
                if gen["2"] == 1:
                    gen["1"] = float(weight_matrix[conn[0], conn[1]]) # Assign new weight

            new_genome_list.append(genome_json)

        res = {"genome_list":new_genome_list, "fitness": fitness}
            
        return res

if __name__ == "__main__":
    # Example Usage:
    genome_list = []

    # 1-based edges
    edges = [
        [0, 4],
        [1, 4],
        [1, 5],
        [2, 4],
        [2, 3],
        [4, 3],
        [4, 5],
        [5, 3]
    ]

    genome_json = {
        "nodes": [0, 0, 0, 3, 5, 5],  # Node activations (sigmoid, relu, relu)
        "connections": edges,  # List of edges (from-to)
        "nInput": 2,
        "nOutput": 1,
        "genome": [
            {"0": 0, "1": 0.5, "2": 1},  # (index, weight, active)
            {"0": 1, "1": -0.8, "2": 1},
            {"0": 2, "1": 0.2, "2": 1},
            {"0": 3, "1": -0.2, "2": 1},
            {"0": 4, "1": 0.6, "2": 1},
            {"0": 5, "1": -0.1, "2": 1},
            {"0": 6, "1": 0.1, "2": 1},
            {"0": 7, "1": 0.5, "2": 1},
        ],
    }
    genome_list.append(genome_json)

    # 1-based edges
    edges = [
        [0, 4],
        [1, 5],
        [2, 4],
        [4, 5],
        [4, 6],
        [5, 3],
        [6, 3]
    ]

    genome_json = {
        "nodes": [0, 0, 0, 3, 5, 5, 5],  # Node activations (sigmoid, relu, relu, relu)
        "connections": edges,  # List of edges (from-to)
        "nInput": 2,
        "nOutput": 1,
        "genome": [
            {"0": 0, "1": 0.5, "2": 1},  # (index, weight, active)
            {"0": 1, "1": -0.8, "2": 1},
            {"0": 2, "1": 0.2, "2": 1},
            {"0": 3, "1": -0.2, "2": 1},
            {"0": 4, "1": 0.6, "2": 1},
            {"0": 5, "1": -0.1, "2": 1},
            {"0": 6, "1": 0.1, "2": 1},
        ],
    }
    genome_list.append(genome_json)


    import json

    with open('input.json', 'r') as f:
        data = json.load(f)

    genome_list = data['genes']
    json_gene_list = []
    for gene in genome_list:
        gene = json.loads(gene)
        json_gene_list.append(gene)


    inputs_w = np.array(list(data["inputs"]["w"].values()), dtype=np.float16)
    targets_w = np.array(list(data["targets"]["w"].values()), dtype=np.float16)

    # Reshape and add bias
    inputs = jnp.array(inputs_w).reshape((data["inputs"]["n"], data["inputs"]["d"])) #/ 5
    bias = jnp.ones((data["inputs"]["n"], 1))
    inputs_with_bias = jnp.concatenate([inputs, bias], axis=1)
    targets = jnp.array(targets_w)

    # inputs = jnp.array([[0.5,0.2,-0.8], [0.1,0.1,-0.2]], dtype=jnp.float16)
    # targets = jnp.array([1, 0], dtype=jnp.float16)


    # Initialize NEAT model
    neat = NEATModel(json_gene_list, lr=0.1)

    out = neat.forward_multiple_genome(inputs_with_bias)
    print('out is', out)
    # Run forward and backward pass
    out = neat.train(inputs_with_bias, targets, nCycles=100)

    # Print results
    print("Output:", out)