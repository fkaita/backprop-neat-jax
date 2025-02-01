import jax
import jax.numpy as jnp
import jax.random as random
import optax
from collections import deque


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
key = random.PRNGKey(0)

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

# Parallel training for multiple genomes 
class NEATModel:
    def __init__(self, genome_list, lr=0.01):
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
            # print(conns, num_nodes)
            steps = sort_graph(conns, num_nodes, START_NODES)
            steps_list.append(steps)
            if len(steps) > max_steps:
                max_steps = len(steps)

            if max(len(seq) for seq in steps) > max_step_length:
                max_step_length = max(len(seq) for seq in steps)
        
        # Process each genome separately
        self.networks = []
        self.optimizers = []
        self.opt_states = []

        self.networks = {
            "nodes": [],
            "adj_matrix": [],
            "weight_matrix": [],
            "steps": []
        }

        for idx, genome_json in enumerate(genome_list):
            # Get values from json
            nodes = jnp.array(genome_json['nodes'])
            padded_nodes = jnp.zeros(self.max_nodes, dtype=jnp.int16)
            padded_nodes = padded_nodes.at[:len(nodes)].set(jnp.array(nodes, dtype=jnp.int16))

            adj_matrix = jnp.zeros((self.max_nodes, self.max_nodes), dtype=jnp.bool_)
            conns = genome_json['connections']

            # Create Adjacency Matrix that represents graph
            src, dst = jnp.array(conns).T
            adj_matrix = adj_matrix.at[src, dst].set(1)

            # Create weight matrix
            weight_matrix = jnp.zeros((self.max_nodes, self.max_nodes), dtype=jnp.float16)
            for gen in genome_json['genome']:
                conn = conns[gen["0"]]
                if gen["2"] == 1:
                    weight_matrix = weight_matrix.at[conn[0], conn[1]].set(gen["1"]) # Assign weight from json genome

            # Create step by Kahn's algorithm Sorting
            # steps = sort_graph(conns, num_nodes, START_NODES)
            padded_steps = []
            steps = steps_list[idx]
            for step in steps:
                if len(step) < max_step_length:
                    step += [0]*(max_step_length-len(step))
                padded_steps.append(step)

            for _ in range(max_steps - len(steps)):
                padded_steps.append([0]*max_step_length) # This doesn't change result?? (NEED TO DOUBLE CHECK!!)
            
            # padded_steps = jnp.full(max_steps, fill_value=-1, dtype=jnp.int32)
            # padded_steps = padded_steps.at[:len(steps)].set(jnp.array(steps, dtype=jnp.int32))
            
            # initialize optimizer
            optimizer = optax.adagrad(learning_rate=lr, initial_accumulator_value=1e-8)
            opt_state = optimizer.init(weight_matrix)

            # Store genome-specific parameters
            self.networks['nodes'].append(padded_nodes)
            self.networks['adj_matrix'].append(adj_matrix)
            self.networks['weight_matrix'].append(weight_matrix)
            self.networks['steps'].append(padded_steps)
            self.optimizers.append(optimizer)
            self.opt_states.append(opt_state)

        self.stacked_networks = {key: jnp.stack(jnp.array(self.networks[key])) for key in self.networks}

    def forward(self, net=None):
        # Get network
        if net is None:
            # fallback to self.networks
            net = self.stacked_networks
        
        # Extract everything from networks_params
        weight_matrix = net['weight_matrix']

        def _apply_activation(x, index):
            # for vmap function of activation function
            return jax.lax.switch(index, ACTIVATION_MAP, x)

        # Initialize input vector, zeros except inputs
        batch_size = self.inputs.shape[0]
        input_vector = jnp.zeros((batch_size, self.max_nodes))
        input_vector = input_vector.at[:, jnp.array(START_NODES)].set(self.inputs)
        # Expand input vector for all genome
        input_vector = jnp.broadcast_to(input_vector, (self.num_genomes, input_vector.shape[0], input_vector.shape[1]))

        for idx in range(1, net['steps'].shape[1]):
            step = net['steps'][:,idx,:] # Select step for each genome
            # step = jnp.unique(step, axis=1) # Select only unique step (because multiple 0 in step)
            
            # Create matrix with shape # of genome x # of max nodes x # of max nodes
            sub_matrix = jnp.zeros((self.num_genomes, self.max_nodes, self.max_nodes), dtype=jnp.float16) 
            sub_matrix = sub_matrix.at[:, :, step].set(weight_matrix[:, :, step]) # Select weight in the step
            sub_matrix = sub_matrix * net['adj_matrix'] # Only existing connections
            input_vector += jax.vmap(jnp.dot, in_axes=(0, 0))(input_vector, sub_matrix)


            # Extract only functions for the node in step
            activation_functions = jnp.zeros((self.num_genomes, self.max_nodes), dtype=int)
            activation_functions = activation_functions.at[:, step].set(net['nodes'][:, step]) # Set activation function for each node
            # print(input_vector)
            # print(idx, activation_functions.shape, input_vector.shape)
            
            # Apply activation function two vmaps (vmap for each node and vmap for each genome)
            vmap_apply_activation = jax.vmap(jax.vmap(_apply_activation, in_axes=(0, 0)), in_axes=(0, 0))
            vmap_apply_activation(input_vector.transpose(0,2,1), activation_functions).transpose(0,2,1)
        return input_vector[:, :, OUTPUT_NODE] # Out put for each genome and input
    
    def mse_loss(self, weight_matrix):
        # Assign updated weight matrix
        networks_copy = dict(self.stacked_networks)
        networks_copy['weight_matrix'] = weight_matrix
        outputs = self.forward(networks_copy)
        return jnp.mean((outputs - self.target_values) ** 2, axis=1) # Get loss for each genome
    
    def backward(self):
        grads_fn = jax.jacrev(self.mse_loss)
        grads = grads_fn(self.stacked_networks['weight_matrix'])
        return jnp.sum(grads, axis=1) # Reduce zero matrixes from jcov
    
    def train(self, inputs, target_values, nCycles=100, batch_size=2):
        jax.clear_caches()

        for cycle in range(nCycles):
            # Randomly select items
            indices = random.choice(key, inputs.shape[0], shape=(batch_size,), replace=False)
            self.inputs = inputs[indices]
            self.target_values = target_values[indices]

            grads = self.backward()

            # Compute current loss and print
            if cycle % 10 == 0 or cycle == nCycles - 1:
                loss_value = self.mse_loss(self.stacked_networks['weight_matrix'])
                print(f"Cycle {cycle}, Loss: {loss_value}")

            for i in range(self.num_genomes):
                updates, self.opt_states[i] = self.optimizers[i].update(grads[i], self.opt_states[i])
                self.stacked_networks["weight_matrix"] = self.stacked_networks["weight_matrix"].at[i].set(optax.apply_updates(self.stacked_networks["weight_matrix"][i], updates))

        # Get final fitness
        fitness = self.mse_loss(self.stacked_networks["weight_matrix"]).tolist()

        # Update weight
        new_genome_list = []
        for i, genome_json in enumerate(self.original_genome_list):
            # Get weight matrix for genome
            weight_matrix = self.stacked_networks["weight_matrix"][i]
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

    inputs = jnp.array([[0.5,0.2,-0.8], [0.1,0.1,-0.2]], dtype=jnp.float16)
    target_values = jnp.array([1, 0], dtype=jnp.float16)


    # Initialize NEAT model
    neat = NEATModel(genome_list, lr=0.1)

    # Run forward and backward pass
    out = neat.train(inputs, target_values, nCycles=50)

    # Print results
    print("Output:", out)