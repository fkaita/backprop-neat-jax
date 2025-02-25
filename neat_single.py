import jax
import jax.numpy as jnp
import jax.random as random
import optax
from collections import deque

import numpy as np


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


from collections import deque, defaultdict

def kahn_topological_sort(edges):
    # Step 1: Build graph and compute in-degrees
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for from_node, to_node in edges:
        graph[from_node].append(to_node)
        in_degree[to_node] += 1
        if from_node not in in_degree:  # Ensure all nodes are included
            in_degree[from_node] = 0

    # Step 2: Initialize queue with nodes having in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topological_order = []

    # Step 3: Process nodes
    while queue:
        node = queue.popleft()
        topological_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Step 4: Check if sorting is successful
    if len(topological_order) == len(in_degree):
        return topological_order
    else:
        return []  # Cycle detected


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
    def __init__(self, genome_json, lr=0.001):
        # Initialize population
        # self.num_genomes = len(genome_list)
        self.original_genome_list = genome_json.copy()

        genome_json = compress(genome_json)

        # # Compress genomes 
        # genome_list = [compress(gen) for gen in genome_list]
        
        # # Find longest genome and step
        # self.max_nodes = 0
        # max_steps = 0
        # max_step_length = 0
        # steps_list = []
        # for genome_json in genome_list:
        #     # Number of nodes
        #     nodes = genome_json['nodes']
        #     num_nodes = len(nodes)
        #     if num_nodes > self.max_nodes:
        #         self.max_nodes = num_nodes

        #     conns = genome_json['connections']
        #     # print(conns, num_nodes)
        #     # small_genome = compress(genome_json)

        #     steps = sort_graph(conns, num_nodes, START_NODES)
        #     print(steps)
        #     steps_list.append(steps)
        #     if len(steps) > max_steps:
        #         max_steps = len(steps)

        #     if max(len(seq) for seq in steps) > max_step_length:
        #         max_step_length = max(len(seq) for seq in steps)
        
        # Process each genome separately
        # self.networks = []
        # self.optimizers = []
        # self.opt_states = []
        # self.nodes = []
        # self.weight_matrix = []
        # self.steps = []
        # print('learning rate is ', lr)

        # self.networks = {
        #     "nodes": [],
        #     # "adj_matrix": [],
        #     "weight_matrix": [],
        #     "steps": []
        # }

        # Get values from json
        self.nodes = jnp.array(genome_json['nodes'])
        self.num_nodes = len(self.nodes)
        # padded_nodes = jnp.zeros(self.max_nodes, dtype=jnp.int16)
        # padded_nodes = padded_nodes.at[:len(nodes)].set(jnp.array(nodes, dtype=jnp.int16))

        # adj_matrix = jnp.zeros((self.max_nodes, self.max_nodes), dtype=jnp.bool_)
        conns = genome_json['connections']

        # Create Adjacency Matrix that represents graph
        # src, dst = jnp.array(conns).T
        # adj_matrix = adj_matrix.at[src, dst].set(1)

        # Create weight matrix
        self.weight_matrix = jnp.zeros((self.num_nodes, self.num_nodes), dtype=jnp.float16)
        genome_data = [(gen["0"], gen["1"], conns[gen["0"]]) for gen in genome_json["genome"] if gen["2"] == 1]

        if genome_data:
            indices = jnp.array([(conn[0], conn[1]) for _, _, conn in genome_data])
            values = jnp.array([gen[1] for gen in genome_data], dtype=jnp.float16)

            # Use JAX's functional update
            self.weight_matrix = self.weight_matrix.at[indices[:, 0], indices[:, 1]].set(values)

        for gen in genome_json['genome']:
            conn = conns[gen["0"]]
            if gen["2"] == 1:
                self.weight_matrix = self.weight_matrix.at[conn[0], conn[1]].set(gen["1"]) # Assign weight from json genome

        # Create step by Kahn's algorithm Sorting
        # self.steps = sort_graph(conns, self.num_nodes, START_NODES)
        self.steps = kahn_topological_sort(conns)
        # padded_steps = []
        # steps = steps_list[idx]
        # for step in steps:
        #     if len(step) < max_step_length:
        #         step += [0]*(max_step_length-len(step))
        #     padded_steps.append(step)

        # for _ in range(max_steps - len(steps)):
        #     padded_steps.append([0]*max_step_length) # This doesn't change result?? (NEED TO DOUBLE CHECK!!)
        
        # padded_steps = jnp.full(max_steps, fill_value=-1, dtype=jnp.int32)
        # padded_steps = padded_steps.at[:len(steps)].set(jnp.array(steps, dtype=jnp.int32))
        
        # initialize optimizer
        self.optimizer = optax.adagrad(learning_rate=lr, initial_accumulator_value=1e-8)
        self.opt_state = self.optimizer.init(self.weight_matrix)

        # Store genome-specific parameters
        # self.nodes.append(padded_nodes)
        # # self.networks['adj_matrix'].append(adj_matrix)
        # self.weight_matrix.append(weight_matrix)
        # self.steps.append(padded_steps)
        # self.optimizers.append(optimizer)
        # self.opt_states.append(opt_state)

        # print(f"Shape is {self.max_nodes}, {max_step_length}, {max_steps}")
        # self.nodes = jnp.stack(self.nodes)
        # self.weight_matrix = jnp.stack(self.weight_matrix)
        # print("initial weight matrix is", jnp.sum(self.weight_matrix[0]))
        # self.steps = jnp.stack(jnp.array(self.steps))

        # self.optimizer = optax.adagrad(learning_rate=lr, initial_accumulator_value=1e-8)
        # self.opt_state = jax.vmap(self.optimizer.init)(self.weight_matrix)
        # self.optimizer = optax.adamw(learning_rate=lr)
        # self.opt_state = self.optimizer.init(self.weight_matrix)

        # self.stacked_networks = {key: jnp.stack(jnp.array(self.networks[key])) for key in self.networks}
        # for key in self.networks:
        #     print(f"{key} shape is {self.stacked_networks[key].shape}")

    def forward(self, weight_matrix=None):
        # Get network 
        if weight_matrix is None:
            # fallback to self.networks
            weight_matrix = self.weight_matrix
        
        # Extract everything from networks_params
        # weight_matrix = net['weight_matrix']

        def _apply_activation(x, index):
            # for vmap function of activation function
            return jax.lax.switch(index, ACTIVATION_MAP, x)
        
        def set_submatrix_for_one_genome(sub_matrix_i, step_i, weight_matrix_i):
            # sub_matrix_i: shape (max_nodes, max_nodes)
            # step_i: shape (step_size,)
            # weight_matrix_i: shape (max_nodes, max_nodes)
            # We want sub_matrix_i[:, step_i] = weight_matrix_i[:, step_i]
            return sub_matrix_i.at[:, step_i].set(weight_matrix_i[:, step_i])


        # Initialize input vector, zeros except inputs
        batch_size = self.inputs.shape[0]
        # input_vector = jnp.zeros((batch_size, self.max_nodes))
        # input_vector = input_vector.at[:, jnp.array(START_NODES)].set(self.inputs)
        # # Expand input vector for all genome
        # input_vector = jnp.broadcast_to(input_vector, (self.num_genomes, input_vector.shape[0], input_vector.shape[1]))
        input_vector = jnp.zeros((self.num_genomes, batch_size, self.max_nodes))
        input_vector = input_vector.at[:, :, START_NODES].set(jnp.broadcast_to(self.inputs, (self.num_genomes, batch_size, len(START_NODES))))


        for idx in range(1, self.steps.shape[1]):
            step = self.steps[:,idx,:] # Select step for each genome
            # step = jnp.unique(step, axis=1) # Select only unique step (because multiple 0 in step)
            
            # Create matrix with shape # of genome x # of max nodes x # of max nodes
            sub_matrix = jnp.zeros((self.num_genomes, self.max_nodes, self.max_nodes), dtype=jnp.float16) 
            sub_matrix = sub_matrix.at[:, :, step].set(weight_matrix[:, :, step]) # Select weight in the step
            sub_matrix = jax.vmap(
                set_submatrix_for_one_genome, 
                in_axes=(0, 0, 0)  # batch each input across index 0
            )(
                sub_matrix,           # shape (num_genomes, max_nodes, max_nodes)
                step,                 # shape (num_genomes, step_size)
                weight_matrix         # shape (num_genomes, max_nodes, max_nodes)
            )
            # sub_matrix = sub_matrix * net['adj_matrix'] # Only existing connections
            input_vector += jax.vmap(jnp.dot, in_axes=(0, 0))(input_vector, sub_matrix)


            # Extract only functions for the node in step
            activation_functions = jnp.zeros((self.num_genomes, self.max_nodes), dtype=int)
            activation_functions = activation_functions.at[:, step].set(self.nodes[:, step]) # Set activation function for each node
            # print(input_vector)
            # print(idx, activation_functions.shape, input_vector.shape)
            
            # Apply activation function two vmaps (vmap for each node and vmap for each genome)
            vmap_apply_activation = jax.vmap(jax.vmap(_apply_activation, in_axes=(0, 0)), in_axes=(0, 0))
            vmap_apply_activation(input_vector.transpose(0,2,1), activation_functions).transpose(0,2,1)
        return input_vector[:, :, OUTPUT_NODE] # Out put for each genome and input
    
    def forward_single_genome(self, weight_matrix):
        # Get network 
        def _apply_activation(x, index):
            # for vmap function of activation function
            return jax.lax.switch(index, ACTIVATION_MAP, x)
        vmap_apply_activation = jax.vmap(jax.vmap(_apply_activation, in_axes=(0, 0)))

        self.weight_matrix = weight_matrix

        # Initialize input vector, zeros except inputs
        # self.inputs = inputs
        batch_size = self.inputs.shape[0]
        input_vector = jnp.zeros((batch_size, self.num_nodes))
        input_vector = input_vector.at[:, jnp.array(START_NODES)].set(self.inputs)
        
        for idx in range(1, len(self.steps)):
            step = self.steps[idx] # Select step for each genome
            
            # Create matrix with shape # of genome x # of max nodes x # of max nodes
            sub_matrix = jnp.zeros((self.num_nodes, self.num_nodes), dtype=jnp.float16) 
            sub_matrix = sub_matrix.at[:, step].set(self.weight_matrix[ :, step]) # Select weight in the step

            # sub_matrix = sub_matrix * net['adj_matrix'] # Only existing connections
            input_vector += jnp.dot(input_vector, sub_matrix)
            # Extract only functions for the node in step
            activation_functions = jnp.zeros((batch_size, self.num_nodes), dtype=int)
            activation_functions = activation_functions.at[:, step].set(self.nodes[jnp.array(step)]) # Set activation function for each node
            
            # Apply activation function two vmaps (vmap for each node and vmap for each genome)
            input_vector = vmap_apply_activation(input_vector, activation_functions)
        return input_vector[:, OUTPUT_NODE] # Out put for each genome and input
    
    def forward_multiple_genome(self, inputs):
        self.inputs = inputs
        output = jax.vmap(self.forward_single_genome,in_axes=(0, 0, 0))(self.weight_matrix, self.steps, self.nodes)
        return jax.nn.sigmoid(output)
    
    def mse_loss(self, weight_matrix):
        # # Assign updated weight matrix
        # networks_copy = dict(self.stacked_networks)
        # networks_copy['weight_matrix'] = weight_matrix
        outputs = self.forward(weight_matrix)
        return jnp.mean((outputs - self.target_values) ** 2, axis=1) # Get loss for each genome
    
    def single_genome_loss(self, weight_matrix):
        assert jnp.all((self.target_values == 0) | (self.target_values == 1)), "Target values must be 0 or 1"
        # forward pass on just that one genome's data
        output = jax.nn.sigmoid(self.forward_single_genome(weight_matrix))
        # output = self.forward_single_genome(weight_matrix_i, step_i, node_i)

        # print(output)
        # print(self.target_values)

        # return jnp.mean((output - self.target_values)**2)
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
    
    def backward(self):
        grads_fn = jax.grad(self.single_genome_loss)
        grads = grads_fn(self.weight_matrix)
        return jnp.sum(grads, axis=1) # Reduce zero matrixes from jcov
    
    def vmap_backward(self):
        grads_fn = jax.vmap(jax.grad(self.single_genome_loss),in_axes=(0, 0, 0))
        grads = grads_fn(self.weight_matrix, self.steps, self.nodes)
        return grads
    
    def train(self, inputs, target_values, nCycles=10, batch_size=16):
        jax.clear_caches()

        for cycle in range(nCycles):
            # Randomly select items
            indices = random.choice(key, inputs.shape[0], shape=(batch_size,))
            self.inputs = inputs[indices]
            self.target_values = target_values[indices]

            grads = self.backward()
            # grads = self.vmap_backward()
            # self.update_genomes()

            # Compute current loss and print
            if cycle % 50 == 0 or cycle == nCycles - 1:
                # loss_value = self.mse_loss(self.weight_matrix)
                loss_value = self.single_genome_loss(self.weight_matrix)
                print(f"Cycle {cycle}, Loss: {loss_value}")
                # print(grads[0])
                # print(self.weight_matrix[0])

            # for i in range(self.num_genomes):
            #     updates, self.opt_states[i] = self.optimizers[i].update(grads[i], self.opt_states[i])
            #     self.weight_matrix = self.weight_matrix.at[i].set(optax.apply_updates(self.weight_matrix[i], updates))

            # Convert lists to JAX arrays if needed
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.weight_matrix = optax.apply_updates(self.weight_matrix, updates)

        # Get final fitness
        # fitness = self.mse_loss(self.weight_matrix).tolist()
        fitness = - self.single_genome_loss(self.weight_matrix)
        fitness = fitness.tolist() # Higher fitness means lower loss

        out = self.forward_single_genome(inputs)
        print('out is', out)

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
    json_gene = json_gene_list[100]
    neat = NEATModel(json_gene, lr=0.1)
    print('before')
    for gen in json_gene['genome']:
        print(json_gene['connections'][gen['0']], gen['0'])
    print('after')
    for gen in compress(json_gene)['genome']:
        print(compress(json_gene)['connections'][gen['0']])

    # print(json_gene)
    print(neat.steps)

    print(kahn_topological_sort(compress(json_gene)['connections']))
    print(neat.weight_matrix)

    # out = neat.forward_single_genome(inputs_with_bias)
    # Run forward and backward pass
    out = neat.train(inputs_with_bias, targets, nCycles=100)

    # Print results
    print("Output:", out)