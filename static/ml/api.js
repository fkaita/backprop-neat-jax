
// Exporting data format
// var data = {
//     nodes: copyArray(nodes), -> list of activation function by each node (4 = tanh etc... defined in other part)
//     connections: copyConnections(connections), -> connection between nodes
//     nInput: nInput, -> number of input for model
//     nOutput: nOutput, -> number of output for model
//     renderMode: renderMode, -> no need to use (return same value)
//     outputIndex: outputIndex, -> no need to use (return same value)
//     genome: this.connections, -> dict with key "0": connection index, "1": weight, "2": if connection is active 1 else 0
//     description: description -> no need to use (return same value)
//   };

(function (global) {
    "use strict";
    const BASE_URL = 'http://127.0.0.1:5001';
    /**
     * Initialize NEAT Model with genome data
     * @param {Object} genomeJson - The genome JSON data
     * @returns {Promise<Object>} Response message
     */
    async function initializeModel(genomeJson, learningRate = 0.01) {
        // console.log("Sending JSON:", JSON.stringify({ genome: genomeJson, learning_rate: learningRate }));

        const response = await fetch(`${BASE_URL}/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ genome: genomeJson, learning_rate: learningRate }),
        });

        if (!response.ok) {
            throw new Error('Failed to initialize model');
        }

        return await response.json();
    }

    /**
     * Perform forward pass with input data
     * @param {Array} inputs - The input data
     * @returns {Promise<Array>} Output values from the model
     */
    async function forwardPass(inputs) {
        const response = await fetch(`${BASE_URL}/forward`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ inputs }),
        });

        if (!response.ok) {
            throw new Error('Forward pass failed');
        }

        const data = await response.json();
        return data.outputs;
    }

    /**
     * Perform backward pass (training) with inputs and targets
     * @param {Array} inputs - The input batch
     * @param {Array} targets - The target batch
     * @returns {Promise<Object>} Updated genome and average error
     */
    async function backwardPass(inputs, targets, nCycles=1) {
        const response = await fetch(`${BASE_URL}/backward`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                inputs,
                targets,
                nCycles
            }),
        });

        if (!response.ok) {
            throw new Error('Backward pass (training) failed');
        }

        const data = await response.json();
        return {
            updatedGenome: data.updated_genome,
            avgError: data.avg_error,
            output: {
                n: data.output.n,  // Batch size
                d: data.output.d,  // Output dimension
                w: Array.from(data.output.w),  // Convert Float32Array to standard JS array
                dw: Array.from(data.output.dw) // Convert Float32Array to standard JS array
            }
        };
    }

    /**
     * Perform batch backward pass (training) for multiple genomes using shared inputs/targets.
     * @param {Array} geneList - List of genome objects
     * @param {Object} inputs - Shared input batch
     * @param {Object} targets - Shared target batch
     * @param {number} nCycles - Number of training cycles (default = 1)
     * @param {number} learnRate - learning rate
     * @returns {Promise<Array>} List of updated genomes and errors
     */
    async function batchBackwardPass(geneList, inputs, targets, nCycles = 1, learnRate=0.01) {
        // console.log(geneList)
        // console.log(inputs)
        // console.log(targets)
        const response = await fetch(`${BASE_URL}/batch_backward`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                genes: geneList,  // Send all genomes
                inputs,  // Send shared inputs
                targets,  // Send shared targets
                nCycles,
                learnRate,
            }),
        });

        if (!response.ok) {
            throw new Error("Batch backward pass (training) failed");
        }

        const results = await response.json();
        return results;  // Return the entire batch results
    }

    // Expose API functions globally for browser and Node.js
    const Api = {
        initializeModel,
        forwardPass,
        backwardPass,
        batchBackwardPass
    };

    // Attach to window for browser
    if (typeof window !== 'undefined') {
        window.Api = Api;
    }

    // Export for Node.js
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = Api;
    }
})(this);