const BASE_URL = 'http://127.0.0.1:5001';



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


async function updateGraphWithBackend(neatGraphJson) {
    const response = await fetch(`${BASE_URL}/update-weights`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: neatGraphJson
    });

    if (!response.ok) {
        throw new Error('Failed to update weights');
    }

    return await response.json();
}

// Export for CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { updateGraphWithBackend };
}

// Attach to window for browser
if (typeof window !== 'undefined') {
    window.updateGraphWithBackend = updateGraphWithBackend;
}
