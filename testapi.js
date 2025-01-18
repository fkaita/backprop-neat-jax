const Api = require('./ml/api');

console.log("Loaded Api:", Api);

async function testApi() {
    const genomeJson = {
        nodes: [3, 4, 5, 4],
        connections: [[0, 1], [1, 2], [2, 3]],
        nInput: 2,
        nOutput: 1,
        genome: [
            { "0": 0, "1": 0.5, "2": 1 },
            { "0": 1, "1": -0.8, "2": 1 },
            { "0": 2, "1": 1.2, "2": 1 }
        ]
    };

    try {
        const initResponse = await Api.initializeModel(genomeJson, 0.05);
        console.log("Model Initialized:", initResponse);

        const inputs = [[0.5, -0.2]];
        const forwardResponse = await Api.forwardPass(inputs);
        console.log("Forward Pass Output:", forwardResponse);

        const targets = [[1]];
        const backwardResponse = await Api.backwardPass(inputs, targets, 0.05);
        console.log("Backward Pass Result:", backwardResponse);
    } catch (error) {
        console.error("Error:", error);
    }
}

testApi();
