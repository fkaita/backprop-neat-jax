const BASE_URL = 'http://127.0.0.1:5000'; 

export async function updateWeightsWithBackend(weights, inputs, targets) {
    const response = await fetch(`${BASE_URL}/update-weights`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            weights: weights,
            inputs: inputs,
            targets: targets,
        }),
    });

    if (!response.ok) {
        throw new Error('Failed to update weights');
    }

    const data = await response.json();
    return data.updated_weights;
}
