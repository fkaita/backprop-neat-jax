export async function updateWeightsWithBackend(weights, inputs, targets) {
    const response = await fetch('/update-weights', {
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
