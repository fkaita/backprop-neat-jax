
async function updateWeightsWithBackend(weights, inputs, targets) {
    const response = await fetch('/update-weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ weights, inputs, targets }),
    });

    if (!response.ok) {
        throw new Error('Failed to update weights');
    }

    const data = await response.json();
    return data.updated_weights;
}

// Export for CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { updateWeightsWithBackend };
}

// Attach to window for browser
if (typeof window !== 'undefined') {
    window.updateWeightsWithBackend = updateWeightsWithBackend;
}
