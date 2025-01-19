import pytest
import json
from app import app  # Import the Flask app from `app.py`

# Sample genome data for initialization
GENOME_JSON = {
    "nodes": [3, 4, 5, 6],  # Activation functions (e.g., 3=sigmoid, 4=tanh, etc.)
    "connections": [[0, 1], [1, 2], [2, 3]],  # Connections between nodes
    "nInput": 2,  # Number of inputs
    "nOutput": 1,  # Number of outputs
    "genome": [
        {"0": 0, "1": 0.5, "2": 1},  # (index, weight, active)
        {"0": 1, "1": -0.8, "2": 1},
        {"0": 2, "1": 1.2, "2": 1},
    ],
}


# Sample input and target data for forward and backward passes
BATCH_SIZE = 3

INPUT_DATA = {
    "inputs": [[0.2, 0.5], [0.1, 0.4], [0.7, 0.3]]
}

TARGET_DATA = {
    "inputs": [[0.2, 0.5], [0.1, 0.4], [0.7, 0.3]],
    "targets": [[1.0], [0.0], [1.0]],  # True label
}

BATCH_GENES = [
    {
        "nodes": [3, 4, 5, 6],
        "connections": [[0, 1], [1, 2], [2, 3]],
        "nInput": 2,
        "nOutput": 1,
        "genome": [
            {"0": 0, "1": 0.5, "2": 1},
            {"0": 1, "1": -0.8, "2": 1},
            {"0": 2, "1": 1.2, "2": 1},
        ],
    },
    {
        "nodes": [3, 4, 5, 6],
        "connections": [[0, 1], [1, 2], [2, 3]],
        "nInput": 2,
        "nOutput": 1,
        "genome": [
            {"0": 0, "1": 0.3, "2": 1},
            {"0": 1, "1": -0.5, "2": 1},
            {"0": 2, "1": 0.9, "2": 1},
        ],
    },
]

BATCH_BACKWARD_DATA = {
    "genes": BATCH_GENES,  # List of genomes to process
    "inputs": TARGET_DATA["inputs"],  # Shared inputs for all genes
    "targets": TARGET_DATA["targets"],  # Shared targets for all genes
    "nCycles": 2,  # Number of training cycles
    "batch_size": 2,  # Mini-batch size
}

@pytest.fixture
def client():
    """Set up Flask test client"""
    with app.test_client() as client:
        yield client

def test_initialize(client):
    """Test NEAT model initialization"""
    response = client.post("/initialize", json={"genome": GENOME_JSON})
    assert response.status_code == 200
    assert response.json["message"] == "NEAT model initialized successfully"

def test_forward(client):
    """Test forward pass of the NEAT model"""
    client.post("/initialize", json={"genome": GENOME_JSON})  # Ensure model is initialized
    response = client.post("/forward", json=INPUT_DATA)
    assert response.status_code == 200
    assert "outputs" in response.json
    assert isinstance(response.json["outputs"], list)
    assert len(response.json["outputs"]) == BATCH_SIZE  # Check batch size
    assert len(response.json["outputs"][0]) == GENOME_JSON["nOutput"]

def test_backward(client):
    """Test backward pass (training) of the NEAT model"""
    client.post("/initialize", json={"genome": GENOME_JSON})  # Ensure model is initialized
    response = client.post("/backward", json=TARGET_DATA)

    assert response.status_code == 200
    response_data = response.json
    assert "updated_genome" in response_data
    assert "avg_error" in response_data
    assert isinstance(response_data["avg_error"], float)

def test_batch_backward(client):
    """Test batch backward pass (training) for multiple genomes with shared inputs/targets"""
    client.post("/initialize", json={"genome": GENOME_JSON})  # Ensure model is initialized
    response = client.post("/batch_backward", json=BATCH_BACKWARD_DATA)

    assert response.status_code == 200
    response_data = response.json

    # Ensure batch response is a list
    assert isinstance(response_data, list)
    assert len(response_data) == len(BATCH_GENES)  # Response should match number of genomes processed

    for result in response_data:
        assert "updated_genome" in result
        assert "avg_error" in result
        assert isinstance(result["avg_error"], float)  # Ensure loss is a valid float


if __name__ == "__main__":
    pytest.main(["-v", "test_app.py"])
