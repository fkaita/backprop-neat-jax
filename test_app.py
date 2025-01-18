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

if __name__ == "__main__":
    pytest.main(["-v", "test_app.py"])
