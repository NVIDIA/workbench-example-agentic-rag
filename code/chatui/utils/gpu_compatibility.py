"""Utility module for GPU compatibility checking."""

import json
import os
from typing import Dict, List, Optional, TypedDict

class GPUConfig(TypedDict):
    """Type definition for GPU configuration."""
    gpu_type: str
    num_gpus: int

class ModelCompatibility(TypedDict):
    """Type definition for model compatibility results."""
    compatible_models: List[str]
    warning_message: Optional[str]

def load_gpu_support_matrix() -> Dict:
    """Load the GPU support matrix from JSON file."""
    matrix_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nim_gpu_support_matrix.json')
    with open(matrix_path, 'r') as f:
        return json.load(f)

def get_compatible_models(gpu_type: str, num_gpus: str) -> ModelCompatibility:
    """
    Get list of compatible models for given GPU configuration.
    
    Args:
        gpu_type: Type of GPU (e.g. "H100", "A100 80GB")
        num_gpus: Number of GPUs as string (e.g. "1", "2", "4", "8", "16")
        
    Returns:
        ModelCompatibility with list of compatible models and optional warning
    """
    support_matrix = load_gpu_support_matrix()
    
    # Validate inputs
    if gpu_type not in support_matrix:
        return ModelCompatibility(
            compatible_models=[],
            warning_message=f"GPU type {gpu_type} not found in support matrix"
        )
        
    if num_gpus not in support_matrix[gpu_type]:
        return ModelCompatibility(
            compatible_models=[],
            warning_message=f"Configuration with {num_gpus} GPUs not supported for {gpu_type}"
        )
    
    # Get compatible models
    models = support_matrix[gpu_type][num_gpus]["models"]
    
    return ModelCompatibility(
        compatible_models=models,
        warning_message=None if models else f"No compatible models found for {gpu_type} with {num_gpus} GPUs"
    )

def get_gpu_types() -> List[str]:
    """Get list of supported GPU types."""
    return list(load_gpu_support_matrix().keys())

def get_supported_gpu_counts(gpu_type: str) -> List[str]:
    """Get list of supported GPU counts for a given GPU type."""
    support_matrix = load_gpu_support_matrix()
    if gpu_type not in support_matrix:
        return []
    return list(support_matrix[gpu_type].keys()) 