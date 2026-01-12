"""
A helper function to get a default model for quick testing
"""
from omegaconf import OmegaConf
import os

import torch
from .matanyone import MatAnyone

def get_matanyone_model(ckpt_path, device=None) -> MatAnyone:
    # Load the existing configuration file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "model", "base.yaml")
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    # Load the base configuration and structure it properly
    model_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.create({
        "model": model_cfg,
        "weights": ckpt_path,
        "amp": False,
        "flip_aug": False,
        "max_internal_size": -1,
        # Inference parameters
        "save_all": True,
        "use_all_masks": False,
        "use_long_term": False,
        "mem_every": 5,
        "max_mem_frames": 5,
        # Long-term memory parameters
        "long_term": {
            "count_usage": True,
            "max_mem_frames": 10,
            "min_mem_frames": 5,
            "num_prototypes": 128,
            "max_num_tokens": 10000,
            "buffer_tokens": 2000
        },
        "top_k": 30,
        "stagger_updates": 5,
        "chunk_size": -1,
        "save_scores": False,
        "save_aux": False,
        "visualize": False
    })

    # Load the network weights
    if device is not None:
        matanyone = MatAnyone(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(cfg.weights, map_location=device)
    else:  # if device is not specified, `.cuda()` by default
        matanyone = MatAnyone(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(cfg.weights)
        
    matanyone.load_weights(model_weights)

    return matanyone

