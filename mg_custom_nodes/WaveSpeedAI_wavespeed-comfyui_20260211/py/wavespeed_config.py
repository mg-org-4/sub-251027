"""
WaveSpeed Configuration Management

Handles persistent storage of API keys and other configuration settings.
"""

import json
import os
import logging

# Configuration file path (in the plugin root directory)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.json")

def save_api_key(api_key):
    """
    Save API key to config file

    Args:
        api_key: The WaveSpeed API key to save
    """
    try:
        config = load_config()
        config['api_key'] = api_key

        # Ensure directory exists
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)

        # Write config file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        # Set restrictive permissions (owner read/write only)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except Exception as e:
            logging.warning(f"[WaveSpeed Config] Could not set file permissions: {e}")

        logging.info("[WaveSpeed Config] API key saved successfully")
        return True
    except Exception as e:
        logging.error(f"[WaveSpeed Config] Failed to save API key: {e}")
        return False

def load_config():
    """
    Load configuration from file

    Returns:
        dict: Configuration dictionary
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logging.info("[WaveSpeed Config] Configuration loaded from file")
                return config
    except Exception as e:
        logging.error(f"[WaveSpeed Config] Failed to load config: {e}")

    return {}

def get_api_key_from_config():
    """
    Get API key from config file or environment variable

    Priority:
    1. Environment variable WAVESPEED_API_KEY
    2. Config file

    Returns:
        str: API key or None if not found
    """
    # Priority 1: Environment variable
    env_key = os.environ.get('WAVESPEED_API_KEY')
    if env_key:
        logging.info("[WaveSpeed Config] Using API key from environment variable")
        return env_key

    # Priority 2: Config file
    config = load_config()
    api_key = config.get('api_key')
    if api_key:
        logging.info("[WaveSpeed Config] Using API key from config file")
        return api_key

    return None

def delete_api_key():
    """
    Delete API key from config file

    Returns:
        bool: True if successful
    """
    try:
        config = load_config()
        if 'api_key' in config:
            del config['api_key']

            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)

            logging.info("[WaveSpeed Config] API key deleted successfully")
        return True
    except Exception as e:
        logging.error(f"[WaveSpeed Config] Failed to delete API key: {e}")
        return False

def has_api_key():
    """
    Check if API key is configured

    Returns:
        bool: True if API key exists
    """
    return get_api_key_from_config() is not None
