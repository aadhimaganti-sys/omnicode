import os
import json
import time

KEYS_FILE = ".api_keys.json"
STATE_FILE = ".api_key_state.json"
ROTATION_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours

def get_rotated_api_key():
    """
    Returns the current API key to use. 
    If there are multiple keys, it rotates to the next one every 24 hours.
    """
    # If keys file doesn't exist, try to use the single environment variable
    if not os.path.exists(KEYS_FILE):
        return os.environ.get("GEMINI_API_KEY")

    with open(KEYS_FILE, "r") as f:
        keys = json.load(f)

    if not keys:
        return os.environ.get("GEMINI_API_KEY")
    
    # If there's only one key, just return it
    if len(keys) == 1:
        return keys[0]

    # We have 2 or more keys. Load or initialize the state file.
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {"current_index": 0, "last_switched": time.time()}

    current_time = time.time()
    
    # Check if 24 hours have passed since the last switch
    if current_time - state.get("last_switched", current_time) >= ROTATION_INTERVAL_SECONDS:
        state["current_index"] = (state["current_index"] + 1) % len(keys)
        state["last_switched"] = current_time
        # Save the updated state
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    
    return keys[state.get("current_index", 0)]

def add_api_key(new_key):
    """
    Adds a new API key to the rotation list.
    """
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "r") as f:
            keys = json.load(f)
    else:
        keys = []
        
    if new_key not in keys:
        keys.append(new_key)
        with open(KEYS_FILE, "w") as f:
            json.dump(keys, f)
        print(f"Added new API key. Total keys available for rotation: {len(keys)}")
    else:
        print("API key is already in the list.")
