import json
import os
import sys


def _resolve_path(*parts):
    """Resolve paths relative to project root (folder containing this file)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *parts)


def load_robot_config(path=None):
    # Keep default fixed for now
    if path is None:
        path = _resolve_path("config", "epuck.json")
    elif not os.path.isabs(path):
        path = _resolve_path(path)

    if not os.path.exists(path):
        print(f"ERROR: Robot config not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_world_file(world_name):
    """
    Accepts:
    - 'n_shape'                 -> worlds/n_shape.json
    - 'n_shape.json'            -> worlds/n_shape.json
    - '/abs/path/to/file.json'  -> absolute path
    - 'worlds/u_shape.json'     -> relative path from project root
    """
    if world_name is None:
        print("ERROR: world_name is required")
        return None

    # If absolute path, use directly
    if os.path.isabs(world_name):
        path = world_name
    else:
        # If user passed only name or filename, place under worlds/
        if "/" not in world_name and "\\" not in world_name:
            filename = world_name if world_name.endswith(".json") else f"{world_name}.json"
            path = _resolve_path("worlds", filename)
        else:
            # relative custom path
            path = _resolve_path(world_name)

    if not os.path.exists(path):
        print(f"ERROR: World not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_braitenberg(controller_name="braitenberg_avoidance.json"):
    """
    Accepts:
    - 'braitenberg_avoidance'                 -> models/braitenberg_avoidance.json
    - 'braitenberg_avoidance.json'            -> models/braitenberg_avoidance.json
    - '/abs/path/to/controller.json'          -> absolute path
    - 'models/custom_controller.json'         -> relative path from project root
    """
    if controller_name is None:
        print("ERROR: controller_name is required")
        return None

    # If absolute path, use directly
    if os.path.isabs(controller_name):
        path = controller_name
    else:
        # If only a file/name, resolve inside models/
        if "/" not in controller_name and "\\" not in controller_name:
            filename = controller_name if controller_name.endswith(".json") else f"{controller_name}.json"
            path = _resolve_path("models", filename)
        else:
            # relative custom path
            path = _resolve_path(controller_name)

    if not os.path.exists(path):
        print(f"ERROR: Controller model not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)