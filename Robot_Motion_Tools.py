# Robot_Tools/Robot_Motion_Tools.py

from pydobot.dobot import MODE_PTP
import pydobot
import time
import numpy as np
import os
import json
from google.genai import types
from config import M, default_port

# =========================================================
# GLOBAL STATE
# =========================================================
device = None          # single global Dobot handle
affine_matrix = M   # optional global 3x3 affine for pixel->robot
default_port = default_port

# =========================================================
# HELPER: ensure device is connected
# =========================================================
def _ensure_device(port = default_port):
    global device
    if device is None:
        device = pydobot.Dobot(port)
    return device

# =========================================================
# ROBOT FUNCTION 1: get_dobot_device
# =========================================================
def get_dobot_device(port = default_port):
    """
    Connect to the Dobot and store the device globally.
    """
    global device
    try:
        device = pydobot.Dobot(port)
        return f"Connected to Dobot on port {port}"
    except Exception as e:
        # Return string so it is JSON-serializable for the tool response
        return f"Error connecting to Dobot on port {port}: {e}"


schema_get_dobot_device = types.FunctionDeclaration(
    name="get_dobot_device",
    description=(
        "Connects to the Dobot robot on the specified serial/USB port and "
        "stores the device handle for subsequent robot commands. "
        "If no port is provided, uses a default like '/dev/ttyACM0'."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "port": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Serial/USB port where the Dobot is connected, e.g. "
                    "'/dev/ttyACM0' on Linux or 'COM3' on Windows. Optional."
                ),
            ),
        },
    ),
)

def device_close():
    device.close()
# =========================================================
# ROBOT FUNCTION 2: move_to_home
# =========================================================
def move_to_home():
    """
    Home the Dobot. If not already connected, connect on default port.
    """
    try:
        dev = _ensure_device()
        print("Homing the robot...")
        dev.home()
        time.sleep(2)
        return "Robot homed"
    except Exception as e:
        return f"Error homing robot: {e}"


schema_move_to_home = types.FunctionDeclaration(
    name="move_to_home",
    description="Homes the Dobot robot (moves it to its reference home position).",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
    ),
)

# =========================================================
# ROBOT FUNCTION 3: move_to_specific_position
# =========================================================
def move_to_specific_position(x: float, y: float, z: float, r: float = 0.0):
    """
    Move the robot to a specific Cartesian pose (x, y, z, r).
    Uses global device; connects if needed.
    """
    try:
        dev = _ensure_device()
        dev.speed(50, 50)
        dev.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x, y=y, z=z, r=r)
        time.sleep(2)
        return f"Moved to position x={x}, y={y}, z={z}, r={r}"
    except Exception as e:
        return f"Error in move_to_specific_position: {e}"


schema_move_to_specific_position = types.FunctionDeclaration(
    name="move_to_specific_position",
    description=(
        "Moves the Dobot robot to a specific Cartesian pose (x, y, z, r) "
        "using PTP motion. Units are typically millimeters and degrees."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "x": types.Schema(
                type=types.Type.NUMBER,
                description="Target X position in robot coordinates (mm).",
            ),
            "y": types.Schema(
                type=types.Type.NUMBER,
                description="Target Y position in robot coordinates (mm).",
            ),
            "z": types.Schema(
                type=types.Type.NUMBER,
                description="Target Z position in robot coordinates (mm).",
            ),
            "r": types.Schema(
                type=types.Type.NUMBER,
                description="Target end-effector rotation around Z (degrees). Optional.",
            ),
        },
    ),
)

# =========================================================
# ROBOT FUNCTION 4: get_current_pose
# =========================================================
def get_current_pose():
    """
    Get current Cartesian pose and joint angles from the Dobot.
    """
    try:
        dev = _ensure_device()
        time.sleep(0.5)
        print("Getting current pose...")
        pose, joint = dev.get_pose()
        # Make it JSON-friendly
        return {
            "pose": list(pose),
            "joint": list(joint),
        }
    except Exception as e:
        return f"Error in get_current_pose: {e}"


schema_get_current_pose = types.FunctionDeclaration(
    name="get_current_pose",
    description="Reads and returns the Dobot's current Cartesian pose and joint angles.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
    ),
)

# =========================================================
# ROBOT FUNCTION 5: suction_on / suction_off
# =========================================================
def suction_on():
    """
    Turn suction on.
    """
    try:
        dev = _ensure_device()
        dev.suck(True)
        return "Suction ON"
    except Exception as e:
        return f"Error turning suction ON: {e}"


schema_suction_on = types.FunctionDeclaration(
    name="suction_on",
    description="Turns the Dobot's suction gripper ON.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
    ),
)


def suction_off():
    """
    Turn suction off.
    """
    try:
        dev = _ensure_device()
        dev.suck(False)
        return "Suction OFF"
    except Exception as e:
        return f"Error turning suction OFF: {e}"


schema_suction_off = types.FunctionDeclaration(
    name="suction_off",
    description="Turns the Dobot's suction gripper OFF.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
    ),
)

# =========================================================
# AFFINE + PIXEL->ROBOT HELPERS
# =========================================================
def apply_affine(M, u, v):
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    XY = M @ uv1
    return float(XY[0]), float(XY[1])


def set_affine_matrix(matrix_flat):
    """
    Optional helper: set the global affine matrix from a flat list of 9 values.
    """
    global affine_matrix
    try:
        arr = np.array(matrix_flat, dtype=np.float64).reshape(3, 3)
        affine_matrix = arr
        return "Affine matrix updated"
    except Exception as e:
        return f"Error setting affine matrix: {e}"


schema_set_affine_matrix = types.FunctionDeclaration(
    name="set_affine_matrix",
    description=(
        "Sets the global 3x3 affine matrix used to convert image pixels (u, v) "
        "into robot coordinates (X, Y)."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "matrix_flat": types.Schema(
                type=types.Type.ARRAY,
                description="Flat list of 9 numbers representing the 3x3 affine matrix.",
                items=types.Schema(type=types.Type.NUMBER),
            ),
        },
    ),
)

# =========================================================
# ROBOT FUNCTION 6: move_robot_point_above
# =========================================================
def move_robot_point_above(u: float, v: float, z_above: float = -30.0):
    """
    Move robot to a point ABOVE the block at image pixel (u, v),
    using the global affine_matrix.
    """
    global affine_matrix
    try:
        dev = _ensure_device()
        if affine_matrix is None:
            return "Error: affine matrix not set. Call set_affine_matrix first."

        Xa, Ya = apply_affine(affine_matrix, u, v)
        print(f"Affine: pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")

        dev.speed(50, 50)
        dev.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=Xa, y=Ya, z=z_above, r=0.0)
        time.sleep(1)
        return {
            "x": Xa,
            "y": Ya,
            "z": z_above,
            "message": "Moved above pixel point",
        }
    except Exception as e:
        return f"Error in move_robot_point_above: {e}"


schema_move_robot_point_above = types.FunctionDeclaration(
    name="move_robot_point_above",
    description=(
        "Moves the Dobot to a point above a pixel location (u, v) using a "
        "precomputed affine camera-to-robot transform. Z height is given by z_above."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "u": types.Schema(
                type=types.Type.NUMBER,
                description="Image pixel x-coordinate.",
            ),
            "v": types.Schema(
                type=types.Type.NUMBER,
                description="Image pixel y-coordinate.",
            ),
            "z_above": types.Schema(
                type=types.Type.NUMBER,
                description="Z height above the block (mm). Default -30.",
            ),
        },
    ),
)

# =========================================================
# ROBOT FUNCTION 7: move_robot_point_block
# =========================================================
def move_robot_point_block(u: float, v: float, block_height: float = -30.0):
    """
    Move robot to the BLOCK height at image pixel (u, v),
    using the global affine_matrix.
    """
    global affine_matrix
    try:
        dev = _ensure_device()
        if affine_matrix is None:
            return "Error: affine matrix not set. Call set_affine_matrix first."

        Xa, Ya = apply_affine(affine_matrix, u, v)
        print(f"Affine: pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")

        dev.speed(50, 50)
        dev.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=Xa, y=Ya, z=block_height, r=0.0)
        time.sleep(1)
        return {
            "x": Xa,
            "y": Ya,
            "z": block_height,
            "message": "Moved to block height",
        }
    except Exception as e:
        return f"Error in move_robot_point_block: {e}"


schema_move_robot_point_block = types.FunctionDeclaration(
    name="move_robot_point_block",
    description=(
        "Moves the Dobot to the block height at the pixel location (u, v) "
        "using the camera-to-robot affine transform."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "u": types.Schema(
                type=types.Type.NUMBER,
                description="Image pixel x-coordinate.",
            ),
            "v": types.Schema(
                type=types.Type.NUMBER,
                description="Image pixel y-coordinate.",
            ),
            "block_height": types.Schema(
                type=types.Type.NUMBER,
                description="Target Z height for the block in robot coordinates (mm). Default -30.",
            ),
        },
    ),
)

# =========================================================
# FUNCTION 8: update_scene_memory (no device needed)
# =========================================================
def update_scene_memory(
    detection_json_path: str,
    scene_memory_path: str = "scene_memory.json",
    default_z: float = -40.0,
):
    """
    Read a detection JSON file and write a scene memory JSON with z added.
    """
    try:
        if not os.path.exists(detection_json_path):
            return f"Detection file not found: {detection_json_path}"

        with open(detection_json_path, "r") as f:
            detections = json.load(f)

        scene_memory = []
        for item in detections:
            label = item.get("label")
            x = item.get("x")
            y = item.get("y")
            scene_entry = {
                "label": label,
                "x": x,
                "y": y,
                "z": default_z,
            }
            scene_memory.append(scene_entry)

        with open(scene_memory_path, "w") as f:
            json.dump(scene_memory, f, indent=2)

        return f"Scene memory updated: {scene_memory_path}"
    except Exception as e:
        return f"Error in update_scene_memory: {e}"


schema_update_scene_memory = types.FunctionDeclaration(
    name="update_scene_memory",
    description=(
        "Reads a detection JSON file containing block labels and (x, y) pixel "
        "positions and updates a scene memory JSON file by adding a constant "
        "z height for each block."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "detection_json_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the detection JSON file.",
            ),
            "scene_memory_path": types.Schema(
                type=types.Type.STRING,
                description="Output path for the scene memory JSON (default 'scene_memory.json').",
            ),
            "default_z": types.Schema(
                type=types.Type.NUMBER,
                description="Default Z height to assign to all blocks (e.g. -40).",
            ),
        },
    ),
)

    
if __name__ == "__main__":
    # Optional manual test
    print(get_dobot_device())
    print(move_to_home())
