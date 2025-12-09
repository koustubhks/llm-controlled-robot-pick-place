import os
import json
from google.genai import types
from config import (
    M,                 # 3x3 affine matrix for pixel -> robot (X,Y)
    z_above,           # safe travel height (e.g. 100)
    z_table,           # Z at table contact
    block_height_mm,   # block physical thickness
    block_length_mm,   # block physical length
    stack_delta_mm,    # extra height when stacking (to avoid collision)
    side_offset_mm,    # extra XY gap when placing beside
)

from Robot_Tools.Robot_Motion_Tools import (
    move_robot_point_above,
    move_robot_point_block,
    suction_on,
    suction_off,
    apply_affine,            # apply_affine(M, u, v) -> (X, Y)
    move_to_specific_position,
)

def pick_and_place_block(
    detection_json_path: str = "captures/capture_scene.json",
    source_label: str = "blue1",
    target_label: str = "green1",
    placement_type: str = "on_top",   # "on_top" or "beside"
    direction: str = "right",         # used only for "beside": "front", "back", "right", "left"
):
    """
    High-level pick-and-place behavior using vision detections and calibrated Z/XY.

    Modes:
    1) placement_type == "on_top":
        - Pick source block from table
        - Place it on TOP of target block
        - Place Z = z_table + 2*block_height_mm + stack_delta_mm

    2) placement_type == "beside":
        - Pick source block from table
        - Place it beside the target block in the given direction:
            - "right", "left": +/- X direction
            - "front", "back": +/- Y direction
        - Offset distance = block_length_mm + side_offset_mm
        - Place Z = z_table + block_height_mm  (top of single block / release height)
    """
    try:
        if not os.path.exists(detection_json_path):
            return {
                "error": f"detection_json_path not found: {detection_json_path}"
            }

        with open(detection_json_path, "r") as f:
            detections = json.load(f)

        # Build lookup: label -> (u, v)
        label_to_uv = {}
        for item in detections:
            label = item.get("label")
            x = item.get("x")
            y = item.get("y")
            if label is not None and x is not None and y is not None:
                label_to_uv[label] = (x, y)

        if source_label not in label_to_uv:
            return {
                "error": f"{source_label} not found in detection file",
                "available_labels": list(label_to_uv.keys()),
            }

        if target_label not in label_to_uv:
            return {
                "error": f"{target_label} not found in detection file",
                "available_labels": list(label_to_uv.keys()),
            }

        src_u, src_v = label_to_uv[source_label]
        tgt_u, tgt_v = label_to_uv[target_label]

        # -----------------------------------------
        # HEIGHT CALCULATION
        # -----------------------------------------
        # Pick from top of block sitting on the table
        pickup_height = z_table + block_height_mm

        if placement_type == "on_top":
            # Place on top of target block (2-block stack)
            place_height = z_table + 2 * block_height_mm + stack_delta_mm
        elif placement_type == "beside":
            # Place on table beside target block; Z to release at block top
            place_height = z_table + block_height_mm
        else:
            return {
                "error": f"Unknown placement_type '{placement_type}'. Use 'on_top' or 'beside'."
            }

        steps = []

        # -----------------------------------------
        # PICK SEQUENCE (always the same)
        # -----------------------------------------
        steps.append({"step": "move_above_source", "u": src_u, "v": src_v})
        steps.append(move_robot_point_above(src_u, src_v, z_above))

        steps.append({"step": "descend_to_pick", "z": pickup_height})
        steps.append(move_robot_point_block(src_u, src_v, pickup_height))

        steps.append({"step": "suction_on"})
        steps.append(suction_on())

        steps.append({"step": "lift_after_pick", "z": z_above})
        steps.append(move_robot_point_above(src_u, src_v, z_above))

        # -----------------------------------------
        # PLACE SEQUENCE
        # -----------------------------------------
        if placement_type == "on_top":
            # Use the target block's pixel (u,v) directly
            steps.append({"step": "move_above_target_on_top", "u": tgt_u, "v": tgt_v})
            steps.append(move_robot_point_above(tgt_u, tgt_v, z_above))

            steps.append({"step": "descend_to_place_on_top", "z": place_height})
            steps.append(move_robot_point_block(tgt_u, tgt_v, place_height))

            place_info = {
                "mode": "on_top",
                "target_pixel": {"u": tgt_u, "v": tgt_v},
            }

        else:  # placement_type == "beside"
            # Convert target pixel -> robot (X, Y)
            tgt_X, tgt_Y = apply_affine(M, tgt_u, tgt_v)

            # Offset distance in robot coordinates
            offset = block_length_mm + side_offset_mm

            # NOTE (adjusted to match your robot):
            #  - We now assume: +X = "front", -X = "back"
            #                    +Y = "right", -Y = "left"
            dX = 0.0
            dY = 0.0
            dir_norm = (direction or "").lower()

            if dir_norm == "front":
                dX = offset          # move +X
            elif dir_norm == "back":
                dX = -offset         # move -X
            elif dir_norm == "right":
                dY = offset          # move +Y
            elif dir_norm == "left":
                dY = -offset         # move -Y
            else:
                return {
                    "error": (
                        f"Unknown direction '{direction}'. "
                        "Use 'front', 'back', 'right', or 'left'."
                    )
                }


            place_X = tgt_X + dX
            place_Y = tgt_Y + dY

            # Move above beside-location in robot coordinates
            steps.append({
                "step": "move_above_target_beside",
                "X": place_X,
                "Y": place_Y,
                "Z": z_above,
            })
            steps.append(
                move_to_specific_position(
                    x=place_X,
                    y=place_Y,
                    z=z_above,
                    r=0.0,
                )
            )

            # Descend to place height
            steps.append({
                "step": "descend_to_place_beside",
                "X": place_X,
                "Y": place_Y,
                "Z": place_height,
            })
            steps.append(
                move_to_specific_position(
                    x=place_X,
                    y=place_Y,
                    z=place_height,
                    r=0.0,
                )
            )

            place_info = {
                "mode": "beside",
                "direction": dir_norm,
                "target_pixel": {"u": tgt_u, "v": tgt_v},
                "place_robot": {"x": place_X, "y": place_Y, "z": place_height},
            }

        # Common: release and lift
        steps.append({"step": "suction_off"})
        steps.append(suction_off())

        steps.append({"step": "lift_after_place", "z": z_above})
        # For 'on_top', we move above target pixel; for 'beside', above the side location.
        if placement_type == "on_top":
            steps.append(move_robot_point_above(tgt_u, tgt_v, z_above))
        else:
            steps.append(
                move_to_specific_position(
                    x=place_info["place_robot"]["x"],
                    y=place_info["place_robot"]["y"],
                    z=z_above,
                    r=0.0,
                )
            )

        return {
            "message": f"Placed {source_label} using placement_type='{placement_type}'",
            "source_label": source_label,
            "target_label": target_label,
            "source_pixel": {"u": src_u, "v": src_v},
            "heights": {
                "pickup_height": pickup_height,
                "place_height": place_height,
                "z_above": z_above,
            },
            "config_constants": {
                "z_table": z_table,
                "block_height_mm": block_height_mm,
                "block_length_mm": block_length_mm,
                "stack_delta_mm": stack_delta_mm,
                "side_offset_mm": side_offset_mm,
            },
            "placement": place_info,
            "steps": steps,
        }

    except Exception as e:
        return {"error": f"pick_and_place_block failed: {e}"}


schema_pick_and_place_block = types.FunctionDeclaration(
    name="pick_and_place_block",
    description=(
        "Pick a block identified in a detection JSON and either stack it on top "
        "of another block ('on_top') or place it beside another block in one of "
        "four directions ('front', 'back', 'right', 'left'). Uses calibrated "
        "heights and offsets from the config file."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "detection_json_path": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Path to detection JSON produced by camera, "
                    "default 'captures/capture_scene.json'."
                ),
            ),
            "source_label": types.Schema(
                type=types.Type.STRING,
                description="Label of the block to pick, e.g. 'blue1'.",
            ),
            "target_label": types.Schema(
                type=types.Type.STRING,
                description="Label of the reference block, e.g. 'green1'.",
            ),
            "placement_type": types.Schema(
                type=types.Type.STRING,
                description=(
                    "How to place the block: 'on_top' to stack on the target, "
                    "or 'beside' to place next to the target."
                ),
            ),
            "direction": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Direction relative to the target when placement_type='beside'. "
                    "One of: 'front', 'back', 'right', 'left'. Ignored for 'on_top'."
                ),
            ),
        },
    ),
)
