# call_function.py
from google.genai import types
from config import *

from Helper_Functions.file_handling import (
    get_files_info,
    schema_get_files_info,
    get_file_content,
    schema_get_file_content,
    run_python_file,
    schema_run_python_file,
    write_file,
    schema_write_file,
)

from Robot_Tools.Robot_Motion_Tools import (
    get_dobot_device,           schema_get_dobot_device,
    move_to_home,               schema_move_to_home,
    move_to_specific_position,  schema_move_to_specific_position,
    get_current_pose,           schema_get_current_pose,
    suction_on,                 schema_suction_on,
    suction_off,                schema_suction_off,
    set_affine_matrix,          schema_set_affine_matrix,
    move_robot_point_above,     schema_move_robot_point_above,
    move_robot_point_block,     schema_move_robot_point_block,
    update_scene_memory,        schema_update_scene_memory,
)

from Robot_Tools.Pick_Place_Tool import pick_and_place_block,schema_pick_and_place_block

from Robot_Tools.Camera_Capture_Tools import capture_scene_with_detection, schema_capture_scene_with_detection

# This is what you pass to Gemini as tools:
available_functions = types.Tool(
    function_declarations=[
        # FILES
        schema_get_files_info,
        schema_get_file_content,
        schema_run_python_file,
        schema_write_file,
        # ROBOT
        schema_get_dobot_device,
        schema_move_to_home,
        schema_move_to_specific_position,
        schema_get_current_pose,
        schema_suction_on,
        schema_suction_off,
        schema_set_affine_matrix,
        schema_move_robot_point_above,
        schema_move_robot_point_block,
        schema_update_scene_memory,
        # CAMERA CAPTURE
        schema_capture_scene_with_detection,
        # PICK AND PLACE
        schema_pick_and_place_block
    ]
)


# Names of tools that need working_directory injected
FILE_FUNCS = {"get_files_info", "get_file_content", "run_python_file", "write_file"}


def call_function(function_call_part, verbose=False):
    if verbose:
        print(
            f" - Calling function: {function_call_part.name},({function_call_part.args})"
        )
    else:
        print(f" - Calling function: {function_call_part.name}")

    # Map tool names (from FunctionDeclaration.name) to real Python functions
    function_map = {
    # FILES 
    "get_files_info": get_files_info,
    "get_file_content": get_file_content,
    "run_python_file": run_python_file,
    "write_file": write_file,
    # ROBOT
    "get_dobot_device": get_dobot_device,
    "move_to_home": move_to_home,
    "move_to_specific_position": move_to_specific_position,
    "get_current_pose": get_current_pose,
    "suction_on": suction_on,
    "suction_off": suction_off,
    "set_affine_matrix": set_affine_matrix,
    "move_robot_point_above": move_robot_point_above,
    "move_robot_point_block": move_robot_point_block,
    "update_scene_memory": update_scene_memory,
    # CAMERA CAPTURE
    "capture_scene_with_detection": capture_scene_with_detection,
    # PICK AND PLACE
    "pick_and_place_block":pick_and_place_block
}
    function_name = function_call_part.name

    if function_name not in function_map:
        # This is the error you were seeing
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"error": f"Unknown function: {function_name}"},
                )
            ],
        )

    args = dict(function_call_part.args)

    # Only add working_directory for file tools
    if function_name in FILE_FUNCS:
        args["working_directory"] = WORKING_DIR

    # Call the actual Python function
    function_result = function_map[function_name](**args)

    # Wrap result back to the model as a tool response
    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_name,
                response={"result": function_result},
            )
        ],
    )
