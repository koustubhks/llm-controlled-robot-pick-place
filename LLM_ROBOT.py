import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from call_function import call_function, available_functions
from Robot_Tools.Robot_Motion_Tools import device_close

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    MODEL_ID = "gemini-2.5-flash"
    verbose = False
    max_iter = 20

    # Optional initial prompt from CLI
    initial_prompt = None
    if len(sys.argv) >= 2:
        initial_prompt = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == "--verbose":
        verbose = True

    system_prompt = """
    You are a helpful AI coding agent.
    You are controlling a robot called Dobot.
    You have all the tools to control and move the robot, including connecting to it.
    
    Every time you start or a new prompt is given which requires to pick and place some blocks, do these
    actions:
    Default Actions: 
    1) Move to home position
    2) Open the camera and take the frame, using your default wait time of 10 sec and updated the capture_scene.json file
    3) Give me the output as what has been detected:
        blue1 at (18, 362)
        blue2 at (108, 309)
        blue3 at (67, 16)
        blue4 at (19, 32)
        green1 at (499, 285)
        yellow1 at (339, 266)
    4) Close the camera and ask the user which block has to be moved to what place.

    When the user asks a command like 'move home', you must connect to the robot,
    move to home, and then return that the action was executed.

    When prompted to pick and place some blocks, you must take into account the following steps:

    1) If user has provided the pick up block name and place block name, then continue, else:
        1.1) Ask the user to give any of the two missing information.
        1.2) The user can only tell to move to a particular block, when asked again, in that case
             just move to that block. 
    2) Take these two pick and place bloack names, access the capture_scene.json file saved from the camera 
        and get the pixel location.
    3) Pass these into the pick_place tool function to perform the action.
    4) Print out the executed task completion.
    5) Then wait for new command, if it is pick command then go default actions and repeat.
    # ADD TO THE END OF system_prompt
    6) After each pick-and-place task, update the memory dictionary with the new position of the moved block.
       This memory should persist and reflect where each block currently is.
       Use the following format to store memory in a separate file, for example: memory.json

      {
       "blue1": [100, 320],
       "yellow1": [340, 280],
       "green1": [499, 285]
      }

    7) If the user later says something like "move blue1 back" or "place it near its original position", you should
       look into the memory file to recall where the block was previously, and execute the task accordingly.

    8) Always update memory.json after every block move, even if it's to the same position.

    """

    print("\n================ SYSTEM PROMPT ================\n")
    print(system_prompt.strip(), "\n")

    # Conversation history (user + assistant + tool messages)
    messages = []

    # If we got an initial CLI prompt, use it as the first user message
    if initial_prompt:
        print("\n================ USER PROMPT (CLI) ================\n")
        print(initial_prompt)
        messages.append(
            types.Content(role="user", parts=[types.Part(text=initial_prompt)])
        )
    else:
        # Otherwise, ask interactively for the first input
        user_text = input("\nYou (type 'quit' to exit): ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            return
        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=system_prompt
    )

    func_count = 0

    # ================= INTERACTIVE CONVERSATION LOOP =================
    while True:
        # For each user message, allow multiple tool/model turns
        for i in range(max_iter):
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=messages,
                config=config
            )

            # ------------ MODEL TEXT ------------
            if response.text:
                print("\n================ MODEL TEXT RESPONSE ================\n")
                print(response.text)

            if verbose and response.usage_metadata:
                print(f'prompt = {messages[-1].parts[0].text if messages else ""}')
                print(f'Response = {response.text}')
                print(f'Prompt Token = {response.usage_metadata.prompt_token_count}')
                print(f'Response Token = {response.usage_metadata.candidates_token_count}')

            # Add assistant content to history
            if response.candidates:
                for candidate in response.candidates:
                    if candidate and candidate.content:
                        messages.append(candidate.content)

            # ------------ TOOL CALLS ------------
            if response.function_calls:
                for function_call_part in response.function_calls:
                    func_count += 1
                    fname = getattr(function_call_part, "name", None)
                    fargs = getattr(function_call_part, "args", {})

                    print(f"\n================ FUNCTION CALL #{func_count} ================\n")
                    print(f"Function name: {fname}")
                    print("Arguments (tool prompt):")
                    try:
                        print(json.dumps(fargs, indent=2))
                    except TypeError:
                        print(fargs)

                    # Run tool
                    result = call_function(function_call_part, verbose=True)

                    print(f"\n================ FUNCTION RESULT #{func_count} ================\n")
                    print(result)

                    # Append tool result so the model can see it next iteration
                    messages.append(result)

                # continue inner for-loop to let the model react to the tool results
                continue

            # ---------- NO FUNCTION CALLS -> END OF THIS TURN ----------
            break  # break out of the max_iter loop; ready for next user input

        # ================= ASK FOR NEXT USER INPUT =================
        print("\n================ AWAITING USER INPUT (type 'quit' to exit) ================\n")
        user_text = input("You: ").strip()

        if user_text.lower() in {"quit", "exit", "q"}:
            print("Closing robot connection before exit...")
            try:
                result = device_close()
                print(result)
            except Exception as e:
                print(f"Error closing device: {e}")

            print("Exiting interactive session.")
            break
        print("\n================ USER PROMPT ================\n")
        print(user_text)

        # Add new user message and loop again
        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

if __name__ == "__main__":
    main()

