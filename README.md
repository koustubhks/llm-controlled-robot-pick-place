# LLM-Controlled Agentic Robot for Pick and Place

Natural-language robot control project that connects an LLM planner with computer vision and Dobot robot arm tools.

## Overview

The system accepts natural language commands, lets the language model select the appropriate tool, captures the workspace, and executes pick-and-place primitives through robot control functions.

Example command:

```text
Pick up the green block and place it on the left side of the workspace.
```

## Architecture

```text
User command
    -> LLM planner
    -> function/tool selection
    -> camera capture and scene state
    -> robot motion primitive
    -> pick/place execution
```

## Key Features

- Gemini/GPT-style function-calling workflow.
- Modular robot tools for motion, camera capture, and pick-place operations.
- Dobot arm integration through `pydobot`.
- Workspace state stored as JSON for tool feedback.
- Safety-aware separation between language planning and robot execution primitives.

## Tech Stack

Python, Google Gemini API, OpenCV, NumPy, pydobot, dotenv, JSON tool state.

## Main Files

- `LLM_ROBOT.py`: main LLM planning loop.
- `call_function.py`: tool dispatch layer.
- `Camera_Capture_Tools.py`: camera capture and scene perception helper.
- `Pick_Place_Tool.py`: pick-and-place command schema and execution.
- `Robot_Motion_Tools.py`: low-level Dobot motion helpers.
- `config.py`: workspace and hardware configuration.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set your own API key in `.env`:

```text
GEMINI_API_KEY=your_key_here
```

## Safety

Run robot commands at low speed first, keep the workspace clear, and keep the power cutoff accessible. Do not directly execute unconstrained model output; route commands through tested motion primitives.

## Repository Hygiene

Generated cache files and private `.env` files are intentionally excluded.

