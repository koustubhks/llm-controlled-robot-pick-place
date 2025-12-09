import cv2
import time
import os
import json
import numpy as np
from collections import defaultdict
from google.genai import types
from config import camera_index, capture_wait_time

# ---------------------------------------------------------
# Helper: draw outlined text
# ---------------------------------------------------------
def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ---------------------------------------------------------
# Helper: detect colored blocks and annotate frame
# ---------------------------------------------------------
def detect_and_annotate_blocks(frame, area_threshold=500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "blue":   [(np.array([100, 120, 50]), np.array([130, 255, 255]))],
        "green":  [(np.array([40,  70, 50]), np.array([ 80, 255, 255]))],
        "yellow":[(np.array([20, 120, 80]), np.array([ 35, 255, 255]))],
        "red":   [
            (np.array([  0, 120, 80]), np.array([ 10, 255, 255])),
            (np.array([160, 120, 80]), np.array([179, 255, 255]))
        ],
    }

    detected_blocks = []
    color_counts = defaultdict(int)  # e.g., blue1, blue2, red1, ...

    for color_name, ranges in color_ranges.items():
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_threshold:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            # color_no style labels: red1, blue2, green1, etc.
            color_counts[color_name] += 1
            label_name = f"{color_name}{color_counts[color_name]}"

            put_text(frame, f"{label_name} ({cx},{cy})", (x, y - 10))
            detected_blocks.append((label_name, (cx, cy)))

    return detected_blocks

# ---------------------------------------------------------
# MAIN TOOL FUNCTION
# ---------------------------------------------------------
def capture_scene_with_detection(
    cam_index=camera_index,
    width: int = 640,
    height: int = 480,
    save_dir: str = "captures",
    capture_interval_sec=capture_wait_time,
):
    """
    Open the camera, show live view, detect colored blocks, annotate them,
    show a countdown from capture_interval_sec to 0, auto-save one annotated
    frame + JSON when countdown reaches 0, then keep showing the labeled feed
    until the user presses 'q' to quit.

    Returns a JSON-serializable dict with paths and detected blocks.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            return {
                "error": f"Could not open camera index {cam_index}"
            }

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = "Camera (q=quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        # start time for countdown
        start_time = time.time()
        has_saved_once = False

        img_path = None
        json_path = None
        last_detected_blocks_data = []

        while True:
            ok, frame = cap.read()
            if not ok:
                ok, frame = cap.read()
                if not ok:
                    print("Frame grab failed.")
                    break

            detected_blocks = detect_and_annotate_blocks(frame)

            # --- COUNTDOWN / STATUS TEXT ---
            if has_saved_once:
                # After capture: no timer, just info text
                put_text(frame, "Capture done — labels only (q=quit)", (10, 60))
            else:
                elapsed = time.time() - start_time
                remaining = max(0, int(capture_interval_sec - elapsed))
                put_text(
                    frame,
                    f"Auto-save in {remaining}s (q=quit)",
                    (10, 60),
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # --- AUTO-SAVE ONCE WHEN TIMER EXPIRES ---
            if (not has_saved_once) and (time.time() - start_time >= capture_interval_sec):
                has_saved_once = True

                img_path = os.path.join(save_dir, "capture_scene.png")
                json_path = os.path.join(save_dir, "capture_scene.json")

                cv2.imwrite(img_path, frame)
                print(f"Saved image: {img_path}")

                data = [
                    {"label": label, "x": cx, "y": cy}
                    for label, (cx, cy) in detected_blocks
                ]
                last_detected_blocks_data = data

                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"Saved JSON: {json_path}")

        cap.release()
        cv2.destroyAllWindows()

        # Return info to the LLM/tool caller
        if img_path is None or json_path is None:
            return {
                "message": "Camera closed before auto-save completed.",
                "image_path": img_path,
                "json_path": json_path,
                "detected_blocks": last_detected_blocks_data,
            }

        return {
            "message": "Capture completed.",
            "image_path": img_path,
            "json_path": json_path,
            "detected_blocks": last_detected_blocks_data,
        }

    except Exception as e:
        # Tool-safe error return
        return {
            "error": f"Error in capture_scene_with_detection: {e}"
        }

# ---------------------------------------------------------
# SCHEMA FOR LLM TOOL CALLING
# ---------------------------------------------------------
schema_capture_scene_with_detection = types.FunctionDeclaration(
    name="capture_scene_with_detection",
    description=(
        "Opens the camera, detects blue/green/yellow/red blocks on a black "
        "background, annotates them with labels like 'red1', 'blue2', etc., "
        "shows a countdown, saves one annotated frame and a JSON file with "
        "pixel positions, and returns the file paths and detected blocks."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "cam_index": types.Schema(
                type=types.Type.INTEGER,
                description="Camera index to open (e.g., 0, 1, 2, 4).",
            ),
            "width": types.Schema(
                type=types.Type.INTEGER,
                description="Capture width in pixels. Default 640.",
            ),
            "height": types.Schema(
                type=types.Type.INTEGER,
                description="Capture height in pixels. Default 480.",
            ),
            "save_dir": types.Schema(
                type=types.Type.STRING,
                description="Directory where capture_scene.png and capture_scene.json will be saved. Default 'captures'.",
            ),
            "capture_interval_sec": types.Schema(
                type=types.Type.NUMBER,
                description="Seconds to wait before auto-saving the frame once. Default taken from config.capture_wait_time.",
            ),
        },
    ),
)
