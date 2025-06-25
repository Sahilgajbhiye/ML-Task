# Player Re-Identification in Sports Footage

This repository contains the solution for the Liat.ai AI Intern assignment on player re-identification in sports footage.

## Task Chosen: Option 2: Re-Identification in a Single Feed

**Objective:** Given a 15-second video (15sec_input_720p.mp4), identify each player and ensure that players who go out of frame and reappear are assigned the same identity as before.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    This project uses `ultralytics` for YOLOv8 object detection and tracking, and `opencv-python` for video processing.
    ```bash
    pip install ultralytics opencv-python
    ```

3.  **Place the video file:**
    Ensure the `15sec_input_720p.mp4` video file is placed in the root directory of the project.

## How to Run

To run the player re-identification solution, execute the `reid_solution.py` script:

```bash
python3 reid_solution.py
```

This will open a video window displaying the processed video with player IDs. Press `q` to quit the video playback.

## Evaluation

To run the evaluation script, execute the `evaluate_reid.py` script:

```bash
python3 evaluate_reid.py
```

This script will process the video and print out the total frames processed, total unique players identified, and estimated re-identification events.

