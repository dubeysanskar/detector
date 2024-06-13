# detector

# Video Object Detection with PaliGemma

This project demonstrates how to perform object detection on video frames using the `google/paligemma-3b-mix-224` model. The script processes an input video, detects objects specified in a prompt, and draws bounding boxes around detected objects in the output video.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
  
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/video-object-detection.git
    cd video-object-detection
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Log in to Hugging Face:
    ```sh
    !huggingface-cli login
    ```

## Usage

1. Place your input video file in the project directory and name it `input_video.mp4`. You can also change the input video filename in the script if needed.

2. Run the script:
    ```sh
    python video_object_detection.py
    ```

3. The processed video with bounding boxes will be saved as `output_video.avi`.

## Project Structure

video-object-detection

input_video.mp4 # Input video file

output_video.avi # Output video file (generated)

video_object_detection.py# Main script for object detection

requirements.txt # List of dependencies

README.md # Project README file
