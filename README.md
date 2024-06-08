# Football Analysis System

This project uses machine learning, computer vision, and deep learning techniques to create a football analysis system. The main goals are:

1. Detect and track players, referees, and the ball across video frames using object detection and tracking algorithms.
2. Train a custom object detector to enhance the output of state-of-the-art models.
3. Assign players to teams based on the color of their jerseys using K-means clustering for pixel segmentation.
4. Analyze ball possession and control by assigning the ball to the closest player.
5. Annotate output video with player IDs, teams, ball possession, and other relevant information.

Example output :
![Example](https://github.com/kaustubh285/football_analysis_yolo/blob/main/example_output/output.png?raw=true)

## Installation & Setup

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Train your model using the training notebook provided
4. Download the model weights and store in models folder
5. Store the video to be analysed in the input folder

## Usage

To run the analysis pipeline:

```
python main.py
```

This will process the specified input video, perform analysis, and generate an annotated output video in the output folder.

## Project Structure

- `main.py`: The main entry point that orchestrates the analysis pipeline.
- `utils/`: Contains utility functions for reading/writing videos and analysing them.
  - `bbox_utils.py`
  - `helper.py`
  - `video.py`
- `tracking/`: Contains implementation of object tracking algorithms.
  - `tracker.py`
- `player_ball_assigner/`: Contains logic for assigning the ball to the closest player.
  - `player_ball_assigner.py`
- `team_assigner/`: Contains algorithms for assigning players to teams based on jersey color.
  - `team_assigner.py`
- `input/`: Directory for input video files.
- `output/`: Directory for annotated output videos.
- `models/`: Directory for storing trained models.
- `stubs/`: Directory for storing pre-computed tracking data (optional).
- `development_and_analysis/`: Contains notebooks for testing and analysis.
  - `color_assignment.ipynb`
- `training/`: Contains model training notebook to be used separately from the project to generate weights of the model.
  - `football_training.ipynb`

## Pipeline Overview

1. **Object Detection and Tracking**: The `Tracker` class utilizes a pre-trained object detection model to detect players, referees, and the ball in each frame. It then tracks these objects across frames using tracking algorithms.

2. **Ball Interpolation**: The ball might not be detected in some of the frames, hence, the missing ball positions are interpolated using the `interpolate_ball_positions` method to ensure smooth tracking throughout the video.

3. **Team Assignment**: The `TeamAssigner` class segments player jersey colours using K-means clustering and assigns players to teams based on their jersey colours.

4. **Ball Assignment**: The `PlayerBallAssigner` class determines the player closest to the ball and assigns ball possession accordingly.

5. **Annotation and Output**: The processed frames are annotated with player IDs, teams, ball possession, and other relevant information. The annotated frames are then compiled into an output video file.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection
- [Abdullah Tarek](https://github.com/abdullahtarek) for the project tutorial
