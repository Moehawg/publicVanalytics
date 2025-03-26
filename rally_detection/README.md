Volleyball Ball Tracking and Trajectory Analysis
Overview
This module is part of a larger volleyball action recognition pipeline. It focuses on detecting and tracking the volleyball throughout match videos and refining the resulting trajectories. Using a YOLO-based detector, the code extracts ball positions in each frame, associates detections into continuous tracks, and processes these tracks through interpolation and smoothing. Finally, it classifies segments of ball movement (rallies) and provides several visualization functions to support analysis and debugging.

Features
Detection and Tracking:
Uses a pre-trained YOLO model to detect the ball in each frame and computes the center of each detected bounding box. Detected points are grouped into continuous tracks using a custom tracking algorithm.

Trajectory Refinement:
Applies cubic spline interpolation to generate continuous trajectories from discrete track points. Further smooths trajectories using a Savitzky-Golay filter.

Rally Classification:
Segments trajectories into rally segments based on movement patterns and classifies them (e.g., as "Play" or "No Play") based on a threshold on the y-position distribution.

Visualization:
Provides multiple plotting functions to visualize raw predictions, refined trajectories, ball velocity, and histograms of ball positions, which can help in debugging and performance analysis.

Project Structure
process_video(video_path, read_from_stub=False, pkl_file_path=None)
The main function that processes a volleyball match video. It handles:

Reading the video and retrieving the frame rate.

Running the YOLO detector on each frame.

Tracking detections across frames using the Track class.

Compiling and saving predictions.

Interpolating, smoothing, and refining the trajectories.

Classifying rally segments and visualizing results.

Track Class (formerly Blob)
Represents an individual track for a detected ball. It stores:

The sequence of positions (points) over time.

Frame indices corresponding to each point.

A status history indicating the nature of the movement (initial, static, or directed).

Interpolation & Smoothing Functions:
Functions such as interpolate_directed_predictions and smooth_cleaned_predictions process raw tracking data into refined, continuous trajectories.

Rally and Trajectory Processing Functions:
Functions like refine_rallies and classify_rallies split trajectories into segments based on detected trend changes and classify these segments based on ball movement patterns.

Plotting Functions:
Several functions (e.g., plot_refined_trajectories, plot_all_predictions, plot_velocity) generate visual representations of the tracking results, aiding in both analysis and presentation.

Requirements
Python 3.x

Key Libraries:

OpenCV (cv2)

NumPy

Matplotlib

SciPy

ultralytics (for YOLO implementation)

Standard libraries: math, pickle, itertools

Pre-trained YOLO Model:
The model file is expected at models/yolov5RealBallTracker/best.pt.

Input and Output
Input
Video File:
A volleyball match video (e.g., MP4 format) provided via a file path.

Optional Pre-computed Predictions:
A pickle file containing pre-computed ball detections (used when read_from_stub is set to True).

Output
Pickle Files:
Intermediate predictions are saved (e.g., hembachaltpredictions.pkl), which can be reloaded for testing.

Visualizations:
A series of plots saved in the graphs/ directory, including:

Refined trajectory segments

Overall ball trajectories over time

Velocity plots

Histograms of ball positions

Rally Classification Data:
Rally segments are classified (e.g., "Play" vs. "No Play") based on movement criteria, and results are output for further processing.

How It Works
Ball Detection:
The YOLO model processes each video frame and outputs bounding boxes. The code computes the center point of each box as the ball's detected position.

Tracking:
Each detection is associated with an existing Track (if within a defined matching radius and recent enough). If no match is found, a new Track is created.
The Track class updates its internal state and stores a history of positions, frame indices, and status (e.g., initial, static, or directed movement).

Trajectory Extraction:
After processing all frames, the code aggregates points from all tracks and saves them as a set of raw predictions.

Interpolation and Smoothing:
Directed points (indicating consistent movement) are grouped into trajectories. Cubic spline interpolation is used to generate smooth trajectories, and a Savitzky-Golay filter is applied to smooth these further.

Rally Classification:
The refined trajectories are analyzed to detect significant trend changes, which are used to split the trajectories into segments (rallies). Each segment is then classified based on the proportion of points within specific y-value bounds.

Visualization:
Multiple plotting functions generate graphs showing the refined trajectories, velocity over time, and histograms of y-values to aid in analysis and presentation.

Usage
To process a video and generate tracking and analysis outputs, call the process_video function with the appropriate parameters. For example:

python
Copy
from your_module import process_video

# Process the video with live YOLO detections
video_path = "path/to/volleyball_match.mp4"
process_video(video_path, read_from_stub=False)

# Alternatively, to load pre-computed predictions:
# process_video(video_path, read_from_stub=True, pkl_file_path="path/to/predictions.pkl")
