# Volleyball Mini-Court Visualization and Homography Module

## Overview

This module is part of the volleyball analytics system. It implements a mini-court visualization that is drawn on video frames. The module computes a homography transformation to map real-world court coordinates to a scaled-down mini-court, allowing for an overlay of court lines, key court areas, and player positions. This facilitates a better understanding of player movements and court positioning in the context of match analysis.

Additionally, the module contains (commented-out) code for performing a bird's-eye perspective transformation on an image. Although the bird's-eye view function does not work perfectly and is not integrated into the main pipeline, it is included for reference and potential future use.

The code is build on basis of this Youtube Video which demonstrates a Tennis Minicourt example: https://www.youtube.com/watch?v=L23oIHZE14w

## Features

- **Mini-Court Drawing:**  
  Creates a designated area on the video frame where the court is drawn with boundaries, a net line, and attack lines.
  
- **Homography Computation:**  
  Computes a homography matrix to map real-world court points to mini-court coordinates, enabling accurate overlay of player positions.
  
- **Player Position Overlay:**  
  Transforms player positions from the original frame into the mini-court view and draws them onto the mini-court.
  
- **Frame Processing:**  
  Processes a sequence of video frames to overlay the mini-court and player positions consistently.
  
- **Bird’s-Eye Perspective Transformation (Commented Out):**  
  Provides an optional perspective transform function for generating a bird's-eye view of the court. This function is included for reference but is not part of the core functionality due to its imperfect performance.

## Project Structure

- **`MiniCourt` Class**  
  The core class of this module that encapsulates:
  - Calculation of the drawing area for the mini-court.
  - Computation of the homography transformation from real court points to the mini-court.
  - Methods to draw the background, court lines, key points, and overlay player positions.
  
- **Homography and Drawing Methods:**  
  Functions within the class to:
  - Set positions for the canvas and mini-court.
  - Define key court points and lines for visualization.
  - Apply the computed homography to transform points.
  
- **Bird’s-Eye Perspective Transformation (Commented Out):**  
  A standalone function is included (in commented form) that demonstrates how to convert an image to a bird's-eye view using perspective transformation.

## Requirements

- **Python 3.x**

- **Key Libraries:**
  - OpenCV (`cv2`)
  - NumPy

- **Input Data:**
  - A single video frame (or a sequence of frames) to define the drawing area.
  - A set of real-world court points (provided as a list of (x, y) tuples) used for computing the homography.

## Input and Output

### Input

- **Video Frame:**  
  An image frame (numpy array) from a volleyball match video.
  
- **Real Court Points:**  
  A list of (x, y) coordinates representing key positions on the real volleyball court used for the homography transformation.

- **Player Positions (Optional):**  
  A list of player positions (in the original frame coordinates) for each frame, which will be transformed and drawn on the mini-court.

### Output

- **Processed Frames:**  
  Frames with a semi-transparent mini-court overlay, court lines, and optionally, player positions.
  
- **Visualization:**  
  The module is intended to be used as part of a larger system. The processed frames can be displayed, saved, or further analyzed.

## How It Works

1. **Initialization:**  
   The `MiniCourt` class is instantiated with a video frame and real court points. It sets up the drawing area for the mini-court and computes the homography matrix.
   
2. **Drawing the Mini-Court:**  
   The module calculates the mini-court boundaries and key points (e.g., corners, net, attack lines). These key points are used to draw the court lines on the frame.
   
3. **Homography Application:**  
   The computed homography is applied to transform points (such as player positions) from the original frame coordinates to the mini-court coordinates.
   
4. **Overlaying Players:**  
   If provided, player positions are transformed using the homography and drawn as markers on the mini-court.
   
5. **Bird’s-Eye Perspective (Optional):**  
   The commented-out function demonstrates how to perform a bird's-eye view transformation on an image. This can be useful for other applications, though it is not fully integrated.

## Usage

To use this module, import the `MiniCourt` class from the module and instantiate it with a frame and a list of real court points. For example:

```python
from mini_court import MiniCourt
import cv2

# Load a video frame (for example, from a video capture or an image file)
frame = cv2.imread('path/to/frame.jpg')

# Define the real court points (list of (x, y) tuples)
real_court_points = [
    (811, 671), (1522, 662),        # Example points; adjust as necessary
    (714, 751), (1587, 740),
    (650, 805), (1631, 796),
    (560, 875), (1690, 864),
    (303, 1091), (1874, 1086)
]

# Initialize the MiniCourt object
mini_court = MiniCourt(frame, real_court_points)

# Optionally, overlay player positions on the mini-court
# For demonstration, assume an empty list or provide actual positions per frame
player_positions_per_frame = [[]]  # Replace with actual player positions if available

# Draw the mini-court with players on the current frame
output_frame = mini_court.draw_background_rectangle(frame)
output_frame = mini_court.draw_court(output_frame)
mini_court.draw_players_on_court(output_frame, player_positions_per_frame[0])

# Display or save the output frame
cv2.imshow("Mini-Court", output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

