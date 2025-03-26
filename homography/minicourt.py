"""
Mini Court Module

This module implements a mini-court representation for volleyball analysis.
It computes a homography from real court coordinates to a scaled-down, drawn
"mini-court" on a video frame. The mini-court is used to visualize court lines,
player positions, and other court-related analytics.

Key functionalities:
- Define a drawing area (mini-court) on a given frame.
- Compute the homography transformation from real court coordinates to the mini-court.
- Draw court boundaries, net, and attack lines.
- Overlay player positions onto the mini-court.
- Optionally, generate a composite view of the mini-court on each frame.

Note: This module also includes a commented-out bird’s-eye perspective transformation
function. Although it is not used in the current pipeline (and may not work perfectly),
it is included for reference in case further investigation is needed.
"""

import cv2
import numpy as np


class MiniCourt:
    def __init__(self, frame, real_court_points):
        """
        Initialize the MiniCourt instance.

        Args:
            frame (np.array): The video frame to define the drawing area.
            real_court_points (list): List of real-world court points for homography,
                                      provided as (x, y) tuples.
        """
        # Configuration parameters for drawing the mini-court
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        # Set positions and key points for drawing the mini-court on the frame
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        # Set the real court points and compute the homography transformation matrix
        self.real_court_points = np.array(real_court_points, dtype=np.float32)
        self.compute_homography()

    def set_canvas_background_box_position(self, frame):
        """
        Determine the position of the background rectangle on the frame where the mini-court will be drawn.

        Args:
            frame (np.array): The video frame.
        """
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        """
        Set the position of the mini-court within the background rectangle,
        applying padding to the court drawing area.
        """
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def set_court_drawing_key_points(self):
        """
        Define the key points for drawing the mini-court.

        The mini-court is defined by four main points; here we choose to set
        points at the top-left, top-right, and midpoints on the left and right
        (for a two-part division). Additional key points for net and attack lines
        are also computed.
        """
        # Court dimensions in meters (used for proportional calculations)
        court_length_m = 18.0  # Total court length
        court_width_m = 9.0    # Court width
        attack_line_distance_m = 3.0  # Distance from net to attack line

        # Define the mini-court points (using a two-segment division for simplicity)
        self.mini_court_points = np.array([
            [self.court_start_x, self.court_start_y],  # Top-left
            [self.court_end_x, self.court_start_y],    # Top-right
            [self.court_start_x, self.court_start_y + self.court_drawing_height / 2],  # Middle-left
            [self.court_end_x, self.court_start_y + self.court_drawing_height / 2]     # Middle-right
        ], dtype=np.float32)

        # Create a dictionary of drawing key points for court boundaries and lines
        self.drawing_key_points = {
            'top_left': (int(self.court_start_x), int(self.court_start_y)),
            'top_right': (int(self.court_end_x), int(self.court_start_y)),
            'bottom_left': (int(self.court_start_x), int(self.court_end_y)),
            'bottom_right': (int(self.court_end_x), int(self.court_end_y))
        }

        # Net line (middle of the court)
        net_y = int(self.court_start_y + self.court_drawing_height / 2)
        self.drawing_key_points['net_left'] = (int(self.court_start_x), net_y)
        self.drawing_key_points['net_right'] = (int(self.court_end_x), net_y)

        # Compute attack line offset in pixels (based on proportional distance)
        half_court_height = self.court_drawing_height / 2
        attack_line_offset_pixels = int((attack_line_distance_m / (court_length_m / 2)) * half_court_height)

        # Define attack line key points for both the top and bottom halves of the court
        self.drawing_key_points['attack_left_top'] = (int(self.court_start_x), net_y - attack_line_offset_pixels)
        self.drawing_key_points['attack_right_top'] = (int(self.court_end_x), net_y - attack_line_offset_pixels)
        self.drawing_key_points['attack_left_bottom'] = (int(self.court_start_x), net_y + attack_line_offset_pixels)
        self.drawing_key_points['attack_right_bottom'] = (int(self.court_end_x), net_y + attack_line_offset_pixels)

    def set_court_lines(self):
        """
        Define the court lines to be drawn on the mini-court.

        The lines are represented as pairs of keys from the drawing_key_points dictionary.
        """
        self.lines = [
            ('top_left', 'top_right'),                   # Top boundary line
            ('bottom_left', 'bottom_right'),             # Bottom boundary line
            ('top_left', 'bottom_left'),                 # Left boundary line
            ('top_right', 'bottom_right'),               # Right boundary line
            ('net_left', 'net_right'),                   # Net line in the middle
            ('attack_left_top', 'attack_right_top'),     # Attack line on the top half
            ('attack_left_bottom', 'attack_right_bottom')  # Attack line on the bottom half
        ]

    def compute_homography(self):
        """
        Compute the homography matrix from the real court points to the mini-court points.

        This matrix is used to map points (e.g., player positions) from the original frame
        to the mini-court view.
        """
        self.homography_matrix, _ = cv2.findHomography(self.real_court_points, self.mini_court_points)

    def apply_homography(self, point):
        """
        Apply the computed homography to map a point from the real court to the mini-court.

        Args:
            point (tuple): A point (x, y) in the original frame.

        Returns:
            np.array: Transformed point (x, y) in the mini-court coordinate system.
        """
        point = np.array([[point]], dtype=np.float32)  # Shape (1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return transformed_point[0][0]  # Return as (x, y)

    def draw_background_rectangle(self, frame):
        """
        Draw a semi-transparent white rectangle as the background for the mini-court.

        Args:
            frame (np.array): The input video frame.

        Returns:
            np.array: The frame with the background rectangle drawn.
        """
        out = frame.copy()
        alpha = 0.5

        # Define rectangle coordinates
        x1, y1 = self.start_x, self.start_y
        x2, y2 = self.end_x, self.end_y

        # Extract region of interest (ROI) and create overlay
        roi = frame[y1:y2, x1:x2]
        overlay = np.full_like(roi, (255, 255, 255), dtype=np.uint8)

        # Blend the ROI with the overlay
        blended = cv2.addWeighted(roi, alpha, overlay, 1 - alpha, 0)
        out[y1:y2, x1:x2] = blended

        return out

    def draw_court(self, frame):
        """
        Draw the mini-court lines and key points on the given frame.

        Args:
            frame (np.array): The frame on which to draw the mini-court.

        Returns:
            np.array: The frame with the mini-court drawn.
        """
        # Draw key points (e.g., corners, net, attack lines)
        for key, (x, y) in self.drawing_key_points.items():
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Draw lines connecting the key points
        for start_key, end_key in self.lines:
            start_point = self.drawing_key_points[start_key]
            end_point = self.drawing_key_points[end_key]
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)  # Black lines

        return frame

    def draw_players_on_court(self, frame, player_positions):
        """
        Draw player positions on the mini-court.

        Args:
            frame (np.array): The frame to draw on.
            player_positions (list): List of player positions in original frame coordinates.
        """
        for pos in player_positions:
            transformed_point = self.apply_homography(pos)
            x, y = int(transformed_point[0]), int(transformed_point[1])
            # Only draw the player if within the mini-court boundaries
            if self.court_start_x <= x <= self.court_end_x and self.court_start_y <= y <= self.court_end_y:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circle

    def draw_mini_court_with_players(self, frames, player_positions_per_frame):
        """
        Overlay the mini-court and player positions onto a list of frames.

        Args:
            frames (list): List of video frames (np.array).
            player_positions_per_frame (list): List of player positions for each frame.

        Returns:
            list: List of frames with the mini-court and players drawn.
        """
        output_frames = []
        for idx, frame in enumerate(frames):
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            # Draw player positions if available for this frame
            if idx < len(player_positions_per_frame):
                self.draw_players_on_court(frame, player_positions_per_frame[idx])
            output_frames.append(frame)
        return output_frames


###############################
# Bird's-Eye Perspective Transformation (Commented Out)
###############################
# The following function performs a bird's-eye view transform on a single image.
# Although it is not fully robust and is not used in the current pipeline,
# it is included for reference or potential future use.
#
# import cv2
# import numpy as np
#
# def bird_eye_view_transform(input_image_path, output_image_path, src_points, dest_points):
#     """
#     Transform an image to a bird's-eye view using a perspective transform.
#
#     Args:
#         input_image_path (str): Path to the input image.
#         output_image_path (str): Path to save the transformed image.
#         src_points (list of tuples): Source points in the input image.
#         dest_points (np.array): Destination points defining the target perspective.
#     """
#     image = cv2.imread(input_image_path)
#     if image is None:
#         print("Error loading image!")
#         return
#
#     src_points_array = np.array(src_points, dtype="float32")
#     homography_matrix, status = cv2.findHomography(src_points_array, dest_points, cv2.RANSAC, 5.0)
#
#     height, width = 500, 1000  # Desired dimensions for the bird's-eye view image
#     transformed_image = cv2.warpPerspective(image, homography_matrix, (width, height))
#
#     cv2.imwrite(output_image_path, transformed_image)
#     print(f"Bird's-eye view image saved as {output_image_path}")
#
# # Example usage:
# real_court_points = [
#     (811, 671), (1522, 662),        # Top-left, Top-right
#     (714, 751), (1587, 740),          # 3m line other side left, 3m line other side right
#     (650, 805), (1631, 796),          # Mid line left, Mid line right
#     (560, 875), (1690, 864),          # 3m line camera side left, 3m line camera side right
#     (303, 1091), (1874, 1086)         # Bottom left, Bottom right
# ]
#
# dest_pts = np.array([
#     [100, 50], [900, 50],
#     [150, 150], [850, 150],
#     [200, 250], [800, 250],
#     [250, 350], [750, 350],
#     [300, 450], [700, 450]
# ], dtype="float32")
#
# bird_eye_view_transform('../input_video/hg.png', 'output_bird_eye.jpg', real_court_points, dest_pts)
