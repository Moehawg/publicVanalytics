"""
This module implements tracking and trajectory refinement for volleyball action recognition.
It leverages a YOLO model for detection, tracks detected points over time, and applies interpolation,
smoothing, and various plotting functions to analyze the results.
"""

import math
import pickle
from itertools import groupby

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from ultralytics import YOLO

# Define status constants for tracking
STATUS_INIT = 0
STATUS_STATIC = 1
STATUS_DIRECTED = 2

# Matching radius (pixels) and maximum age (frames) for a track to remain active
R = 60       # Matching radius
max_age = 4  # Maximum frames to keep a track active without an update

def pt_dist(x1, y1, x2, y2):
    """
    Compute the Euclidean distance between two points.

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: Euclidean distance.
    """
    dx = x1 - x2
    dy = y1 - y2
    return math.hypot(dx, dy)

class Track:
    """
    Represents a tracked object by storing its sequence of positions,
    corresponding frame indices, and status updates over time.
    """
    counter = 1  # Class-level counter to assign unique IDs

    def __init__(self, x, y, frame_id):
        """
        Initialize a new track with the initial position.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            frame_id (int): Frame index of the initial detection.
        """
        self.id = Track.counter
        Track.counter += 1
        self.points = [[x, y]]
        self.frames = [frame_id]
        self.statuses = [STATUS_INIT]
        self.status = STATUS_INIT
        self.age = frame_id

    def is_close(self, x, y):
        """
        Determine whether a new point (x, y) is within the matching radius
        of the last point in the track.

        Args:
            x (float): x-coordinate of the new point.
            y (float): y-coordinate of the new point.

        Returns:
            (bool, float): Tuple with a boolean indicating if the new point
                           is close enough and the computed distance.
        """
        distance = pt_dist(self.points[-1][0], self.points[-1][1], x, y)
        return distance < R, distance

    def update_track(self, x, y, frame_id):
        """
        Append a new detection to the track and update its status based on movement.

        Args:
            x (float): x-coordinate of the new detection.
            y (float): y-coordinate of the new detection.
            frame_id (int): Current frame index.
        """
        self.points.append([x, y])
        self.frames.append(frame_id)
        self.age = frame_id

        if len(self.points) > 2:
            # Compute differences between consecutive points
            dx1 = self.points[-2][0] - self.points[-3][0]
            dy1 = self.points[-2][1] - self.points[-3][1]
            dx2 = self.points[-1][0] - self.points[-2][0]
            dy2 = self.points[-1][1] - self.points[-2][1]

            # Compute distances between points
            d1 = pt_dist(self.points[-2][0], self.points[-2][1],
                         self.points[-1][0], self.points[-1][1])
            d2 = pt_dist(self.points[-3][0], self.points[-3][1],
                         self.points[-2][0], self.points[-2][1])

            # Update status based on directional consistency and movement thresholds
            if dx1 * dx2 > 0 and dy1 * dy2 > 0 and d1 > 5 and d2 > 5:
                self.status = STATUS_DIRECTED
            elif self.status != STATUS_DIRECTED:
                self.status = STATUS_STATIC
        else:
            self.status = STATUS_INIT
        self.statuses.append(self.status)

def find_matching_track(x, y, frame_id, tracks):
    """
    Find an existing track that is close enough to the new detection point.

    Args:
        x (float): x-coordinate of the detection.
        y (float): y-coordinate of the detection.
        frame_id (int): Current frame index.
        tracks (list): List of existing Track objects.

    Returns:
        Track or None: The matching track if found; otherwise, None.
    """
    possible_tracks = []
    for track in tracks:
        # Consider only recently updated tracks
        if frame_id - track.age < max_age:
            is_close, distance = track.is_close(x, y)
            if is_close:
                possible_tracks.append((track, distance))

    if not possible_tracks:
        return None

    # Return the track with the smallest distance to the new detection
    possible_tracks.sort(key=lambda t: t[1])
    return possible_tracks[0][0]

# Initialize the YOLO model (only used if read_from_stub is False)
model = YOLO("models/yolov5RealBallTracker/best.pt")

def process_video(video_path, read_from_stub=False, pkl_file_path=None):
    """
    Process a video file to track detected objects and generate refined trajectories.

    This function performs detection using a YOLO model, associates detections into tracks,
    interpolates and smooths the resulting trajectories, and finally processes the trajectories
    (e.g., classification and local minima detection).

    Args:
        video_path (str): Path to the input video file.
        read_from_stub (bool): If True, load pre-computed predictions from a pickle file.
        pkl_file_path (str): Path to the pickle file with stored predictions (used when read_from_stub is True).

    Returns:
        None
    """
    # Retrieve video frame rate
    vs = cv.VideoCapture(video_path)
    fps = vs.get(cv.CAP_PROP_FPS)
    vs.release()

    if read_from_stub and pkl_file_path is not None:
        # Load pre-computed predictions
        with open(pkl_file_path, 'rb') as f:
            all_points = pickle.load(f)
    else:
        vs = cv.VideoCapture(video_path)
        tracks = []
        frame_id = 0

        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                break

            # Perform detection using YOLO
            results = model(frame)
            detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes

            # Compute center points of detections
            centers = []
            for box in detections:
                x1, y1, x2, y2 = box
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                centers.append((x, y))

            # Associate detections with existing tracks or create new tracks
            for (x, y) in centers:
                matched_track = find_matching_track(x, y, frame_id, tracks)
                if matched_track is None:
                    new_track = Track(x, y, frame_id)
                    tracks.append(new_track)
                else:
                    matched_track.update_track(x, y, frame_id)

            frame_id += 1

        # Compile predictions from all tracks
        all_points = []
        for track in tracks:
            for i in range(len(track.points)):
                x, y = track.points[i]
                frame_num = track.frames[i]
                status = track.statuses[i]
                all_points.append({'x': x, 'y': y, 'frame': frame_num, 'status': status})

        # Save predictions to a pickle file
        with open('hembachaltpredictions.pkl', 'wb') as f:
            pickle.dump(all_points, f)

    # Interpolate directed predictions
    trajectories = interpolate_directed_predictions(all_points, fps)
    
    # Determine y-axis bounds for further processing
    y_bounds = find_y_bounds(trajectories)
    y2 = y_bounds[1]
    print(f"Y-Bounds: 0 to {y2}")

    # Refine trajectories by splitting at significant trend changes
    refined_trajectories = refine_rallies(trajectories, y2, fps)

    # Build cleaned predictions from refined trajectories
    cleaned_predictions = []
    for idx, traj in enumerate(refined_trajectories):
        if len(traj) < 2:
            continue  # Skip short trajectories

        frames = [p['frame'] for p in traj]
        x_positions = [p['x'] for p in traj]
        y_positions = [p['y'] for p in traj]
        times_sec = [f / fps for f in frames]

        # Sort frames and corresponding positions
        frames = np.array(frames)
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)

        sorted_indices = np.argsort(frames)
        frames_sorted = frames[sorted_indices]
        x_sorted = x_positions[sorted_indices]
        y_sorted = y_positions[sorted_indices]

        # Remove duplicate frames
        unique_frames, unique_indices = np.unique(frames_sorted, return_index=True)
        x_unique = x_sorted[unique_indices]
        y_unique = y_sorted[unique_indices]

        # Create interpolation frames (ensuring strictly increasing frame numbers)
        interp_frames = np.arange(unique_frames[0], unique_frames[-1] + 1)
        interp_times_sec = interp_frames / fps

        # Apply cubic spline interpolation
        try:
            x_spline = CubicSpline(unique_frames, x_unique)
            y_spline = CubicSpline(unique_frames, y_unique)
            interp_x = x_spline(interp_frames)
            interp_y = y_spline(interp_frames)
        except ValueError as e:
            print(f"Skipping refined trajectory {idx} due to interpolation error: {e}")
            continue

        for i in range(len(interp_frames)):
            cleaned_predictions.append({
                'time': interp_times_sec[i],
                'x': interp_x[i],
                'y': interp_y[i],
                'traj_id': idx
            })
    
    # Smooth the cleaned predictions using a Savitzky-Golay filter
    smoothed_predictions = smooth_cleaned_predictions(cleaned_predictions)

    # Classify rallies based on the smoothed predictions and y-bounds
    rally_classifications = classify_rallies(smoothed_predictions, y_bounds)

    # Process local minima sequences and other post-processing (dummy functions provided below)
    local_minima = find_local_minima_rally0(smoothed_predictions, fps, rally_classifications)
    process_local_minima_sequences(local_minima, smoothed_predictions, rally_classifications, fps)

def interpolate_directed_predictions(all_points, fps):
    """
    Interpolate and group directed predictions into trajectories.

    Args:
        all_points (list): List of dictionaries with keys 'x', 'y', 'frame', and 'status'.
        fps (float): Frames per second of the video.

    Returns:
        list: List of trajectories (each trajectory is a list of point dictionaries).
    """
    directed_points = [p for p in all_points if p['status'] == STATUS_DIRECTED]
    if not directed_points:
        return []

    directed_points.sort(key=lambda p: p['frame'])
    trajectories = []
    current_traj = [directed_points[0]]
    for i in range(1, len(directed_points)):
        time_diff_sec = (directed_points[i]['frame'] - directed_points[i-1]['frame']) / fps
        if time_diff_sec > 2.0:  # Gap larger than 2 seconds indicates a new trajectory
            trajectories.append(current_traj)
            current_traj = [directed_points[i]]
        else:
            current_traj.append(directed_points[i])
    trajectories.append(current_traj)

    return trajectories

def smooth_cleaned_predictions(cleaned_predictions):
    """
    Smooth the cleaned predictions using a Savitzky-Golay filter.

    Args:
        cleaned_predictions (list): List of dictionaries with keys 'time', 'x', 'y', and 'traj_id'.

    Returns:
        list: Smoothed predictions sorted by time.
    """
    if not cleaned_predictions:
        print("No predictions to smooth.")
        return []

    times = np.array([p['time'] for p in cleaned_predictions])
    x_positions = np.array([p['x'] for p in cleaned_predictions])
    y_positions = np.array([p['y'] for p in cleaned_predictions])
    traj_ids = np.array([p['traj_id'] for p in cleaned_predictions])

    unique_traj_ids = np.unique(traj_ids)
    smoothed_predictions = []

    for traj_id in unique_traj_ids:
        idxs = np.where(traj_ids == traj_id)[0]
        traj_times = times[idxs]
        traj_x = x_positions[idxs]
        traj_y = y_positions[idxs]

        # Ensure window length is odd and at least 3
        window_length = min(5, len(traj_y) if len(traj_y) % 2 != 0 else len(traj_y) - 1)
        if window_length < 3:
            window_length = 3

        try:
            smoothed_y = savgol_filter(traj_y, window_length=window_length, polyorder=2)
        except ValueError:
            smoothed_y = traj_y

        for i, index in enumerate(idxs):
            smoothed_predictions.append({
                'time': times[index],
                'x': x_positions[index],
                'y': smoothed_y[i],
                'traj_id': traj_id
            })

    smoothed_predictions.sort(key=lambda x: x['time'])
    return smoothed_predictions

def find_y_bounds(trajectories):
    """
    Calculate y-bounds based on the y-positions from interpolated trajectories.

    Args:
        trajectories (list): List of trajectories, each a list of point dictionaries.

    Returns:
        tuple: (0, y2) where y2 is the y-value at which 60% of the data falls below.
    """
    y_values = [point['y'] for traj in trajectories for point in traj]
    total_predictions = len(y_values)

    if total_predictions == 0:
        print("No y-values found in the provided trajectories.")
        return (0, 0)

    counts, bin_edges = np.histogram(y_values, bins=100, range=(min(y_values), max(y_values)))
    cumulative_counts = np.cumsum(counts)
    threshold = 0.60 * total_predictions
    index = np.searchsorted(cumulative_counts, threshold)

    if index >= len(bin_edges) - 1:
        y2 = bin_edges[-1]
    else:
        y2 = bin_edges[index + 1]

    return (0, y2)

def classify_rallies(smoothed_predictions, y_bounds):
    """
    Classify rallies based on the proportion of points within specified y-bounds.

    Args:
        smoothed_predictions (list): List of prediction dictionaries.
        y_bounds (tuple): Tuple (y_lower, y_upper) used as a threshold.

    Returns:
        list: List of dictionaries containing rally classification details.
    """
    y_lower, y_upper = y_bounds
    smoothed_predictions.sort(key=lambda x: x['traj_id'])
    grouped = groupby(smoothed_predictions, key=lambda x: x['traj_id'])

    rally_classifications = []

    for traj_id, points in grouped:
        points = list(points)
        total_points = len(points)
        points_in_bounds = [p for p in points if y_lower <= p['y'] <= y_upper]
        percent_in_bounds = (len(points_in_bounds) / total_points) * 100 if total_points > 0 else 0

        classification = "Play" if percent_in_bounds >= 70 else "No Play"

        start_time = points[0]['time']
        end_time = points[-1]['time']

        # Override classification for very short rallies
        if (end_time - start_time) < 0.5:
            classification = "No Play"

        rally_classifications.append({
            'traj_id': traj_id,
            'classification': classification,
            'start_time': start_time,
            'end_time': end_time
        })

    return rally_classifications

def refine_rallies(trajectories, y2, fps, window_size=60, min_segment_fraction=0.2):
    """
    Refine trajectories by splitting them at significant trend change points.

    Args:
        trajectories (list): List of trajectories (each a list of point dictionaries with 'frame' and 'y').
        y2 (float): Y-value threshold used for detecting transitions.
        fps (float): Frames per second.
        window_size (int): Number of points on each side for local averaging.
        min_segment_fraction (float): Minimum fraction of the total length required for a valid segment.

    Returns:
        list: Refined trajectories (list of trajectory segments).
    """
    refined_trajectories = []

    for idx, traj in enumerate(trajectories):
        y_positions = [p['y'] for p in traj]
        crossing_indices = np.where(np.diff(np.signbit(np.array(y_positions) - y2)))[0]
        trend_changes = []

        if len(crossing_indices) > 0:
            for ci in crossing_indices:
                start_idx = max(0, ci - window_size)
                end_idx = min(len(y_positions), ci + window_size)

                before = y_positions[start_idx: ci + 1]
                after = y_positions[ci + 1: end_idx]

                if not before or not after:
                    continue

                mean_before = np.mean(before)
                mean_after = np.mean(after)

                if mean_before > y2 and mean_after < y2:
                    crossing_type = 'above_to_below'
                elif mean_before < y2 and mean_after > y2:
                    crossing_type = 'below_to_above'
                else:
                    continue

                trend_change_index = ci + 1
                total_length = len(traj)
                pre_segment_length = trend_change_index

                if pre_segment_length < total_length * min_segment_fraction:
                    continue

                trend_changes.append((trend_change_index, crossing_type))

            crossings_above_to_below = [tc for tc in trend_changes if tc[1] == 'above_to_below']
            crossings_below_to_above = [tc for tc in trend_changes if tc[1] == 'below_to_above']

            if crossings_above_to_below:
                last_above_to_below = crossings_above_to_below[-1]
                crossings_above_to_below = [last_above_to_below]

            all_trend_changes = crossings_above_to_below + crossings_below_to_above
            all_trend_changes.sort(key=lambda x: x[0])

            if all_trend_changes:
                split_indices = [tc[0] for tc in all_trend_changes]
                prev_index = 0
                for split_index in split_indices:
                    traj_segment = traj[prev_index:split_index]
                    if traj_segment:
                        refined_trajectories.append(traj_segment)
                    prev_index = split_index
                traj_segment = traj[prev_index:]
                if traj_segment:
                    refined_trajectories.append(traj_segment)
            else:
                refined_trajectories.append(traj)
        else:
            refined_trajectories.append(traj)

    return refined_trajectories

def plot_refined_trajectories(refined_trajectories, fps=30):
    """
    Plot each refined trajectory segment on a separate figure.

    Args:
        refined_trajectories (list): List of trajectory segments.
        fps (float): Frames per second for time conversion.
    """
    if not refined_trajectories:
        print("No refined trajectories to plot.")
        return

    # Determine overall y-axis limits for consistency across plots
    all_y = [p['y'] for segment in refined_trajectories for p in segment]
    if not all_y:
        print("No valid y-values to plot in refined trajectories.")
        return

    y_min = min(all_y)
    y_max = max(all_y)

    for idx, segment in enumerate(refined_trajectories):
        if not segment:
            continue

        frames = [p['frame'] for p in segment]
        times = [f / fps for f in frames]
        y_positions = [p['y'] for p in segment]

        plt.figure(figsize=(8, 4))
        plt.plot(times, y_positions, '-o', color='blue', markersize=1, label=f"Segment {idx}")
        plt.legend()
        plt.title(f"Refined Trajectory Segment {idx}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.ylim(y_max, y_min)
        if times:
            min_time = min(times)
            max_time = max(times)
            plt.xticks(np.arange(int(min_time), int(max_time) + 1, 1))
        output_filename = f"graphs/refined_segment_{idx}.png"
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved refined segment plot: {output_filename}")

def plot_all_predictions(all_points, fps):
    """
    Plot ball predictions over time, categorized by status.

    Args:
        all_points (list): List of prediction dictionaries with 'frame', 'y', and 'status'.
        fps (float): Frames per second of the video.
    """
    time_directed, y_directed = [], []
    time_static, y_static = [], []
    time_random, y_random = [], []

    for point in all_points:
        frame_num = point['frame']
        time_sec = frame_num / fps
        y = point['y']
        status = point['status']
        if status == STATUS_DIRECTED:
            time_directed.append(time_sec)
            y_directed.append(y)
        elif status == STATUS_STATIC:
            time_static.append(time_sec)
            y_static.append(y)
        else:
            time_random.append(time_sec)
            y_random.append(y)

    plt.figure(figsize=(12, 6))
    plt.scatter(time_random, y_random, color='grey', label='Random', s=5, alpha=0.5)
    plt.scatter(time_static, y_static, color='green', label='Static', s=5, alpha=0.5)
    plt.scatter(time_directed, y_directed, color='blue', label='Directed', s=5, alpha=0.5)
    plt.legend()
    plt.title('Ball Predictions')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Y Position')
    plt.gca().invert_yaxis()

    ax = plt.gca()
    max_time = max(time_directed + time_static + time_random) if (time_directed + time_static + time_random) else 0
    max_time = int(max_time) + 1

    major_ticks = np.arange(0, max_time + 10, 10)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    minor_ticks = np.arange(0, max_time + 5, 5)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    ax.grid(which='both', axis='x', linestyle='--', alpha=0.7)
    plt.xticks(major_ticks)
    plt.savefig('graphs/ball_predictions.png')
    plt.close()

def plot_cleaned_predictions(cleaned_predictions):
    """
    Plot cleaned ball trajectories for each trajectory ID.

    Args:
        cleaned_predictions (list): List of prediction dictionaries with 'time', 'y', and 'traj_id'.
    """
    if not cleaned_predictions:
        print("No directed predictions to plot.")
        return

    y_values = [p['y'] for p in cleaned_predictions]
    y_min = min(y_values)
    y_max = max(y_values)

    cleaned_predictions.sort(key=lambda x: x['traj_id'])
    grouped = groupby(cleaned_predictions, key=lambda x: x['traj_id'])

    for traj_id, points in grouped:
        points = list(points)
        times = [p['time'] for p in points]
        y_positions = [p['y'] for p in points]

        plt.figure(figsize=(8, 4))
        plt.plot(times, y_positions, '-o', color='blue', markersize=1, label=f'Trajectory {traj_id}')
        plt.legend()
        plt.title(f'Cleaned Ball Trajectory {traj_id}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.ylim(y_max, y_min)
        min_time = min(times)
        max_time = max(times)
        plt.xticks(np.arange(int(min_time), int(max_time) + 1, 1))
        plt.savefig(f'graphs/cleaned_trajectory_{traj_id}.png')
        plt.close()

def plot_cleaned_trajectories_xy(cleaned_predictions):
    """
    Plot ball trajectories in the X-Y plane for each trajectory ID.

    Args:
        cleaned_predictions (list): List of prediction dictionaries with 'x', 'y', and 'traj_id'.
    """
    if not cleaned_predictions:
        print("No directed predictions to plot.")
        return

    cleaned_predictions.sort(key=lambda x: x['traj_id'])
    grouped = groupby(cleaned_predictions, key=lambda x: x['traj_id'])

    for traj_id, points in grouped:
        points = list(points)
        x_positions = [p['x'] for p in points]
        y_positions = [p['y'] for p in points]

        plt.figure(figsize=(8, 6))
        plt.plot(x_positions, y_positions, '-o', color='red', markersize=3, label=f'Trajectory {traj_id}')
        plt.legend()
        plt.title(f'Ball Trajectory {traj_id} (X vs Y)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.savefig(f'graphs/ball_trajectory_xy_{traj_id}.png')
        plt.close()

def plot_cleaned_predictions_overall(cleaned_predictions, fps):
    """
    Plot overall cleaned ball trajectories over time.

    Args:
        cleaned_predictions (list): List of prediction dictionaries with 'time' and 'y'.
        fps (float): Frames per second.
    """
    if not cleaned_predictions:
        print("No directed predictions to plot.")
        return

    plt.figure(figsize=(15, 6))
    cleaned_predictions.sort(key=lambda x: x['time'])
    times = [p['time'] for p in cleaned_predictions]
    y_positions = [p['y'] for p in cleaned_predictions]
    plt.plot(times, y_positions, '-o', color='blue', markersize=2, linewidth=1)
    plt.title('Cleaned Ball Trajectories Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Y Position')
    plt.gca().invert_yaxis()

    ax = plt.gca()
    max_time = max(times) if times else 0
    max_time = int(max_time) + 1
    major_ticks = np.arange(0, max_time + 10, 10)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    minor_ticks = np.arange(0, max_time + 5, 5)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    ax.grid(which='both', axis='x', linestyle='--', alpha=0.7)
    plt.xticks(major_ticks)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graphs/overall_cleaned_predictions.png')
    plt.close()

def plot_velocity(cleaned_predictions):
    """
    Plot the velocity of ball trajectories for each trajectory ID.

    Args:
        cleaned_predictions (list): List of prediction dictionaries with 'time', 'y', and 'traj_id'.
    """
    if not cleaned_predictions:
        print("No predictions to plot velocities.")
        return

    velocities_all = []
    traj_velocities = {}
    cleaned_predictions.sort(key=lambda x: x['traj_id'])
    grouped = groupby(cleaned_predictions, key=lambda x: x['traj_id'])

    for traj_id, points in grouped:
        points = list(points)
        times = np.array([p['time'] for p in points])
        y_positions = np.array([p['y'] for p in points])

        # Calculate velocity as the derivative of y with respect to time
        dy = np.diff(y_positions)
        dt = np.diff(times)
        velocities = dy / dt

        velocities_all.extend(velocities)
        traj_velocities[traj_id] = (times[:-1], velocities)

    if velocities_all:
        v_min = min(velocities_all)
        v_max = max(velocities_all)
    else:
        v_min, v_max = -1, 1

    for traj_id, (times, velocities) in traj_velocities.items():
        if len(times) == 0:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(times, velocities, '-o', color='orange', markersize=1, label=f'Trajectory {traj_id} Velocity')
        plt.legend()
        plt.title(f'Velocity of Ball Trajectory {traj_id}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (units/sec)')
        plt.grid(True)
        plt.ylim(v_min, v_max)
        min_time = min(times)
        max_time = max(times)
        plt.xticks(np.arange(int(min_time), int(max_time) + 1, 1))
        plt.tight_layout()
        plt.savefig(f'graphs/trajectory_velocity_{traj_id}.png')
        plt.close()

def plot_y_value_histogram(smoothed_predictions):
    """
    Plot a histogram of y-values from the smoothed predictions.

    Args:
        smoothed_predictions (list): List of prediction dictionaries containing 'y'.
    """
    if not smoothed_predictions:
        print("No predictions to plot histogram.")
        return

    y_values = [p['y'] for p in smoothed_predictions]
    total_predictions = len(y_values)
    y_in_range = [y for y in y_values if 0 <= y <= 500]
    total_in_range = len(y_in_range)
    percentage_in_range = (total_in_range / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Total predictions: {total_predictions}")
    print(f"Total predictions in the range 0-500: {total_in_range}")
    print(f"Percentage of predictions in the range 0-500: {percentage_in_range:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.hist(y_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Y-Values')
    plt.xlabel('Y Position')
    plt.ylabel('Frequency')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('graphs/y_value_histogram.png')
    plt.close()

# Placeholder functions for missing implementations (The real implementations of these functions will not be published for privacy reasons)
def find_local_minima_rally0(smoothed_predictions, fps, rally_classifications):
    """
    Placeholder for local minima detection in a rally.
    Replace with the actual implementation.
    
    Args:
        smoothed_predictions (list): List of smoothed prediction dictionaries.
        fps (float): Frames per second.
        rally_classifications (list): Rally classification results.

    Returns:
        list: Detected local minima (empty list for placeholder).
    """
    return []
# Placeholder for the most important function of the code. It returns all actions with labels as a JSON .txt file
def process_local_minima_sequences(local_minima, smoothed_predictions, rally_classifications, fps):
    """
    Placeholder for processing local minima sequences.
    Replace with the actual implementation.
    
    Args:
        local_minima (list): Detected local minima.
        smoothed_predictions (list): List of smoothed prediction dictionaries.
        rally_classifications (list): Rally classification results.
        fps (float): Frames per second.
    """
    pass

