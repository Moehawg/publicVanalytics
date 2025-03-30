# publicVanalytics
This repository contains a portion of the code used (or intended to be used) in the development of the Vanalytics Platform — an AI-powered system for automated volleyball analytics based on action recognition models applied to volleyball match videos.

The purpose of this repository is to support and inspire research and development efforts in the field of sports action recognition, while preserving the privacy and integrity of the core technologies that drive Vanalytics.

To maintain the competitive edge and proprietary nature of our platform, datasets, specific model architectures, and internally developed algorithms that are critical or unique to Vanalytics will not be published in this repository.

## Contents

- `rally_detection/`  
  Code for automatically splitting a full volleyball match video into individual rallies.

- `homography/`  
  Tools to transform player and ball coordinates from camera view to a 2D minimap using homography.

- `player_identification/`  (Needs to be revised before publication; ETA around June 2026)
  Scripts for identifying players using individual features and OCR-based jersey number recognition.

- `court_line_detection/`  (Needs to be revised before publication; ETA around June 2026)
  Code for detecting volleyball court lines from video frames.

- `action_classification/`  
  A casestudy with a lightweight ML model to explore possibiliities for classifying the type of reception (e.g., upper hand vs. lower hand).

Each module contains its own README with a short explanation and usage instructions.

## Exclusions

To maintain the privacy and uniqueness of Vanalytics, this repository does **not** include:
- Full datasets
- End-to-end model training pipelines
- Custom model architectures
- Proprietary postprocessing algorithms
