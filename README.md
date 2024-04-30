# Football-Analysis

# Football Analysis Project

This project utilizes YOLOv8 for player detection, means to cluster players into two teams, and ByteTracker for player tracking in football matches. The combination of these technologies enables comprehensive analysis of player movements and calculate possession during games.

<video width="640" height="360" controls>
  <source src="output_videos\output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

Football Analysis Project aims to provide insightful analysis of football matches by leveraging state-of-the-art computer vision techniques. The project focuses on three main components:

1. **Player Detection**: YOLOv8, a real-time object detection system, is employed to detect players in football videos.

3. **Player Tracking**: Supervision ByteTracker is utilized for player tracking throughout the duration of football matches. 

2. **Team Clustering**: Players detected by YOLOv8 are clustered into two teams using *kmeans clustering algorithm*. This step enables the segmentation of players based on their shirt color using two method *foreground*, *center-box*.

3. **Player Tracking**: Supervisely ByteTracker is utilized for player tracking throughout the duration of football matches. This enables the analysis of player trajectories and interactions over time.

## Installation

To install and set up the project, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/yourusername/football-analysis-project.git
