# Football-Analysis

# Football Analysis Project

This project utilizes YOLOv8 for player detection, means to cluster players into two teams, and ByteTracker for player tracking in football matches. The combination of these technologies enables comprehensive analysis of player movements and calculate possession during games.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

Football Analysis Project aims to provide insightful analysis of football matches by leveraging state-of-the-art computer vision techniques. The project focuses on three main components:

1. **Player Detection**: YOLOv8, a real-time object detection system, is employed to detect players in football videos.

2. **Team Clustering**: Players detected by YOLOv8 are clustered into two teams using means clustering techniques. This step enables the segmentation of players based on their positions and movements.

3. **Player Tracking**: Supervisely ByteTracker is utilized for player tracking throughout the duration of football matches. This enables the analysis of player trajectories and interactions over time.

## Installation

To install and set up the project, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/yourusername/football-analysis-project.git
