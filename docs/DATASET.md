# Human Activity Recognition (HAR) Dataset



## Table of Contents

- [Human Activity Recognition (HAR) Dataset](#human-activity-recognition-har-dataset)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Data Collection](#2-data-collection)
    - [2.1 Participants and Experiment Setup](#21-participants-and-experiment-setup)
    - [2.2 Sensors](#22-sensors)
    - [2.3 Sampling Rate](#23-sampling-rate)
    - [2.4 Train / Test Split](#24-train--test-split)
  - [3. Activity Classes](#3-activity-classes)
  - [4. From Sensor Readings to Data Windows](#4-from-sensor-readings-to-data-windows)
    - [4.1 Axes of Measurement](#41-axes-of-measurement)
    - [4.2 The Sliding Window](#42-the-sliding-window)
    - [4.3 Overlapping Windows](#43-overlapping-windows)
    - [4.4 Raw Data per Window](#44-raw-data-per-window)
  - [5. Raw Data Samples](#5-raw-data-samples)
    - [Table A — Single Data Point at t = 20 ms](#table-a--single-data-point-at-t--20-ms)
    - [Table B — One Complete Window (128 readings)](#table-b--one-complete-window-128-readings)

---

## 1. Overview

The Human Activity Recognition (HAR) dataset is a structured collection of movement data, recorded from 30 individuals wearing a smartphone on their waist as they performed six common physical activities. It was developed to train machine learning models to automatically identify what a person is doing based solely on how their phone moves.

The dataset originates from the UCI Machine Learning Repository and has become a widely used benchmark in the field of activity recognition. Its relevance extends well beyond research: the same principles underpin fitness tracking, health monitoring applications, and smart device interactions used every day.

> **Core purpose:** Give a machine learning model enough labeled examples of human movement so that, given new sensor data from an unfamiliar person, it can correctly identify the activity being performed.

---

## 2. Data Collection

### 2.1 Participants and Experiment Setup

Thirty volunteers, aged between 19 and 48, participated in a controlled experiment. Each person wore a Samsung Galaxy S II smartphone secured to their waist, which recorded sensor data while they performed six predefined activities. To ensure labeling accuracy, the entire experiment was video-recorded and researchers reviewed the footage to manually assign the correct activity label to each moment in time.

### 2.2 Sensors

The smartphone contains two built-in sensors responsible for all movement measurement in this dataset:

| Sensor | What It Measures |
|---|---|
| **Accelerometer** | Linear acceleration — how fast the phone's velocity is changing in any direction. Picks up movement like steps, sitting down, or the jolt of climbing stairs. |
| **Gyroscope** | Angular velocity — how fast the phone is rotating. Captures orientation changes such as tilting forward when leaning, or the spin of a turn. |

Together, these two sensors give a full three-dimensional picture of the phone's movement at any given moment. The accelerometer cannot distinguish a slow tilt from a fast rotation, and the gyroscope cannot detect forward motion — each sensor covers what the other cannot.

### 2.3 Sampling Rate

Both sensors recorded data at **50 Hz**, meaning they captured 50 readings every second. At this rate, a single second of activity produces 50 data points per sensor axis. This frequency is fast enough to capture fine-grained movement details, such as the brief pause between steps or the subtle deceleration before sitting down, without generating excessive data volume.

### 2.4 Train / Test Split

After collection, the data was partitioned into two non-overlapping groups based on **participants, not time**. This is a deliberate design choice: using completely different people in each group tests whether the model has learned general patterns of human movement, rather than memorizing the habits of specific individuals.

| Split | Participants | Share | Purpose |
|---|---|---|---|
| **Training** | 21 people | 70% | Teaching the model what each activity looks like |
| **Test** | 9 people | 30% | Verifying the model works on unfamiliar individuals |

---

## 3. Activity Classes

Every data window in the dataset is assigned one of six activity labels, determined by human annotators reviewing the video recordings frame by frame. The six activities fall into two natural groups.

| Activity Label | Type | Description |
|---|---|---|
| `WALKING` | Dynamic | Moving on a flat surface such as a hallway or pavement |
| `WALKING_UPSTAIRS` | Dynamic | Ascending a flight of stairs |
| `WALKING_DOWNSTAIRS` | Dynamic | Descending a flight of stairs |
| `SITTING` | Static | Seated in a chair or on a surface, relatively still |
| `STANDING` | Static | Upright and stationary on both feet |
| `LAYING` | Static | Lying down, whether on the back, side, or stomach |

**Dynamic activities** involve continuous rhythmic body movement and produce distinctly periodic sensor signals. **Static activities** involve a largely stationary body, where differentiation depends on orientation rather than motion.

> **Note on difficulty:** Distinguishing dynamic from static activities is relatively straightforward. The harder classification challenge is telling `SITTING` apart from `STANDING`, since both involve an upright, still body. The accelerometer's gravity component — which differs between a seated and an upright posture — becomes critical for making that distinction.

---

## 4. From Sensor Readings to Data Windows

### 4.1 Axes of Measurement

Each sensor measures movement along three spatial axes — X, Y, and Z — covering the full three-dimensional space the phone can move through.

| Sensor | Axis | What It Captures |
|---|---|---|
| Accelerometer | X | Left / right linear acceleration |
| Accelerometer | Y | Forward / backward linear acceleration |
| Accelerometer | Z | Up / down linear acceleration |
| Gyroscope | X | Roll — tilting left or right |
| Gyroscope | Y | Pitch — tilting forward or backward |
| Gyroscope | Z | Yaw — rotating left or right |

At any single instant, the accelerometer produces three numbers (one per axis) and the gyroscope produces three numbers, giving **six values per timestamp**.

### 4.2 The Sliding Window

Rather than working with the entire continuous stream of sensor data at once, the dataset is segmented into fixed-length windows. Each window represents exactly **2.56 seconds** of activity.

| Parameter | Value | Explanation |
|---|---|---|
| Sampling rate | 50 Hz | 50 readings captured every second |
| Window duration | 2.56 seconds | Long enough to capture a meaningful movement pattern |
| Readings per window | **128** | 50 Hz × 2.56 s = 128 readings |
| Raw values per window | **768** | 128 readings × 3 axes × 2 sensors |
| Window overlap | 50% | Each new window starts 64 readings into the previous one |

The 128-reading window is long enough to capture meaningful movement patterns — such as a full walking stride or the motion of sitting down — while remaining short enough that it is unlikely to contain more than one activity.

### 4.3 Overlapping Windows

Windows do not sit end-to-end. Each new window begins 64 readings into the previous one, creating a **50% overlap**. This means the same sensor reading may appear in two consecutive windows.

```
Reading:  1 ────────────── 64 ─────────────── 128
          |                 |                   |
Window 1: [════════════════════════════════════]
Window 2:                  [════════════════════════════════════]
Window 3:                                       [════════════════════════════════════]
          |←── 64 readings ─→|←── step ──→|
```

The purpose of this overlap is continuity. If a person transitions from walking to standing near the boundary of a window, a non-overlapping approach might split that transition poorly. Overlapping windows ensure every transition is fully captured in at least one window, keeping the labeled data clean and coherent.

### 4.4 Raw Data per Window

| Component | Count | Description |
|---|---|---|
| Accelerometer readings | 384 | 128 readings × 3 axes (X, Y, Z) |
| Gyroscope readings | 384 | 128 readings × 3 axes (X, Y, Z) |
| **Total raw values** | **768** | Combined per window |

Each window is then processed through a signal pipeline — noise filtering, gravity separation, and statistical feature extraction — to produce the final **561-feature vector** used for model training.
The details of this is cover in [Feature README](FEATURE.md).

---

## 5. Raw Data Samples

All values are normalized to the range **[-1, 1]**. Negative values indicate movement or rotation in the opposite direction along that axis. The sensors record at 50 Hz, so one reading is captured every **20 milliseconds**.

### Table A — Single Data Point at t = 20 ms

This is what the raw data records from the sensor at one instant in time looks like: six numbers from both sensors combined, one per spatial axis. This single row is reading number 1 out of the 128 readings that make up one window.

| Reading | Time (ms) | Activity | Acc X | Acc Y | Acc Z | Gyro X | Gyro Y | Gyro Z |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 20 | `WALKING` | +0.2781 | -0.0318 | -0.9127 | +0.0121 | +0.0453 | -0.0231 |

That single row is the complete sensor record for one 20-millisecond instant. The dataset chains 128 such rows together to form one complete window.

---

### Table B — One Complete Window (128 readings)

Activity: **`WALKING`** &nbsp;|&nbsp; Duration: **2.56 seconds** &nbsp;|&nbsp; Total readings: **128**

Each row is one reading, taken 20 ms after the previous. Together, the 128 rows form a single labeled example. Readings 11 through 127 follow the same structure and are omitted here to save space.

| Reading | Time (ms) | Acc X | Acc Y | Acc Z | Gyro X | Gyro Y | Gyro Z |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 20 | +0.2781 | -0.0318 | -0.9127 | +0.0121 | +0.0453 | -0.0231 |
| 2 | 40 | +0.2934 | -0.0412 | -0.8843 | +0.0187 | +0.0678 | -0.0312 |
| 3 | 60 | +0.2104 | -0.0891 | -0.8612 | -0.0234 | +0.1123 | -0.0456 |
| 4 | 80 | +0.0892 | -0.1672 | -0.8234 | -0.0891 | +0.2341 | -0.0789 |
| 5 | 100 | -0.0412 | -0.1981 | -0.8156 | -0.1456 | +0.2987 | -0.0923 |
| 6 | 120 | -0.1234 | -0.1543 | -0.8523 | -0.1234 | +0.2134 | -0.0678 |
| 7 | 140 | -0.1678 | -0.0891 | -0.8967 | -0.0678 | +0.1456 | -0.0412 |
| 8 | 160 | -0.1423 | -0.0234 | -0.9234 | -0.0123 | +0.0789 | -0.0189 |
| 9 | 180 | -0.0678 | +0.0412 | -0.9456 | +0.0345 | +0.0234 | -0.0067 |
| 10 | 200 | +0.0234 | +0.0891 | -0.9312 | +0.0789 | -0.0123 | +0.0123 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| _11 – 127_ | _220 – 2540_ | — | — | — | — | — | — |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| **128** | **2560** | -0.1673 | +0.2214 | -0.8911 | -0.0921 | +0.2983 | +0.0413 |

> Reading 128 lands at exactly t = 2,560 ms, closing out the 2.56-second window. The next window begins at reading 65 of this sequence, overlapping the second half of this one.

**How to read the values:**
- Accelerometer (Acc): measures linear acceleration. A value near `0` means little to no movement in that direction. The large negative `Acc Z` values throughout reflect the constant downward pull of gravity on the Z axis.
- Gyroscope (Gyro): measures angular velocity (rotation speed). Values near `0` mean the phone is not rotating. The Y-axis gyroscope values alternate between positive and negative, capturing the forward-backward lean of each walking stride.

---

*For the full dataset, visit the [UCI HAR Dataset page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).*
