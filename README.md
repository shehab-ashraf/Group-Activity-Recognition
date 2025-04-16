# Group Activity Recognition

This repository implements the paper [Hierarchical Deep Temporal Models for Group Activity Recognition](https://arxiv.org/abs/1607.02643), which introduces a hierarchical approach that integrates spatial and temporal modeling to recognize group activities in videos.

---

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [1. Person-Level Modeling (Stage One)](#1-person-level-modeling-stage-one)
  - [2. Group-Level Modeling (Stage-Two)](#2-group-level-modeling-stage-two)
- [Datasets](#datasets)
- [Results](#results)
- [Summary](#summary)

---

## Overview

This project addresses the challenging task of recognizing activities performed by groups of people in video sequences. The key contributions include:
- **Detection and Tracking:** Identifying individuals within video frames.
- **Person-Level Action Recognition:** Capturing the temporal dynamics of individual actions.
- **Group-Level Activity Recognition:** Aggregating individual cues to understand complex group interactions.
- **Hierarchical Design:** Combining deep CNN features with LSTM-based temporal modeling to create robust representations.

---

## Model Architecture

The model is organized into two primary stages:

### 1. Person-Level Modeling (Stage One)

#### Input Preparation
- **Detection & Tracking:**  
  For every detected and tracked individual, a bounding box is cropped from each frame.
  
- **Feature Extraction:**  
  A pre-trained CNN (e.g., AlexNet) extracts high-level features (e.g., from the fc7 layer) from each cropped image.

#### Temporal Dynamics with LSTM
- **Feeding Features into LSTM:**  
  The extracted CNN features are fed into an LSTM network.
  
- **Capturing Dynamics:**  
  The LSTM models the temporal evolution of each individual's actions, where the hidden state at each time step represents a short-term memory of their behavior.

#### Output
- **Combined Feature Vector:**  
  For every person at each time step, a unified feature vector is produced by concatenating the CNN features and the corresponding LSTM hidden state.
  
- **Classification:**  
  This feature vector can be passed through a softmax classifier for direct person-level action recognition.

---

### 2. Group-Level Modeling (Stage-Two)

#### Aggregation (Pooling)
- **Aggregation Across Individuals:**  
  At every time step, the model aggregates person-level features across the scene.
  
- **Pooling Strategies:**  
  - **Standard Pooling:**  
    Each person’s feature representation is concatenated followed by a max pooling operation to yield a single frame-level feature vector.
  - **Sub-group Pooling (for Team Sports):**  
    - **Divide into Spatial Sub-Groups:**  
      Players are divided into sub-groups based on spatial regions (e.g., left and right teams).
    - **Group-wise Pooling:**  
      Pooling is applied separately to each sub-group, and the resulting features are concatenated. This approach reduces ambiguities when similar global features occur across different teams.

#### Temporal Dynamics of the Group
- **Feeding Pooled Features:**  
  The aggregated frame-level feature vector is then fed into a second LSTM.
  
- **Modeling Group Evolution:**  
  This LSTM captures the temporal evolution of the group activity. Its final hidden state serves as a dynamic summary of the overall group behavior throughout the video sequence.

#### Final Classification
- **Prediction:**  
  A fully connected layer, followed by a softmax classifier, processes the group-level LSTM output to predict the overall group activity.

---

## Datasets

### Volleyball Dataset
- **Size:**  
  4830 frames extracted from 55 videos.
  
- **Annotations:**  
  - **Player Actions:** 9 distinct actions (e.g., set, spike, block).
  - **Team Activities:** 8 different team-level activities.
  - **Additional Information:** Player locations and trajectories.
  
- **Dataset Organization:**
  - **Videos:**  
    Each video is assigned a unique ID (0–54).
  - **Train Videos:**  
    IDs: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
  - **Validation Videos:**  
    IDs: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
  - **Test Videos:**  
    IDs: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

---

## Results

### Volleyball Dataset Performance

| **Baseline** | **Accuracy** |
|--------------|--------------|
| Baseline 1   | 74.2%        |
| Baseline 3   | 77.1%        |
| Baseline 4   | In Progress  |
| Baseline 5   | In Progress  |
| Baseline 6   | In Progress  |
| Baseline 7   | In Progress  |
| Baseline 8   | In Progress  |
| Baseline 9   | In Progress  |

---

## Summary

- **Hierarchical Design:**  
  The two-stage model first extracts and models individual actions, then aggregates these cues to capture dynamic group behavior.
  
- **Spatial and Temporal Integration:**  
  Deep CNN features combined with LSTM-based temporal modeling yield a comprehensive representation for group activity recognition.
  
- **Flexible Pooling Mechanisms:**  
  Both standard pooling and specialized sub-group pooling (especially useful in team sports) ensure effective aggregation of person-level features.
