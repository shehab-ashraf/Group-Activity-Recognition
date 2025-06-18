<h1 align="center">Group Activity Recognition</h1>
<p align="center">
  <strong>State-of-the-art implementation</strong> of the CVPR 2016 paper: 
  <a href="https://arxiv.org/pdf/1607.02643"><em>A Hierarchical Deep Temporal Model for Group Activity Recognition</em></a><br>
</p>

##  Table of Contents
- [ Key Improvements](#-key-improvements)
- [ Volleyball Dataset](#-volleyball-dataset)
- [ Ablation Study](#-ablation-study)


<a name="-key-improvements"></a>
##  Key Improvements

| Improvement | Impact |
|-------------|--------|
| **Modern Backbone** | Replaced AlexNet with ResNet50 for superior feature extraction |
| **Framework Upgrade** | Full PyTorch implementation (original was Caffe) |
| **Performance Boost** | Achieved 92.3% accuracy  |


<a name="-volleyball-dataset"></a>
##  Volleyball Dataset

### Overview
The dataset consists of **4,830 annotated frames** extracted from **55 YouTube volleyball videos**. It includes:
- **8 team activity labels** (e.g., "Left Spike", "Right Winpoint")
- **9 player action labels** (e.g., "Blocking", "Setting")
- Player bounding boxes with action annotations

![Annotation Example](https://github.com/user-attachments/assets/50f906ad-c68c-4882-b9cf-9200f5a380c7)  
*Sample frame showing team activity and player bounding boxes*

###  Dataset Structure Summary

#### Video Organization
- **Total Videos**: 55 (IDs 0-54)
- **Splits**:
  - **Train**: Videos 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38-42, 48, 50, 52-54
  - **Validation**: Videos 0, 2, 8, 12, 17, 19, 24, 26-28, 30, 33, 46, 49, 51
  - **Test**: Videos 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43-45, 47

#### Directory Organization
```bash
volleyball/
└── video_{ID}/                  # Each of the 55 videos (0-54)
    ├── frame_{timestamp_A}/      # First key moment (e.g. 29885)
    │   ├── 00001.jpg            # -20 frames
    │   ├── ...                  # ...
    │   ├── 00021.jpg            # Target frame (timestamp_A)
    │   ├── ...                  # ...
    │   └── 00041.jpg            # +20 frames
    ├── frame_{timestamp_B}/      # Second key moment (e.g. 29886)
    │   ├── 00001.jpg            # -20 frames 
    │   └── ...                  # Same structure
    ├── ...                      # More frame directories
    └── annotations.txt          # Lists ALL key moments
```
[Original Dataset Repository](https://github.com/mostafa-saad/deep-activity-rec), For further information.

<a name="-ablation-study"></a>
##  Ablation Study

### Baseline Models
This section outlines the baselines , based on the CVPR 2016 paper: *A Hierarchical Deep Temporal Model for Group Activity Recognition* by Ibrahim et al.

#### **B1: Image Classification**
- **Architecture**: Single-frame ResNet-50
- **Description**: A basic image-level classifier that processes the entire scene using a CNN (ResNet-50).
  
#### **B2: Person Classification**
- **Architecture**: ResNet-50 per player → feature pooling → FC
- **Description**: CNN applied to each detected person individually. The extracted features are pooled across people and passed to a softmax classifier.
  
#### **B3: Fine-tuned Person Classification**
- **Architecture**: ResNet-50 (fine-tuned for person action) per player → feature pooling → FC
- **Description**: Similar to B2, but the CNN is fine-tuned for person-level action classification.

#### **B4: Temporal Model with Image Features**
- **Architecture**: ResNet-50 on full image → LSTM → FC
- **Description**: Temporal extension of B1. Whole image features are extracted and passed through an LSTM for sequence modeling.

#### **B5: Temporal Model with Person Features**
- **Architecture**: ResNet-50 per player → feature pooling per frame → LSTM → FC
- **Description**: Temporal extension of B2. Pooled person features over time are input to an LSTM to model group activity sequences.

#### **B6: No Player-LSTM (Only Group-LSTM)**
- **Architecture**: ResNet-50 per player (fine-tuned) → pooled → Group-LSTM → FC
- **Description**: Similar to the full model but **removes the first LSTM** responsible for modeling individual person dynamics. Only a group-level LSTM is used.

#### **B7: No Group-LSTM (Only Player-LSTM)**
- **Architecture**: ResNet-50 per player (fine-tuned) → Player-LSTM → pooled → FC
- **Description**: Omits the group-level LSTM. Only temporal modeling is applied at the player level, followed by feature pooling and final classification.

#### **B8: Full Hierarchical Model**
- **Architecture**: ResNet-50 (fine-tuned per player) → Player-LSTM → pooling → Group-LSTM → FC
- **Description**: The complete two-stage model proposed in the paper. Captures both **individual temporal actions** and **group-level temporal dynamics**.

### Key Findings

**1. Temporal Modeling is Essential**
Comparing B3 vs B6 shows:
- Adding LSTM improves accuracy
- Temporal dynamics critical for activity understanding
<table>
  <tr>
    <th>B3: Without Temporal Modeling</th>
    <th>B6: With Player-LSTM</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8a04a70e-55d1-4276-a1b6-fc8d1b974e56" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/f2c48186-42f8-4e3f-8746-0dc0ad1d4521" width="300"></td>
  </tr>
</table>

**2. Team-Aware Pooling** <br />
- Independent team feature processing (Hierarchical Two-stage Temporal Model) imporoves accuracy and reduces confusion between Left/right winpoints
<table>
  <tr>
    <th>B7: Without Team-Aware Pooling</th>
    <th>B8: With Team-Aware Pooling</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/45b89d64-abd4-4bd5-81df-9f6767121ca0" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/5413e79c-757f-4383-9583-496e89e0c85c" width="300"></td>
  </tr>
</table>


### Performance Comparison

| Model                                           |Accuracy | notebook |
|-------------------------------------------------|---------|----------|
| Baseline_1: Image Classification                |  74.8%  |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/baseline-1-image-classification)| 
| Baseline_3: Fine-Tuned Person Classification    |  79.1%  |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/baseline-3-fine-tuned-person-classification)| 
| Baseline_4: Temporal Model with Image Features  |  77.7%  |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/baseline-4-temporal-model-with-image-features)|
| Baseline_5: Temporal Model with Person Features |  49.9   |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/baseline-5-temporal-model-with-person-features)|
| Baseline_6: Two-stage Model without LSTM 1      |  82.0%  |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/baseline-6-two-stage-model-without-lstm-1)|
| Baseline_7: Two-stage Model without LSTM 2      |  84.2%  |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/baseline-7-two-stage-model-without-lstm-2)|
| Hierarchical Model: Two-stage Temporal Model    |  92.3%  |[<img src="https://kaggle.com/static/images/site-logo.svg" width="50">](https://www.kaggle.com/code/ashrafs1/hierarchical-model-two-stage-temporal-model)|
