# Language Conditioned Goal Predicition
## Introduction
This project uses the MetaDrive simulator to generate driving data and train a neural network that predicts the vehicle's future goal position. Each sample contains the ego vehicle's state, LiDAR observations, short-term driving history, and a driving instruction. The model predicts the vehicle's future position in the ego coordinate frame over a five-second horizon.

## Language-Conditioned Prediction

The model is conditioned on a discrete driving instruction that describes the intended maneuver (Fixed before training). the Goals predicted are conditioned based on the instruction given to the model

<img width="1152" height="422" alt="image" src="https://github.com/user-attachments/assets/5720ce8c-b8bd-42ea-afb5-e825c98351f6" />

## Model Architecture

The model is implemented as the `GoalPointPredictor` in `train.py`. It combines current sensor data, vehicle state, temporal history, and the driving instruction.

### Inputs

The model receives:

- **LiDAR:** 240 distance measurements from the current observation.
- **Ego state:** 12 features, including speed, steering, yaw rate, heading, position, velocity, previous action, lateral offset, and heading difference.
- **Driving history:** Three time steps of previous ego positions and speeds.
- **Instruction:** One of the seven discrete driving commands.

### Feature Processing

Each input type is projected into a shared 128-dimensional representation:

1. LiDAR measurements are converted into LiDAR tokens.
2. The current ego state is converted into an ego-state token.
3. Historical positions and speeds are combined and converted into history tokens.
4. Multi-head attention allows the ego-state and history tokens to attend to the LiDAR tokens.
5. The attended features are normalized and averaged into a single scene representation.
6. The instruction ID is mapped to a 32-dimensional instruction embedding.
7. The scene representation and instruction embedding are concatenated.

The combined representation is passed through a multilayer prediction head consisting of linear layers and GELU activations. The final layer outputs the predicted two-dimensional future goal position.

```text
LiDAR measurements ───────┐
                           │
Ego state ────────────────>│
                           │  Multi-head attention
Position/speed history ───>│
                           │
                           v
                    Scene representation
                           │
Driving instruction ──> Embedding
                           │
                           v
                 Concatenated representation
                           │
                           v
                    Prediction head
                           │
                           v
                 Future ego-frame goal (x, y)
```

The trained checkpoint is stored at `model/goal_point_predictor.pt`. Manual instruction-conditioned inference can be run using `manual_inference.py`.
