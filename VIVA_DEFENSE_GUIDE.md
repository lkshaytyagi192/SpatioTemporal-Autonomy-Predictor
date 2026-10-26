# 🎓 Viva & Technical Defense Guide: TrajectoryRisk-AI
### Spatio-Temporal Trajectory Forecasting & Collision Risk Estimation for Autonomous Systems

---

## 🎯 1. The 30-Second Elevator Pitch (Start with This)
> *"Sir/Ma'am, in autonomous robotics, drones, and self-driving vehicles, perception only tells us where obstacles are right now. Safe autonomous navigation requires knowing **where other dynamic agents will move in the next 3 to 5 seconds**. 
>
> In this project, I developed **TrajectoryRisk-AI**: an end-to-end spatio-temporal deep learning pipeline. It takes an agent’s past 8 timesteps (3.2 seconds) of trajectory coordinates, processes them through a **Seq2Seq LSTM with Temporal Bahdanau Attention**, and predicts the next 12 timesteps (4.8 seconds) with non-linear intent forecasting. 
> 
> Furthermore, I built a downstream **Collision Risk Engine** that computes pairwise spatio-temporal proximity and **Time-to-Collision (TTC)** across agents, triggering automated proactive evasive safety alerts. Our model achieves a **22.9% reduction in Average Displacement Error (ADE)** over standard kinematic baselines at **sub-2ms inference latency** on an NVIDIA RTX GPU."*

---

## 💡 Top 10 Viva Questions & Exact Answers

### Q1: Why did you evaluate using ADE and FDE instead of standard MSE or MAE?
**Answer:**
- **Standard MSE/MAE** averages error across individual coordinates $(x, y)$ independently and ignores physical geometry.
- **ADE (Average Displacement Error)** computes the true physical Euclidean distance in metric space (meters) between the predicted and ground truth positions averaged across all forecast timesteps:
  $$\text{ADE} = \frac{1}{N \cdot T_{pred}} \sum_{i=1}^N \sum_{t=1}^{T_{pred}} \|\hat{\mathbf{p}}_{i,t} - \mathbf{p}_{i,t}\|_2$$
- **FDE (Final Displacement Error)** measures the endpoint drift at the final horizon timestep ($t = 12$ / 4.8s):
  $$\text{FDE} = \frac{1}{N} \sum_{i=1}^N \|\hat{\mathbf{p}}_{i,T} - \mathbf{p}_{i,T}\|_2$$
- In robotics, FDE is critical because even if intermediate error is low, missing the destination point means an autonomous drone or robot could plan a path straight into a dynamic obstacle's final position. ADE and FDE are the de-facto gold standard metrics in ETH/UCY, Trajectron++, and Stanford Drone Dataset benchmarks.

---

### Q2: Why Seq2Seq with Temporal Attention? Why not a standard MLP or Vanilla LSTM?
**Answer:**
1. **Multi-Step Horizon Dependency:** An MLP treats input coordinates as fixed tabular values and fails to capture sequential velocity, acceleration, and curvature patterns over variable time horizons.
2. **The Long-Term Vanishing Gradient in Vanilla LSTM:** When an LSTM encodes 8 past steps into a single final hidden vector $h_8$, a "bottleneck" is created. Information about early maneuvers (like a sharp swerve at step 2) degrades.
3. **Temporal Attention Solves the Bottleneck:** Bahdanau additive attention allows the decoder at *every future prediction step* $t$ to dynamically calculate alignment weights $\alpha_{t, i}$ across all past steps $i \in \{1..8\}$:
   $$e_{t, i} = \mathbf{v}^\top \tanh(\mathbf{W}_{enc} h_i + \mathbf{W}_{dec} s_t)$$
   $$\alpha_{t, i} = \frac{\exp(e_{t, i})}{\sum_{k=1}^8 \exp(e_{t, k})}, \quad \mathbf{c}_t = \sum_{i=1}^8 \alpha_{t, i} h_i$$
   This enables the model to selectively focus on crucial kinematic turning points rather than just the most recent frame.

---

### Q3: How did you handle Coordinate Invariance (Normalization)?
**Answer:**
- If we feed raw absolute global coordinates (e.g. $x=45.2\text{m}, y=80.1\text{m}$), the neural network overfits to spatial location on the map rather than learning trajectory geometry.
- **Solution:** We normalize all trajectories **relative to the last observed anchor point** $\mathbf{p}_{obs\_last} = \mathbf{p}_{8}$:
  $$\tilde{\mathbf{p}}_t = \mathbf{p}_t - \mathbf{p}_{8}$$
- This anchors the origin at $(0, 0)$ for each agent's prediction window. The network learns pure relative motion dynamics, making it translation-invariant across any coordinate frame. During inference, we add back $\mathbf{p}_{8}$ to project back into absolute coordinates for path planning.

---

### Q4: Why did you choose Smooth L1 (Huber) Loss instead of standard MSE?
**Answer:**
- Standard MSE squares errors ($e^2$). In real-world sensor logs and moving agent tracking, occasional outliers occur (e.g., sensor noise, sudden swerving). Squared loss heavily penalizes these outliers with huge gradients, destabilizing recurrent weight updates.
- **Smooth L1 Loss (Huber Loss)** acts like quadratic L2 loss when error is small ($|e| < 1.0$), ensuring smooth gradient descent near convergence, but transitions to linear L1 loss ($|e| - 0.5$) for large deviations:
  $$L_{\delta}(e) = \begin{cases} 0.5 e^2 & \text{if } |e| < 1 \\ |e| - 0.5 & \text{otherwise} \end{cases}$$
- This gives optimal numerical stability and prevents gradient explosion in sequence modeling.

---

### Q5: How does your Collision Risk Engine compute Time-to-Collision (TTC)?
**Answer:**
- At inference time, our model predicts $T_{pred} = 12$ future coordinate pairs for all active agents in the environment.
- For every unique agent pair $(i, j)$, the engine computes the pairwise Euclidean distance vector over time:
  $$D_{i,j}(t) = \|\hat{\mathbf{p}}_i(t) - \hat{\mathbf{p}}_j(t)\|_2, \quad \forall t \in \{1..12\}$$
- If the minimum distance $\min_t D_{i,j}(t)$ drops below our critical collision threshold $R_{crit} = 0.8\text{m}$, the earliest timestep index $t^*$ where proximity violates the buffer defines the **Time-to-Collision**:
  $$\text{TTC} = t^* \times \Delta t \quad (\text{where } \Delta t = 0.4\text{s})$$
- It also assigns a normalized **Collision Risk Index (CRI)** between $0.0$ and $1.0$, allowing a downstream navigation controller to prioritize evasive maneuvers.

---

### Q6: What was your Baseline, and did Deep Learning genuinely beat it?
**Answer:**
- **Baseline:** A Constant Velocity (CV) Kinematic Model, which linearly extrapolates the last observed velocity vector:
  $$\mathbf{v} = \frac{\mathbf{p}_8 - \mathbf{p}_7}{\Delta t}, \quad \hat{\mathbf{p}}_{8+k} = \mathbf{p}_8 + k \cdot \mathbf{v} \cdot \Delta t$$
- **Benchmark Results:**
  - Constant Velocity: **ADE = 0.8813 m**, **FDE = 2.1548 m**
  - Our Seq2Seq + Attention: **ADE = 0.6793 m**, **FDE = 1.7869 m**
- **Improvement:** Our model reduced ADE by **22.91%** and FDE by **17.07%**. This proves that the network successfully captures non-linear turning, deceleration, and curved trajectories that linear extrapolation completely misses.

---

### Q7: What is the computational latency? Can this run on an embedded edge computer (like NVIDIA Jetson or Raspberry Pi)?
**Answer:**
- On our test benchmark, inference time per trajectory forecast is **~1.98 ms** (equivalent to **>500 Hz** throughput).
- Autonomous mobile robots and drones typically run perception and planning loops at **10 Hz to 30 Hz** (every 33 to 100 ms).
- At under 2 ms, our model consumes negligible CPU/GPU cycles, leaving 95%+ of compute budget for SLAM, point-cloud processing, and motor control loops on edge platforms like an NVIDIA Jetson Orin Nano or Raspberry Pi 5 with ONNX Runtime.

---

### Q8: How would you integrate this software into a real autonomous robot (ROS 2 / Drone)?
**Answer:**
1. **Perception Subscription:** A ROS 2 node subscribes to tracked dynamic obstacle topics (`/perception/tracked_obstacles` from LiDAR/Camera tracker).
2. **Sliding Window Buffer:** A circular buffer maintains the last 8 coordinate poses for each detected `obstacle_id`.
3. **Model Inference:** When 8 timesteps are logged, the PyTorch/ONNX model runs inference to predict the 12-step path.
4. **Collision Risk Topic:** The node publishes:
   - `/prediction/trajectories` (for RViz 3D trajectory visualization)
   - `/safety/collision_alert` (alerting the path planner, e.g. Nav2 or ArduPilot Guided Mode, to adjust waypoint velocities or inject repulsive potential fields).

---

### Q9: What are the current limitations of this architecture?
**Answer:**
- While temporal attention models individual agent dynamics effectively, it does not explicitly pool social interactions across neighboring agents in a dense crowd simultaneously.
- In dense human crowds, an agent's path is constrained not only by its own momentum but by social conventions (yielding, walking on the right).

---

### Q10: What is your planned future work / Next Steps?
**Answer:**
1. **Spatio-Temporal Graph Neural Networks (ST-GNNs):** Adding spatial graph attention layers (GAT) to model inter-agent social compliance alongside temporal LSTM dynamics.
2. **Multimodal Trajectory Distribution:** Outputting Gaussian Mixture Model (GMM) parameters or conditional diffusion steps to predict multiple plausible future modes (e.g., 60% chance agent turns left, 40% goes straight).
3. **Hardware Deployment:** Compiling the PyTorch model via TensorRT INT8 quantization for real-time deployment on a Jetson-based autonomous mobile robot.
