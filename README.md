# 📌 Cloud Performance Optimization using Deep Reinforcement Learning

## 👤 Author

Vidit Wanjari
B.Tech — Symbiosis Institute of Technology, Nagpur

---

## 🚀 Overview

This project implements **Deep Reinforcement Learning (DRL)** for intelligent cloud resource management.

Two agents are designed for **different objectives**:

* **DQN (Deep Q-Network)** → Optimizes SLA compliance + CPU utilization
* **DDQN + PER (Double DQN)** → Optimizes energy efficiency + QoS stability

⚠️ This is **not a direct algorithm comparison**.
Each model is tailored to a different cloud use-case.

---

## 🧠 Core Idea

Traditional auto-scalers react to thresholds.
This system learns **optimal decisions over time** using reinforcement learning.

The agent interacts with a simulated cloud environment and improves its policy through rewards.

---

## 📊 Dataset / Data Generation

This project does **not use a fixed external dataset**.

Instead, data is generated dynamically using a **stochastic cloud simulator**:

* **Task arrivals** → Poisson distribution (λ = 0.5 – 4.0)
* **Resource demand** → Log-normal distribution (μ = 0.4, σ = 0.3)

Each state is represented as:

```
s_t = [cpu_i, mem_i, tasks_i]  for each node
```

### 🔁 Training Data Format

During training, the system internally generates transitions:

```
(s_t, a_t, r_t, s_t+1)
```

These represent:

* State
* Action taken
* Reward received
* Next state

### 📁 Extracted Dataset 

A dataset can be exported from logs in CSV format:

```
episode,step,cpu_avg,mem_avg,tasks,action,reward,next_cpu
1,1,0.72,0.65,18,SCALE_UP,-0.8,0.75
...
```

Total size:

* **200 episodes × 50 steps = 10,000 samples**

---

## ⚙️ Technologies Used

* **Language:** Java 17
* **ML Approach:** Deep Reinforcement Learning
* **Algorithms:**

  * DQN
  * DDQN
  * Prioritized Experience Replay (PER)
* **Optimizer:** Adam
* **Environment:** Custom cloud simulator

---

## 🏗️ System Architecture

Main components:

* Cloud Environment Simulator
* DRL Agent (DQN / DDQN)
* Experience Replay Buffer
* Policy Evaluator

The system follows the standard RL loop:

1. Observe state
2. Take action
3. Receive reward
4. Store transition
5. Update model
   
<img width="618" height="466" alt="image" src="https://github.com/user-attachments/assets/8001aca2-8bea-4f54-a33d-9e4c180f368e" />
---

## 📈 Results Summary

| Metric       | DQN (P1)      | DDQN (P2)   |
| ------------ | ------------- | ----------- |
| Final Reward | +0.591        | +1.046      |
| Violations   | 4633          | 1707        |
| Convergence  | ~173 episodes | ~5 episodes |
| Stability    | Low           | High        |
| Energy Usage | Lower         | Higher      |

---

## 📊 Key Findings

* DQN reduces SLA violations vs baseline systems
* DDQN reduces QoS violations by ~63%
* DDQN converges faster due to:

  * Soft target updates
  * PER sampling
* DQN is:

  * Simpler
  * Faster
  * More energy-efficient

---

## ⚖️ Trade-Off Analysis

| Factor            | DQN      | DDQN     |
| ----------------- | -------- | -------- |
| Speed             | Faster   | Slower   |
| Stability         | Lower    | Higher   |
| Energy Efficiency | Better   | Worse    |
| SLA Handling      | Strong   | Moderate |
| QoS Stability     | Moderate | Strong   |

---

## 🧪 Statistical Validation

* **ANOVA** → Confirms learning improvement over training phases
* **Mann-Whitney U Test** → Confirms reward distributions differ

👉 Difference is due to:

* Environment
* Reward design
* Not just algorithm

---

## 📦 Project Structure

```
Cloud-DRL/
│
├── src/
│   ├── dqn/
│   ├── ddqn/
│   ├── environment/
│   ├── replay/
│   └── utils/
│
├── results/
├── dataset/
├── docs/
│   └── report.pdf
│
└── README.md
```

---

## ▶️ How to Run

### 1. Clone Repository

```
git clone https://github.com/Vidit72wanjari/CLOUD-PERFORMANCE-OPTIMIZATION
```

### 2. Compile

```
javac Main.java
```

### 3. Run

```
java Main
```

---

## 📌 Limitations

* Simulation-based (no real cloud deployment)
* Small-scale environment (6–8 nodes)
* No GPU acceleration
* Dataset not pre-stored (generated dynamically)

---

## 🔮 Future Work

* Integration with Kubernetes / OpenStack
* Real cloud trace validation
* Multi-agent RL
* Dueling DQN / PPO comparison
* Large-scale cluster simulation

---

## 📬 Contact

Vidit Wanjari
B.Tech Student
Symbiosis Institute of Technology

---
