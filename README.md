
# 📌 Cloud Performance Optimization using Deep Reinforcement Learning

**Author:** Vidit Wanjari
**Course:** Cloud Computing Tools and Techniques
**Institute:** Symbiosis Institute of Technology, Nagpur

---

## 🚀 Overview

This project implements **Deep Reinforcement Learning (DRL)** for cloud resource management using:

* **DQN (Deep Q-Network)** → Performance Optimization
* **DDQN (Double DQN + PER)** → Energy-Aware VM Consolidation

Unlike typical Python-based implementations, this system is built **entirely in Java**, making it directly compatible with JVM-based cloud platforms like OpenStack and Kubernetes.

The project evaluates how **different RL designs behave under different cloud objectives**, instead of incorrectly comparing them as identical problems.

---

## ⚠️ Core Insight (Most People Miss This)

This is **not a fair DQN vs DDQN comparison**.

* DQN → Optimizes SLA + CPU
* DDQN → Optimizes Energy + QoS

Different:

* Environments
* Reward functions
* Action spaces

So the real contribution is:

> **Use-case-specific DRL design for cloud systems**

---

## 🧠 Key Features

* ✅ Pure **Java implementation** (no TensorFlow / PyTorch)
* ✅ Custom **cloud simulator** (Poisson + Log-normal workloads)
* ✅ Two DRL agents:

  * DQN (Performance)
  * DDQN + PER (Energy Optimization)
* ✅ Statistical validation:

  * ANOVA
  * Mann-Whitney U Test
* ✅ Interpretable **Q-value policy table**
* ✅ Complexity analysis (DQN vs DDQN runtime)

---
<img width="618" height="466" alt="image" src="https://github.com/user-attachments/assets/8001aca2-8bea-4f54-a33d-9e4c180f368e" />


## ⚙️ Technologies Used

* **Language:** Java 17
* **ML Approach:** Deep Reinforcement Learning
* **Algorithms:**

  * DQN
  * DDQN
  * Prioritized Experience Replay (PER)
* **Optimization:** Adam Optimizer
* **Simulation:** Custom-built cloud environment

---

## 📊 Results Summary

| Metric       | DQN (P1)      | DDQN (P2)   |
| ------------ | ------------- | ----------- |
| Final Reward | +0.591        | +1.046      |
| Violations   | 4633          | 1707        |
| Convergence  | ~173 episodes | ~5 episodes |
| Energy Usage | Lower         | Higher      |
| Stability    | Low           | High        |

---

## 📈 Key Findings

* DQN improves SLA handling by **23.5% vs threshold autoscaler** 
* DDQN reduces QoS violations by **~63%** 
* DDQN is more stable due to:

  * Soft updates
  * PER
* DQN is:

  * Faster
  * More energy-efficient

---

## ⚖️ Trade-Off Analysis

| Factor            | DQN      | DDQN     |
| ----------------- | -------- | -------- |
| Speed             | Faster   | Slower   |
| Stability         | Lower    | Higher   |
| Energy Efficiency | Better   | Worse    |
| SLA Handling      | Good     | Moderate |
| QoS Stability     | Moderate | Strong   |

---

## 🧪 Statistical Validation

* **ANOVA:** Significant learning improvement
* **Mann-Whitney U Test:** Reward distributions differ significantly

This confirms:

> Differences are due to **environment + reward design**, not just algorithm.

---

## 📦 Project Structure (Suggested)

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
├── screenshots/
├── docs/
│   └── report.pdf
│
└── README.md
```

---

## ▶️ How to Run

1. Clone repository

```
git clone https://github.com/Vidit72wanjari/CLOUD-PERFORMANCE-OPTIMIZATION-AND-ENERGY-AWARE-VM-CONSOLIDATION-VIA-DEEP-REINFORCEMENT-LEARNING
```

2. Compile

```
javac Main.java
```

3. Run

```
java Main
```

---

## 📌 Limitations

* Simulator-based (no real cloud deployment)
* Small-scale environment (6–8 nodes)
* Manual neural network implementation
* Reward tuning not fully optimized

---

## 🔮 Future Work

* Kubernetes integration
* Real cloud trace validation
* Multi-agent RL
* Dueling DQN / PPO comparison
* Larger-scale cluster testing

---

## 📚 References

[1] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529–533, 2015.  

[2] G. Tesauro et al., "A hybrid reinforcement learning approach to autonomic resource allocation," in Proc. 4th Int. Conf. Autonomic Computing (ICAC), 2007, pp. 65–73.  

[3] X. Dutreilh et al., "Using reinforcement learning for autonomic resource allocation in clouds: towards a fully automated workflow," in Proc. ICAS 2011, pp. 68–77.  

[4] E. Barrett, E. Howley, and J. Duggan, "Applying reinforcement learning towards automating resource allocation and application scalability in the cloud," Concurrency Comput.: Pract. Exp., vol. 25, no. 12, pp. 1656–1675, 2013.  

[5] N. Liu et al., "A hierarchical framework of cloud resource allocation and power management using deep reinforcement learning," in Proc. IEEE ICDCS 2017, pp. 372–382.  

[6] M. Cheng, J. Li, and S. Nazarian, "DRL-cloud: Deep reinforcement learning-based resource provisioning and task scheduling for cloud service providers," in Proc. ASP-DAC 2018, pp. 129–134.  

[7] H. Arabnejad, C. Pahl, P. Jamshidi, and G. Estrada, "A comparison of reinforcement learning techniques for fuzzy cloud auto-scaling," in Proc. IEEE/ACM CCGrid 2017, pp. 64–73.  

[8] H. Van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proc. AAAI 2016, pp. 2094–2100.  

[9] Z. Wang et al., "Dueling network architectures for deep reinforcement learning," in Proc. ICML 2016, pp. 1995–2003.  

[10] Y. Zhang, F. Gu, J. Liu, and M. Huang, "Asynchronous advantage actor-critic-based resource management for edge computing," IEEE Internet Things J., vol. 7, no. 4, pp. 3469–3480, 2020.  

[11] J. Kumar and A. K. Singh, "QoS-aware cloud service composition using a fuzzy-based multi-objective optimization approach," J. Netw. Comput. Appl., vol. 72, pp. 42–53, 2022.  

[12] Z. Peng et al., "Random task scheduling scheme based on reinforcement learning in cloud computing," Concurrency Comput.: Pract. Exp., vol. 27, no. 5, pp. 1097–1110, 2023.  

[13] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.  

[14] C. J. C. H. Watkins and P. Dayan, "Q-learning," Mach. Learn., vol. 8, no. 3–4, pp. 279–292, 1992.  

[15] R. Buyya et al., "Cloud computing and emerging IT platforms: Vision, hype, and reality," Future Gener. Comput. Syst., vol. 25, no. 6, pp. 599–616, 2009.  

[16] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, "Prioritized experience replay," in Proc. ICLR 2016.  

[17] T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," in Proc. ICLR 2016.  

[18] T. X. Tran and D. Pompili, "Joint task offloading and resource allocation for multi-server mobile-edge computing networks," IEEE Trans. Veh. Technol., vol. 68, no. 1, pp. 856–868, 2019.  

[19] M. G. Bellemare, W. Dabney, and R. Munos, "A distributional perspective on reinforcement learning," in Proc. ICML 2017, pp. 449–458.  

[20] X. Chen et al., "Optimized computation offloading performance in virtual edge computing systems via deep reinforcement learning," IEEE Internet Things J., vol. 6, no. 3, pp. 4005–4018, 2018.  

[21] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR 2015.  
 

---

## 📬 Contact

**Vidit Wanjari**
B.Tech Student
Symbiosis Institute of Technology

---

