# 🤖 Autonomous Navigation of e-puck Robot using Q-Learning in Webots

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Webots](https://img.shields.io/badge/Webots-Simulator-red?style=for-the-badge&logo=robot&logoColor=white)](https://cyberbotics.com/)
[![Reinforcement Learning](https://img.shields.io/badge/AI-Q--Learning-brightgreen?style=for-the-badge)](https://en.wikipedia.org/wiki/Q-learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An intelligent mobile robotics project implementing **Tabular Q-Learning (Reinforcement Learning)** to enable an **e-puck** differential-drive robot to navigate autonomously and avoid obstacles in a simulated **Cyberbotics Webots** environment.

---

## 📌 Project Overview

This project bridges **mobile robotics**, **sensory perception**, and **reinforcement learning (RL)**. Without using hardcoded navigation rules, the robot learns an optimal obstacle-avoidance policy through repeated trial-and-error interactions with its environment.

### 🎯 Key Objectives
- **Autonomous Navigation:** Enable the e-puck robot to explore an enclosed arena without collisions.
- **Sensor Fusion & Discretization:** Convert continuous infrared proximity data into a discrete state space suitable for tabular Q-learning.
- **Reinforcement Learning from Scratch:** Implement the Bellman update rule with an $\epsilon$-greedy exploration-exploitation tradeoff.
- **Simulation in Webots:** Bridge Python RL algorithms with Webots robot controllers and physics.

