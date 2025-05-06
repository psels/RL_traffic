# RL_traffic

## Description
**RL_traffic** is a reinforcement learning project focused on intelligent traffic light management to optimize urban traffic flow. This project was developed during an intensive bootcamp at **Le Wagon** in Paris by a team of three developers, each contributing equally.

The goal of the project is to leverage reinforcement learning algorithms, specifically **3DQN** (Double Deep Q-Network), to train a model that dynamically controls traffic lights based on simulated traffic conditions using **SUMO** (Simulation of Urban MObility).

## Project Overview
### 20th-Century Traffic Lights
Since their invention, traffic lights have evolved little. They operate in fixed phases with variable durations based on road size (e.g., 30 seconds green, 30 seconds red). This system works well for symmetric traffic, ensuring smooth flow and quick intersection clearance.

### The Problem of Asymmetric Traffic
Issues arise during peak hours, such as 6 a.m., when traffic becomes asymmetric. Main roads to business districts become congested, while secondary roads remain empty. The rigidity of fixed cycles leads to traffic jams that persist until traffic normalizes, causing delays and frustration.

### Our Adaptive Solution
Our project introduces a dynamic traffic management model that adjusts traffic light behavior in real-time based on traffic conditions. Using reinforcement learning, our **3DQN** model optimizes traffic flow during peak hours or adapts to events like road closures or large gatherings (e.g., sports events). In a simulated scenario with heavy traffic asymmetry, our intelligent agent doubled average vehicle speeds and halved waiting times compared to traditional fixed-cycle systems.

To handle increased complexity, we extended the model to manage multiple intersections, with each agent optimizing its own traffic light while coordinating with others, demonstrating scalability.

### Reward System Design
The model’s decision-making relies on a carefully designed reward function. Initially, we compared traffic states before and 20 seconds after an action, rewarding actions that cleared the most vehicles from an intersection. However, this approach neglected vehicles on less busy roads, causing unfair delays for some users.

To address this, we incorporated cumulative waiting time into the reward function, penalizing the model when any vehicle was stuck for too long. This refinement improved fairness but introduced a challenge: **reward hacking**, where the agent maximized rewards by delaying vehicles to accumulate higher penalties. Through iterative refinement, we eliminated this bias, achieving a balanced and effective reward system.

### Addressing Reward Hacking
Reward hacking is a common challenge in reinforcement learning, where an agent exploits the reward function in unintended ways. In our case, the agent prioritized busy roads excessively, leaving quieter roads blocked to inflate rewards. By fine-tuning the reward function to balance vehicle clearance and waiting times, we mitigated this issue, ensuring equitable traffic management.

### From Simulation to Real-World Deployment
Using **SUMO**, our model can simulate complex intersections, neighborhoods, or even entire cities. In a real-world setting, traffic data could be collected via cameras and computer vision neural networks, enabling cost-effective implementation. Potential strategies include:
- Prioritizing overall traffic flow.
- Favoring high-occupancy vehicles.
- Supporting eco-friendly transport options.

These applications open avenues for further research and policy discussions.

## Technologies Used
- **Reinforcement Learning**: 3DQN (Double Deep Q-Network), 2DQN
- **Simulation**: SUMO (Simulation of Urban MObility)
- **Language**: Python
- **Tools**: Jupyter Notebook, Git, GitHub

## Context
This project was completed during a coding bootcamp at **Le Wagon** in Paris, where our team of three developers collaborated to design and deliver a comprehensive reinforcement learning solution. Each member contributed equally, with clear task delegation to maximize efficiency.

## Demo
A video demonstration showcasing the traffic simulation and the decisions made by the reinforcement learning model is available here: [Demo](https://www.youtube.com/watch?v=bnvSJbV-G6g).
