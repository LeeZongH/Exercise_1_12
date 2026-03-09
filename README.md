# Exercise 1.12: PPO Implementation for MountainCar-v0

## Description
This repository contains the Python code for Exercise 1.12. It implements the **Proximal Policy Optimization (PPO)** algorithm (Actor-Critic architecture) to solve the `MountainCar-v0` environment.

> **⚠️ Important Note for TAs:** 
> While completing the `# YOUR CODE HERE` block for the PPO core logic, I also made several necessary upgrades and optimizations to the original template to ensure the code runs on modern libraries and actually converges within the given 1000 epochs.

## Major Modifications from the Original Template

1. **Migrated to `gymnasium` (API Update):**
   * Replaced the deprecated `gym` library with the actively maintained `gymnasium`.
   * Updated `env.reset()` to return `(observation, info)`.
   * Updated `env.step()` to handle the 5-value return tuple: `(observation, reward, terminated, truncated, info)`.
   * Replaced the deprecated `env.seed()` with `env.action_space.seed()` for reproducibility.

2. **Reward Shaping (Crucial for Convergence):**
   * Vanilla `MountainCar-v0` provides a sparse reward of `-1` per step, making it notoriously difficult for PPO to learn without vast amounts of exploration. 
   * I implemented **Reward Shaping** (`shaped_reward = reward + 100 * abs(velocity)`) to explicitly encourage the agent to build momentum. I also added a bonus (`+10`) for reaching the right side of the hill (`position > 0.1`).

3. **Entropy Bonus for Exploration:**
   * Added an entropy term to the actor's loss (`action_loss = action_loss - 0.01 * entropy`) to encourage exploration and prevent premature convergence to sub-optimal deterministic policies.

4. **Final Evaluation Display:**
   * Added a testing block at the end of `main()`. If the flag `display_final_result = True` is set, a Pygame window will render the trained agent's performance after the 1000 epochs are completed.

## Environment & Key Dependencies
This project was developed and tested using an Anaconda virtual environment. The core dependencies are:

* **python**==3.10.19
* **torch**==2.5.1+cu121
* **numpy**==2.2.6
* **gymnasium**==1.2.3
* **pygame**==2.6.1
* **matplotlib**==3.10.8
* **tensorboardX**==2.6.4
* **tensorboard**==2.20.0

Other required packages are listed in the `requirements.txt` file.

## Setup Instructions
To avoid any dependency conflicts and ensure reproducibility, please follow these steps to set up the environment using Conda:

1. **Create the virtual environment with the specific Python version:**
    ```bash
    conda create -n torch_rlgym_env python=3.10.19 -y

2. **Activate the environment:**
    ```bash
    conda activate torch_rlgym_env

3. **Install CUDA Toolkit via Conda:**
    ```bash
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

4. **Install all required packages via requirements.txt:**
    ```bash
    pip install -r requirements.txt

## How to Run
1. **Once the environment is fully set up and activated, you can run the agent by executing:**
    ```bash
    python PPO_MountainCar-314513025.py

(Note: If render_mode="human" is enabled in the script, a Pygame window will pop up showing the trained car reaching the flag.)