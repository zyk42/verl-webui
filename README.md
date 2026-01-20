# VERL WebUI

VERL WebUI is a user-friendly graphical interface designed for **[VERL](https://github.com/volcengine/verl)** (Volcano Engine Reinforcement Learning). It simplifies the configuration and command generation for RLHF training of Large Language Models.

![Project Logo/Banner Placeholder](path/to/logo.png)

## Introduction

This WebUI provides an intuitive way to configure various components of VERL, including PPO/GRPO algorithms, model parameters (Actor, Critic, Reward, Reference), and data settings. It streamlines the process of generating complex training commands for large-scale RLHF experiments without needing to manually write lengthy shell scripts.

## 🔗 References

*   **VERL GitHub Repository:** [https://github.com/volcengine/verl](https://github.com/volcengine/verl)
*   **VERL Documentation:** [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html)

## ✨ WebUI Usage

We provide a user-friendly Web Interface (WebUI) to generate training configurations and commands easily.

![WebUI Screenshot Placeholder](path/to/webui_screenshot.png)

### 🚀 Quick Start

To launch the WebUI, ensure you have the required dependencies installed (including `gradio`).

```bash
pip install gradio
```

#### Method 1: Python Command (Recommended)

You can start the WebUI directly using Python. By default, it runs on port `7860`.

```bash
python webui.py
```

**Specify a custom port:**

```bash
python webui.py --port 8888
```

**Enable public sharing:**

```bash
python webui.py --share
```

#### Method 2: PowerShell Script (Windows)

For Windows users, we provide a convenient PowerShell script `run_webui.ps1`.

```powershell
# Basic usage
.\run_webui.ps1

# Specify port
.\run_webui.ps1 -Port 7862

# Enable public link
.\run_webui.ps1 -Port 7862 -Share
```

## 🛠 Features & Configuration

The WebUI allows you to configure the following modules:

*   **Data Configuration**: Setup training and validation datasets, batch sizes, and prompt lengths.
*   **Model Configuration**:
    *   **Actor**: Configure the policy model, PPO hyperparameters, and strategies (FSDP/Megatron).
    *   **Reference Model**: Enable/Disable reference models with flexible KL implementation choices (`use_kl_loss` or `use_kl_in_reward`).
    *   **Critic Model**: Configure value function models.
    *   **Reward Model**: Setup reward model parameters and managers (Naive, Prime, DAPO).
*   **Algorithm**: Choose between GAE, GRPO, Reinforce++, and more.
*   **Trainer**: Manage experiment names, logging (WandB, Tensorboard, etc.), and checkpointing.

![Configuration Section Screenshot Placeholder](figure/screenshot.png)

