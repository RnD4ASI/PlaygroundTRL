# TRL Model Fine-Tuner UI (Simulated) 🚀

## Overview

This project provides a user-friendly Streamlit web interface to guide users through the lifecycle of fine-tuning Hugging Face Transformer models using the TRL (Transformer Reinforcement Learning) library. It covers model and dataset selection, trainer configuration (including PEFT/LoRA for efficiency), and (simulated) training execution and model saving.

**🔴 IMPORTANT NOTE ON SIMULATION:**
Due to the limitations of typical sandbox environments (like the one this application might have been developed or demonstrated in), **actual model training, GPU utilization, and saving of large model files are SIMULATED.**
The primary purpose of this application in such a context is to:
1.  Demonstrate the user interface and workflow for TRL.
2.  Allow users to explore and configure various parameters for different TRL trainers.
3.  Show how one might structure such a fine-tuning assistant.
For actual training, you would need to run this application in an environment with appropriate hardware (GPU), sufficient disk space, and all Python packages (like `torch` with CUDA, `bitsandbytes` with CUDA) correctly installed.

## Key Features

*   **Step-by-Step Workflow:** Guides users from configuration to (simulated) training and results.
*   **Model Selection:** Choose models from the Hugging Face Hub or specify local paths. Includes basic accessibility checks.
*   **Dataset Handling:**
    *   Load datasets from the Hugging Face Hub.
    *   Upload custom datasets (JSONL, CSV, TXT).
    *   Select train/validation splits.
    *   Guidance on expected dataset formats for different trainers.
*   **Trainer Configuration:**
    *   Select from common TRL trainers: `SFTTrainer`, `DPOTrainer` (others like `RewardTrainer`, `PPOTrainer`, `GRPOTrainer` are listed but less configured in this version).
    *   Configure common `TrainingArguments` (learning rate, epochs, batch size, etc.).
    *   Extensive PEFT/LoRA/QLoRA configuration options for efficient fine-tuning.
    *   Trainer-specific parameter settings.
*   **Simulated Training:**
    *   "Start Training" initiates a simulated training loop.
    *   Displays mock training logs and a progress bar.
    *   Option to "Cancel" the simulated training.
*   **Simulated Results & Saving:**
    *   Displays mock final metrics.
    *   Simulates saving the model to a local path (creates dummy config files).
    *   Simulates pushing the model to the Hugging Face Hub.
*   **State Management:** Persists user configurations across steps within a session.
*   **"Start Over" Functionality:** Easily reset the application state to begin a new configuration.

## Running the Application

1.  **Installation:**
    *   Clone the repository.
    *   It's highly recommended to set up a Python virtual environment.
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   For detailed installation instructions and troubleshooting, please see `INSTALL.md`.

2.  **Launch Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    Or, if the `streamlit` command is not directly in your PATH:
    ```bash
    python -m streamlit run app.py
    ```
    The application should open in your web browser.

## User Workflow

The application is divided into the following steps, accessible via the sidebar navigation:

1.  **Home:** Overview and important notes.
2.  **1. Configure Model & Dataset:** Set up your base model and data.
3.  **2. Configure Trainer & Parameters:** Choose your TRL trainer and define all training hyperparameters, including any PEFT/LoRA settings.
4.  **3. Start Training (Simulated):** Initiate the simulated training process and monitor its mock progress.
5.  **4. View Results & Save (Simulated):** Review mock final metrics and simulate saving the model or pushing it to the Hub.

---

This UI aims to simplify the interaction with the powerful TRL library, making advanced fine-tuning techniques more accessible. Remember the simulation aspect if running in a restricted environment. For real training, ensure your environment meets all hardware and software prerequisites.
