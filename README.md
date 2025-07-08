# TRL Model Fine-Tuner UI (Local MPS/CPU) 🚀

## Overview

This project provides a user-friendly Streamlit web interface to guide users through the lifecycle of fine-tuning Hugging Face Transformer models using the TRL (Transformer Reinforcement Learning) library. It enables **actual local training** on Apple Silicon (M-series Macs via MPS) or CPU, covering model and dataset selection, trainer configuration (including PEFT/LoRA for efficiency), training execution, and model saving.

## Key Features

*   **Step-by-Step Workflow:** Guides users from configuration to actual training and results.
*   **Local Training:** Supports training on Apple Silicon (MPS) for GPU acceleration or CPU.
*   **Model Selection:** Choose models from the Hugging Face Hub or specify local paths. Includes accessibility checks.
*   **Dataset Handling:**
    *   Load datasets from the Hugging Face Hub.
    *   Upload custom datasets (JSONL, CSV, TXT).
    *   Select train/validation splits.
    *   Guidance on expected dataset formats for different trainers.
*   **Trainer Configuration:**
    *   Select from implemented TRL trainers: `SFTTrainer`, `DPOTrainer`.
    *   Configure common `TrainingArguments` (learning rate, epochs, batch size, etc.).
    *   PEFT/LoRA configuration options for efficient fine-tuning (QLoRA is disabled due to MPS incompatibility with `bitsandbytes`).
    *   Trainer-specific parameter settings.
*   **Actual Training Execution:**
    *   "Start Training" initiates the training loop on the selected local device (MPS/CPU).
    *   Displays real-time logs from the training process.
    *   Option to "Cancel" the training (attempts to halt the training thread).
*   **Model Saving & Hub Integration:**
    *   Saves the fine-tuned model (adapters if using PEFT, or full model) locally.
    *   Option to push the saved model to the Hugging Face Hub.
*   **State Management:** Persists user configurations across steps within a session.
*   **"Start Over" Functionality:** Easily reset the application state.

## Running the Application

1.  **Installation:**
    *   Clone the repository (if applicable).
    *   It's highly recommended to set up a Python virtual environment.
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   For detailed installation instructions, especially regarding PyTorch for MPS, please see `INSTALL.md`.

2.  **Launch Streamlit App:**
    ```bash
    python -m streamlit run app.py
    ```
    Or, if the `streamlit` command is directly in your PATH:
    ```bash
    streamlit run app.py
    ```
    The application should open in your web browser (usually `http://localhost:8501`).

## User Workflow

The application is divided into the following steps, accessible via the sidebar:

1.  **Home:** Overview and device (MPS/CPU) detection.
2.  **1. Configure Model & Dataset:** Set up your base model and data.
3.  **2. Configure Trainer & Parameters:** Choose your TRL trainer and define training hyperparameters, including LoRA settings.
4.  **3. Start Training:** Initiate the training process on your local machine and monitor its progress.
5.  **4. View Results & Save:** Review training outcomes, save the model locally, or push it to the Hub.

---

This UI aims to simplify interaction with the TRL library for local fine-tuning on Apple Silicon or CPU. Remember that training large models can be resource-intensive and time-consuming.
