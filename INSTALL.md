# Installation Guide for TRL Model Fine-Tuner UI (Local MPS/CPU)

This guide provides detailed instructions for setting up and running the TRL Model Fine-Tuner UI application on your local machine, with a focus on Apple Silicon (M-series Macs with MPS) and CPU setups.

## Prerequisites

*   **Python:** Python 3.9 or higher is recommended.
*   **For MPS (Apple Silicon GPU):** macOS 12.3 or later is required for PyTorch MPS support.
*   **Git (Optional):** If cloning from a repository.

## Setup Instructions

### 1. Clone the Repository (if applicable)

If you have the project files in a repository, clone it:
```bash
git clone <repository_url>
cd <repository_directory>
```
Otherwise, navigate to the directory containing `app.py` and other project files.

### 2. Create a Python Virtual Environment (Highly Recommended)

This isolates project dependencies.

*   **Create:**
    ```bash
    python3 -m venv .venv
    ```
    (Using `python3` is often more explicit on macOS)
*   **Activate:**
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows (for CPU-only usage):
        ```bash
        .\.venv\Scripts\activate
        ```
    Your shell prompt should now indicate the active environment (e.g., `(.venv) user@host:...$`).

### 3. Install Dependencies from `requirements.txt`

With the virtual environment activated, install packages:
```bash
pip install -r requirements.txt
```
This will install Streamlit, TRL, Transformers, PyTorch, PEFT, etc.

**Important Notes on PyTorch (`torch`):**

*   **For MPS (Apple Silicon):**
    *   The `requirements.txt` lists `torch>=2.0.0`. Standard `pip install torch` on an M-series Mac should automatically install a version with MPS support.
    *   Verify MPS availability in Python after installation:
        ```python
        import torch
        if torch.backends.mps.is_available():
            print("PyTorch MPS backend is available!")
        else:
            print("PyTorch MPS backend not available.")
        ```
    *   If you encounter issues, you might need to consult the [official PyTorch website](https://pytorch.org/get-started/locally/) for specific commands tailored to your macOS version and desired PyTorch version.

*   **For CPU-only:**
    *   If you are on a non-Mac system or wish to force CPU-only PyTorch, `pip install torch` usually works. If you need a specific CPU-only build, refer to the PyTorch website. The application will detect if MPS is unavailable and default to CPU.

**Notes on Other Dependencies:**

*   **`peft`:** Used for LoRA. QLoRA support (which relies on `bitsandbytes`) is **disabled** in this application as `bitsandbytes` is not generally compatible with MPS.
*   **`sentencepiece`:** Often required by various tokenizers.

### 4. Running the Application

Once dependencies are installed:

*   **Recommended command:**
    ```bash
    python3 -m streamlit run app.py
    ```
*   Alternative (if `streamlit` is in your PATH):
    ```bash
    streamlit run app.py
    ```

The application should open in your web browser (usually at `http://localhost:8501`). The Home page will indicate if MPS is detected and being used.

## Troubleshooting

*   **`streamlit: command not found` or `No module named streamlit`:**
    1.  Ensure your virtual environment is **activated**.
    2.  Verify Streamlit installation: `pip show streamlit`. If missing or incorrect, reinstall: `pip install --force-reinstall streamlit`.
    3.  Always prefer `python3 -m streamlit run app.py`.

*   **PyTorch MPS Issues (`torch.backends.mps.is_available()` is `False`):**
    1.  Ensure you are on macOS 12.3+ and have a compatible PyTorch version (>=1.12, ideally >=2.0).
    2.  Try reinstalling PyTorch: `pip uninstall torch`, then `pip install torch`.
    3.  Check for any known issues on the PyTorch GitHub repository related to your specific macOS and PyTorch versions.
    4.  The application will fall back to CPU if MPS is not usable.

*   **Insufficient Memory/Performance Issues:**
    *   Fine-tuning language models is resource-intensive. Even with MPS, large models or large batch sizes can exhaust unified memory or lead to slow performance.
    *   **Recommendations:**
        *   Start with smaller models (e.g., `gpt2`, `distilgpt2`).
        *   Use smaller batch sizes (e.g., `per_device_train_batch_size = 1`).
        *   Reduce `max_seq_length`.
        *   Use LoRA (PEFT) to significantly reduce trainable parameters.
        *   Train for fewer epochs initially.

## Next Steps

After successful installation, the application's Home page will guide you. Refer to `README.md` for an application overview. Remember that while this UI simplifies the process, understanding the underlying TRL and Hugging Face concepts is beneficial for effective fine-tuning.
