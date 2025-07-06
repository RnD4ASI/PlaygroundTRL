# Installation Guide for TRL Model Fine-Tuner UI

This guide provides detailed instructions for setting up and running the TRL Model Fine-Tuner UI application.

## Prerequisites

*   **Python:** Python 3.9 or higher is recommended. You can check your Python version by running:
    ```bash
    python --version
    ```
    or
    ```bash
    python3 --version
    ```

## Setup Instructions

### 1. Clone the Repository (if applicable)

If you have the project files in a repository, clone it to your local machine:
```bash
git clone <repository_url>
cd <repository_directory>
```
If you just have the `app.py` and other files, navigate to that directory.

### 2. Create a Python Virtual Environment (Recommended)

Using a virtual environment is highly recommended to manage dependencies and avoid conflicts with other Python projects.

*   **Create the virtual environment:**
    ```bash
    python -m venv .venv
    ```
    (Replace `.venv` with your preferred environment name, e.g., `trl_ui_env`)

*   **Activate the virtual environment:**
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    You should see the virtual environment's name in your shell prompt (e.g., `(.venv) user@host:...$`).

### 3. Install Dependencies

Once your virtual environment is activated, install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
This command will install Streamlit, TRL, Transformers, and other necessary libraries.

**Notes on Specific Dependencies:**

*   **`torch` (PyTorch):**
    *   The `requirements.txt` file lists `torch>=2.0.0`. By default, `pip` will try to install a version compatible with your system, which might include CUDA support if you have an NVIDIA GPU and compatible drivers.
    *   **For CPU-only environments (or if you encounter CUDA issues):** You can install a CPU-specific version of PyTorch if needed. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your OS and package manager (pip/conda). For example:
        ```bash
        # Example for pip, CPU only on Linux/Windows
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
        Since this application **simulates** training, a CPU-only version of PyTorch is sufficient for the UI to function.
    *   **For actual GPU training (outside this simulated UI):** Ensure your NVIDIA drivers, CUDA toolkit, and cuDNN are compatible with the PyTorch version you install.

*   **`bitsandbytes`:**
    *   This package is listed for QLoRA support. For actual 4-bit training, it requires a compatible CUDA environment.
    *   Installation can sometimes be tricky. If you encounter issues and are only using the simulated UI, you might be able to proceed without it or with a CPU-only version if one exists that satisfies TRL's import checks, but full QLoRA functionality will not be available.

### 4. Running the Application

Once dependencies are installed, you can run the Streamlit application:

*   **Recommended command:**
    ```bash
    python -m streamlit run app.py
    ```
    This method is generally more robust in finding the Streamlit module.

*   **Alternative command (if `streamlit` is in your PATH):**
    ```bash
    streamlit run app.py
    ```

The application should open in your default web browser, usually at `http://localhost:8501`.

## Troubleshooting Common Issues

*   **`streamlit: command not found` or `python: No module named streamlit`:**
    1.  **Ensure your virtual environment is activated.** This is the most common cause.
    2.  **Verify Streamlit installation:**
        ```bash
        pip show streamlit
        ```
        If it's not listed or the location seems off, try reinstalling it within your activated virtual environment:
        ```bash
        pip install --force-reinstall streamlit
        ```
    3.  **Use `python -m streamlit run app.py`:** This explicitly tells Python to run Streamlit as a module and is often more reliable than relying on the `streamlit` script being in the system PATH, especially within virtual environments or complex Python setups.

*   **Issues with `torch` or `bitsandbytes` CUDA versions:**
    *   As mentioned, this application simulates training. If you are setting up an environment for *actual* TRL training, these errors usually mean there's a mismatch between your CUDA toolkit, NVIDIA drivers, and the installed versions of these packages. Consult their respective GitHub repositories and the NVIDIA documentation for compatibility matrices.
    *   For this UI's simulated purposes, CPU-only versions or ensuring the basic Python parts of the libraries install correctly is the main goal. The UI itself does not perform CUDA operations.

*   **"No space left on device" during `pip install`:**
    *   This can happen in resource-constrained environments (like some cloud sandboxes or Docker containers with small disk allocations).
    *   **Solutions:**
        *   Try to free up disk space.
        *   Install packages one by one or in smaller groups.
        *   Use the `--no-cache-dir` option with `pip install` to prevent caching large wheel files:
            ```bash
            pip install --no-cache-dir -r requirements.txt
            ```
        *   If installing `torch` with CUDA is the issue, try installing a CPU-only version first if that's acceptable for your immediate needs (see `torch` notes above).

## Next Steps

Once the application is running, you can start configuring your (simulated) fine-tuning job by following the steps in the UI sidebar. Refer to the `README.md` for an overview of the application's features and workflow.
```
