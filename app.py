import streamlit as st
from huggingface_hub import HfApi, RepositoryNotFoundError, GatedRepoError
from datasets import load_dataset_builder, get_dataset_split_names
from transformers import AutoConfig
import os
import json
import time # For simulation

# --- Application Constants ---
APP_TITLE = "TRL Model Fine-Tuner (Simulated) 🚀"
SIMULATION_WARNING = "⚠️ **Important:** This application runs in a restricted environment. Actual model training, GPU usage, and large file operations are **simulated**. Use this tool to understand the workflow and parameter configuration for TRL."

# --- Session State Initialization ---
def init_session_state():
    """Initializes or re-initializes the session state with default values."""
    defaults = {
        "app_mode": "Home", # Current page/step of the application
        # Step 1: Model and Dataset Config
        "model_name": "gpt2",
        "dataset_name": "imdb",
        "dataset_path": None,  # Stores UploadedFile object for custom datasets
        "custom_dataset_file_name": None, # Name of the uploaded custom file
        "train_split": "train",
        "validation_split": "test",
        "dataset_load_success": False, # Flag indicating if dataset is accessible/uploaded
        "model_load_success": False,   # Flag indicating if model is accessible
        "available_splits": [],        # List of splits for a Hugging Face dataset
        "hf_token": "",                # Optional Hugging Face token
        # Step 2: Trainer Configuration
        "trainer_config": {
            "selected_trainer": "SFTTrainer",
            "output_dir": "./results_simulated", # Default output directory
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-5,
            "logging_steps": 10,
            "save_steps": 500,
            "evaluation_strategy": "no",
            "eval_steps": 500,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "optim": "adamw_hf",
            "use_peft": True, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,v_proj", "use_qlora": False,
            "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16",
            "sft_packing": False, "sft_max_seq_length": 1024, "sft_dataset_text_field": "text",
            "dpo_beta": 0.1, "dpo_loss_type": "sigmoid"
        },
        # Step 3: Training State
        "training_started": False,
        "training_complete": False,
        "training_logs": "Training logs will appear here once training starts.",
        "cancel_training_flag": False, # Flag to signal cancellation to the simulation
        "training_progress": 0.0,
        "navigated_away_after_training": False, # Tracks if user moved from training page after completion
        # Step 4: Results and Saving
        "model_save_path": "", # Path for saving the simulated model
        "hf_repo_name": ""     # Repo name for simulated Hub push
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, dict):
                st.session_state[key] = value.copy() # Ensure dicts are copied
            else:
                st.session_state[key] = value

# --- Helper Functions for API/File Checks ---
def check_model_exists(model_name, token=None):
    """Checks if a model exists on Hugging Face Hub or locally (by trying to load its config)."""
    if not model_name.strip():
        return False, "Model name/path cannot be empty."
    try:
        AutoConfig.from_pretrained(model_name, token=token)
        return True, f"✅ Model '{model_name}' seems accessible."
    except RepositoryNotFoundError:
        return False, f"❌ Model '{model_name}' not found on Hugging Face Hub."
    except GatedRepoError:
        return False, f" gated. Please provide a HF token or log in via `huggingface-cli login`."
    except OSError:
        return False, f"❌ Model '{model_name}' not found locally or invalid path."
    except Exception as e:
        return False, f"❌ Error checking model '{model_name}': {str(e)}"

def check_dataset_exists(dataset_name, token=None):
    """Checks if a dataset exists on Hugging Face Hub (by trying to load its builder)."""
    if not dataset_name.strip():
        return False, "Dataset name cannot be empty."
    try:
        load_dataset_builder(dataset_name, token=token) # Lightweight check
        return True, f"✅ Dataset '{dataset_name}' seems accessible."
    except FileNotFoundError: # Should not happen for Hub datasets with load_dataset_builder
        return False, f"❌ Dataset '{dataset_name}' not found (unexpected FileNotFoundError)."
    except RepositoryNotFoundError:
         return False, f"❌ Dataset '{dataset_name}' not found on Hugging Face Hub."
    except GatedRepoError:
        return False, f" gated. Please provide a HF token or log in via `huggingface-cli login`."
    except Exception as e:
        return False, f"❌ Error checking dataset '{dataset_name}': {str(e)}"

def get_dataset_splits_info(dataset_name_or_path, is_custom_file=False, token=None):
    """Retrieves available splits for a Hugging Face dataset."""
    if is_custom_file:
        return ["train"], "For uploaded files, 'train' is assumed. Ensure your file contains the training data."
    if not dataset_name_or_path.strip():
        return [], "Dataset name is empty."
    try:
        split_names = get_dataset_split_names(dataset_name_or_path, token=token)
        if not split_names:
            return [], "No standard splits found for this dataset. Check dataset configuration on the Hub."
        return split_names, f"Available splits: {split_names}"
    except Exception as e:
        return [], f"Could not retrieve splits for '{dataset_name_or_path}': {str(e)}"

# --- Main Application Router ---
def main():
    """Main function to render the Streamlit application pages."""
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    init_session_state() # Ensure state is initialized on first run or after clearing

    st.title(APP_TITLE)
    st.sidebar.title("Navigation")

    app_mode_options = ["Home", "1. Configure Model & Dataset",
                        "2. Configure Trainer & Parameters",
                        "3. Start Training (Simulated)", "4. View Results & Save (Simulated)"]

    # Ensure app_mode is valid, default to Home if not (e.g., after state clear)
    if st.session_state.app_mode not in app_mode_options:
        st.session_state.app_mode = "Home"

    current_selection_index = app_mode_options.index(st.session_state.app_mode)
    st.session_state.app_mode = st.sidebar.selectbox(
        "Choose the step:", app_mode_options, index=current_selection_index, key="sidebar_nav"
    )

    # Page routing
    if st.session_state.app_mode == "Home":
        render_home_page()
    elif st.session_state.app_mode == "1. Configure Model & Dataset":
        render_config_model_dataset_page()
    elif st.session_state.app_mode == "2. Configure Trainer & Parameters":
        render_config_trainer_parameters_page()
    elif st.session_state.app_mode == "3. Start Training (Simulated)":
        render_training_page()
    elif st.session_state.app_mode == "4. View Results & Save (Simulated)":
        render_results_page()

# --- Page Rendering Functions ---

def render_home_page():
    """Renders the Home page with an overview and warnings."""
    st.header("Welcome to the TRL Fine-Tuning Assistant!")
    st.markdown(SIMULATION_WARNING) # Prominent warning about simulation
    st.markdown("""
        This tool guides you through configuring and (simulating) the fine-tuning of
        Hugging Face Transformers models using the TRL (Transformer Reinforcement Learning) library.
        Use the sidebar to navigate through the steps.
    """)
    st.markdown("### End-to-End Workflow:")
    st.markdown("""
        1.  **Configure Model & Dataset:** Select your base model and the dataset.
        2.  **Configure Trainer & Parameters:** Choose a TRL trainer and set hyperparameters.
        3.  **Start Training (Simulated):** Launch and monitor the (simulated) fine-tuning.
        4.  **View Results & Save (Simulated):** Review (mock) outcomes and (simulate) saving.
    """)
    st.info("💡 **Tip:** Hover over input fields or options for more details where available. Consult the official TRL documentation for in-depth explanations of parameters.")

def render_config_model_dataset_page():
    """Renders Step 1: Model and Dataset Configuration page."""
    st.header("Step 1: Configure Model & Dataset")
    st.markdown("Specify the base model and the dataset you want to use for fine-tuning.")

    with st.expander("Hugging Face Token (Optional)", expanded=False):
        st.session_state.hf_token = st.text_input(
            "Enter your Hugging Face Token (for gated models/datasets)",
            type="password", value=st.session_state.hf_token, key="hf_token_input"
        )
        st.caption("A read-access token is usually sufficient. This can also be set via `huggingface-cli login` in your environment if the app is run locally.")

    token_to_use = st.session_state.hf_token if st.session_state.hf_token.strip() else None

    st.subheader("Model Selection")
    st.session_state.model_name = st.text_input(
        "Model Name or Local Path (e.g., 'gpt2', 'mistralai/Mistral-7B-v0.1')",
        value=st.session_state.model_name, key="model_name_input"
    )
    if st.button("Check Model Accessibility", key="check_model_btn"):
        st.session_state.model_load_success, model_status_msg = check_model_exists(st.session_state.model_name, token=token_to_use)
        if st.session_state.model_load_success: st.success(model_status_msg)
        else: st.error(model_status_msg)

    st.subheader("Dataset Selection")
    dataset_source_key = "dataset_source_radio"
    current_dataset_source_idx = 0 if st.session_state.get(dataset_source_key, "Hugging Face Hub") == "Hugging Face Hub" else 1
    dataset_source = st.radio(
        "Select Dataset Source:", ("Hugging Face Hub", "Upload Custom File"),
        key=dataset_source_key, index=current_dataset_source_idx
    )

    if dataset_source == "Hugging Face Hub":
        if st.session_state.custom_dataset_file_name:
            st.session_state.custom_dataset_file_name = None; st.session_state.dataset_path = None
            st.session_state.dataset_load_success = False
            st.info("Switched to Hugging Face Hub. Custom upload data cleared.")

        st.session_state.dataset_name = st.text_input(
            "Dataset Name from Hugging Face Hub (e.g., 'imdb', 'trl-lib/ultrafeedback_binarized')",
            value=st.session_state.dataset_name, key="dataset_name_input"
        )
        if st.button("Check Dataset & Get Splits", key="check_dataset_btn"):
            st.session_state.dataset_load_success, dataset_status = check_dataset_exists(st.session_state.dataset_name, token=token_to_use)
            if st.session_state.dataset_load_success:
                st.success(dataset_status)
                splits, splits_msg = get_dataset_splits_info(st.session_state.dataset_name, token=token_to_use)
                st.session_state.available_splits = splits
                st.info(splits_msg) # Display message about splits found or not
            else:
                st.error(dataset_status)
                st.session_state.available_splits = []

    else: # Upload Custom File
        if st.session_state.dataset_name:
            st.session_state.dataset_name = ""; st.session_state.available_splits = []
            st.session_state.dataset_load_success = False
            st.info("Switched to Custom File Upload. Hugging Face Hub dataset name cleared.")

        uploaded_file = st.file_uploader(
            "Upload Dataset File (JSONL, CSV, TXT)", type=['jsonl', 'csv', 'json', 'txt'], key="custom_file_uploader"
        )
        if uploaded_file is not None:
            st.session_state.dataset_path = uploaded_file
            st.session_state.custom_dataset_file_name = uploaded_file.name
            st.success(f"Uploaded '{uploaded_file.name}'.")
            st.session_state.dataset_load_success = True
            splits, splits_msg = get_dataset_splits_info(None, is_custom_file=True)
            st.session_state.available_splits = splits # Should be ['train']
            st.info(splits_msg)
            if "train" in splits: st.session_state.train_split = "train"
            st.session_state.validation_split = None
        elif st.session_state.custom_dataset_file_name: # If uploader is cleared
            st.session_state.custom_dataset_file_name = None; st.session_state.dataset_path = None
            st.session_state.dataset_load_success = False

    with st.expander("Dataset Format Guidance (Important!)", expanded=False):
        st.markdown("""
        *   **SFTTrainer:** Needs a `text` field (each entry is a full string for an example) OR a `messages` field (list of dicts with `role` and `content`). Can also be configured for prompt/completion pairs.
        *   **DPOTrainer:** Strictly requires `prompt`, `chosen` (preferred response), and `rejected` (less-preferred response) columns.
        *   **RewardTrainer:** Typically needs columns like `chosen` and `rejected` or `sentence1`, `sentence2` for comparison.
        *   Ensure your uploaded CSV/JSONL file structure matches the requirements of the trainer you intend to use. Column names are key.
        """)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.available_splits:
            try: current_train_idx = st.session_state.available_splits.index(st.session_state.train_split)
            except ValueError: current_train_idx = 0 # Default if not found
            st.session_state.train_split = st.selectbox(
                "Select Train Split:", options=st.session_state.available_splits, index=current_train_idx, key="train_split_select"
            )
        else:
            st.session_state.train_split = st.text_input("Train Split Name", value=st.session_state.train_split, key="train_split_input")

    with col2:
        if st.session_state.available_splits:
            val_options = ["None"] + st.session_state.available_splits
            try: current_val_idx = val_options.index(st.session_state.validation_split if st.session_state.validation_split else "None")
            except ValueError: current_val_idx = 0
            selected_val_split = st.selectbox(
                "Select Validation Split (Optional):", options=val_options, index=current_val_idx, key="val_split_select"
            )
            st.session_state.validation_split = None if selected_val_split == "None" else selected_val_split
        else:
            st.session_state.validation_split = st.text_input("Validation Split Name (Optional)", value=st.session_state.validation_split or "", key="val_split_input")
            if not st.session_state.validation_split.strip(): st.session_state.validation_split = None

    st.subheader("Current Configuration Summary (Step 1):")
    ds_display_name = st.session_state.dataset_name if dataset_source == "Hugging Face Hub" else st.session_state.custom_dataset_file_name
    st.json({
        "Model Name/Path": st.session_state.model_name, "Model Accessible": "✅ Yes" if st.session_state.model_load_success else "❌ No",
        "Dataset Source": dataset_source, "Dataset Name/File": ds_display_name if ds_display_name else "Not Set",
        "Dataset Accessible/Uploaded": "✅ Yes" if st.session_state.dataset_load_success else "❌ No",
        "Train Split": st.session_state.train_split if st.session_state.train_split else "Not Set",
        "Validation Split": st.session_state.validation_split if st.session_state.validation_split else "Not Set"
    })
    if not (st.session_state.model_load_success and st.session_state.dataset_load_success and st.session_state.train_split):
        st.error("Configuration incomplete. Please ensure model & dataset are accessible and train split is defined.")

def render_config_trainer_parameters_page():
    """Renders Step 2: Trainer and Hyperparameter Configuration page."""
    st.header("Step 2: Configure Trainer & Training Parameters")
    st.markdown("Select a TRL trainer and configure its parameters, including PEFT/LoRA for efficient tuning.")

    if not st.session_state.get("model_load_success") or not st.session_state.get("dataset_load_success"):
        st.warning("Please complete Step 1 (Model & Dataset Configuration) successfully before proceeding.")
        if st.button("Go to Step 1", key="cfg_to_step1_btn"): st.session_state.app_mode = "1. Configure Model & Dataset"; st.experimental_rerun()
        return

    tc = st.session_state.trainer_config # Shortcut for trainer configuration dictionary

    st.subheader("Trainer Selection")
    tc["selected_trainer"] = st.selectbox(
        "Select TRL Trainer:",
        options=["SFTTrainer", "DPOTrainer", "RewardTrainer", "PPOTrainer", "GRPOTrainer"], # Add more as supported
        index=["SFTTrainer", "DPOTrainer", "RewardTrainer", "PPOTrainer", "GRPOTrainer"].index(tc["selected_trainer"]),
        key="trainer_select"
    )

    st.subheader("Common Training Arguments")
    col1, col2 = st.columns(2)
    with col1:
        tc["output_dir"] = st.text_input("Output Directory (Simulated)", value=tc["output_dir"], key="output_dir_input")
        tc["num_train_epochs"] = st.number_input("Number of Training Epochs", min_value=1, step=1, value=tc["num_train_epochs"], key="epochs_input")
        tc["per_device_train_batch_size"] = st.number_input("Train Batch Size (per device)", min_value=1, step=1, value=tc["per_device_train_batch_size"], key="batch_size_input")
        tc["learning_rate"] = st.number_input("Learning Rate (e.g., 5e-5)", value=tc["learning_rate"], format="%e", key="lr_input")
    with col2:
        tc["logging_steps"] = st.number_input("Logging Steps", min_value=1, step=10, value=tc["logging_steps"], key="logging_steps_input")
        tc["save_steps"] = st.number_input("Save Steps (0 for no checkpoint saving)", min_value=0, step=100, value=tc["save_steps"], key="save_steps_input")
        tc["evaluation_strategy"] = st.selectbox("Evaluation Strategy", options=["no", "steps", "epoch"], index=["no", "steps", "epoch"].index(tc["evaluation_strategy"]), key="eval_strategy_select")
        if tc["evaluation_strategy"] == "steps":
            tc["eval_steps"] = st.number_input("Evaluation Steps", min_value=1, step=50, value=tc["eval_steps"], key="eval_steps_input")

    with st.expander("Advanced Training Arguments", expanded=False):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            tc["gradient_accumulation_steps"] = st.number_input("Gradient Accumulation Steps", min_value=1, step=1, value=tc["gradient_accumulation_steps"], key="grad_accum_input")
            tc["optim"] = st.selectbox("Optimizer", options=["adamw_hf", "adamw_torch", "adafactor", "sgd", "adamw_bnb_8bit"], index=["adamw_hf", "adamw_torch", "adafactor", "sgd", "adamw_bnb_8bit"].index(tc["optim"]), key="optimizer_select")
        with adv_col2:
            tc["warmup_steps"] = st.number_input("Warmup Steps", min_value=0, step=50, value=tc["warmup_steps"], key="warmup_steps_input")
            tc["max_grad_norm"] = st.number_input("Max Gradient Norm", min_value=0.0, value=tc["max_grad_norm"], format="%.1f", key="max_grad_norm_input")
            tc["lr_scheduler_type"] = st.selectbox("LR Scheduler Type", options=["linear", "cosine", "constant", "constant_with_warmup"], index=["linear", "cosine", "constant", "constant_with_warmup"].index(tc["lr_scheduler_type"]), key="lr_scheduler_select")

    st.subheader("PEFT (LoRA / QLoRA) Configuration")
    tc["use_peft"] = st.checkbox("Enable PEFT (LoRA/QLoRA)", value=tc["use_peft"], key="use_peft_checkbox")
    if tc["use_peft"]:
        peft_col1, peft_col2 = st.columns(2)
        with peft_col1:
            tc["lora_r"] = st.number_input("LoRA r (Rank)", min_value=1, step=1, value=tc["lora_r"], key="lora_r_input")
            tc["lora_alpha"] = st.number_input("LoRA alpha", min_value=1, step=1, value=tc["lora_alpha"], key="lora_alpha_input")
            tc["lora_dropout"] = st.slider("LoRA Dropout", min_value=0.0, max_value=1.0, value=tc["lora_dropout"], step=0.01, key="lora_dropout_slider")
        with peft_col2:
            tc["lora_target_modules"] = st.text_input(
                "LoRA Target Modules (comma-separated, e.g., q_proj,v_proj or 'all-linear')",
                value=tc["lora_target_modules"], key="lora_target_modules_input"
            )
            st.caption("Tip: Use 'all-linear' to target common linear layers (model-dependent).")

            tc["use_qlora"] = st.checkbox("Enable QLoRA (4-bit Quantization)", value=tc["use_qlora"], key="use_qlora_checkbox")
            if tc["use_qlora"]:
                tc["bnb_4bit_quant_type"] = st.selectbox("BitsAndBytes 4-bit Quant Type", options=["nf4", "fp4"], index=["nf4", "fp4"].index(tc["bnb_4bit_quant_type"]), key="bnb_quant_select")
                tc["bnb_4bit_compute_dtype"] = st.selectbox("BitsAndBytes 4-bit Compute Dtype", options=["float16", "bfloat16"], index=["float16", "bfloat16"].index(tc["bnb_4bit_compute_dtype"]), key="bnb_compute_select") # float32 might be too slow
                st.info("💡 QLoRA requires `bitsandbytes` to be correctly installed with CUDA support for actual training.")

    st.subheader(f"{tc['selected_trainer']} Specific Arguments")
    # Trainer specific arguments
    if tc["selected_trainer"] == "SFTTrainer":
        tc["sft_packing"] = st.checkbox("Enable Packing (efficiently packs short examples)", value=tc["sft_packing"], key="sft_packing_checkbox")
        tc["sft_max_seq_length"] = st.number_input("Max Sequence Length", min_value=32, step=32, value=tc["sft_max_seq_length"], key="sft_max_seq_input")
        tc["sft_dataset_text_field"] = st.text_input(
            "Dataset Text Field for SFT (e.g., 'text', 'prompt', or field with 'messages' list)",
            value=tc["sft_dataset_text_field"], key="sft_text_field_input"
        )
        st.caption("This is the column SFTTrainer uses for training data. For conversational format, provide the column name containing the list of message dicts (e.g., 'messages').")

    elif tc["selected_trainer"] == "DPOTrainer":
        tc["dpo_beta"] = st.slider("DPO Beta (regularization, 0.1-0.5 typical)", min_value=0.0, max_value=1.0, value=tc["dpo_beta"], step=0.01, key="dpo_beta_slider")
        tc["dpo_loss_type"] = st.selectbox("DPO Loss Type", options=["sigmoid", "hinge", "ipo", "kpo_pair"], index=["sigmoid", "hinge", "ipo", "kpo_pair"].index(tc["dpo_loss_type"]), key="dpo_loss_select")
        st.info("ℹ️ DPOTrainer expects dataset columns: `prompt`, `chosen`, and `rejected`.")
    # Add elif blocks for RewardTrainer, PPOTrainer, GRPOTrainer specific args here when implemented

    st.subheader("Current Configuration Summary (Step 2):")
    # Create a filtered copy for display
    display_config = {"Trainer": tc["selected_trainer"], "Output Dir": tc["output_dir"], "Epochs": tc["num_train_epochs"], "LR": tc["learning_rate"]}
    if tc["use_peft"]: display_config["PEFT/LoRA"] = f"Enabled (r={tc['lora_r']}, alpha={tc['lora_alpha']})"
    if tc["use_qlora"] and tc["use_peft"]: display_config["QLoRA"] = "Enabled" # QLoRA implies LoRA
    # Add specific args to summary if they exist for the selected trainer
    if tc["selected_trainer"] == "SFTTrainer": display_config["SFT Max Seq Len"] = tc.get("sft_max_seq_length")
    if tc["selected_trainer"] == "DPOTrainer": display_config["DPO Beta"] = tc.get("dpo_beta")
    st.json(display_config, expanded=False)
    st.success("Trainer configuration updated. You can proceed to the next step when ready.")

def render_training_page():
    """Renders Step 3: Training Execution and Monitoring page (Simulated)."""
    st.header("Step 3: Start Training & Monitor (Simulated)")
    st.markdown(SIMULATION_WARNING) # Remind user about simulation

    if not st.session_state.get("model_load_success", False) or \
       not st.session_state.get("dataset_load_success", False) or \
       "trainer_config" not in st.session_state:
        st.warning("Please complete Steps 1 and 2 (Model, Dataset, & Trainer Configuration) successfully before proceeding.")
        if st.button("Go to Step 1", key="train_to_step1_btn"): st.session_state.app_mode = "1. Configure Model & Dataset"; st.experimental_rerun()
        elif st.button("Go to Step 2", key="train_to_step2_btn"): st.session_state.app_mode = "2. Configure Trainer & Parameters"; st.experimental_rerun()
        return

    st.subheader("Training Execution Controls")
    log_placeholder = st.empty() # For dynamically updating log text_area
    progress_bar_placeholder = st.empty() # For dynamically updating progress_bar

    col_run, col_cancel = st.columns(2)
    with col_run:
        start_disabled = (st.session_state.training_started and not st.session_state.training_complete) or \
                         (st.session_state.training_complete and not st.session_state.navigated_away_after_training)
        if st.button("▶️ Start Training", disabled=start_disabled, key="start_training_button"):
            st.session_state.training_started = True
            st.session_state.training_complete = False
            st.session_state.cancel_training_flag = False
            st.session_state.training_logs = "INFO: Initializing SIMULATED training...\n"
            st.session_state.training_progress = 0.0
            st.session_state.navigated_away_after_training = False # Reset this flag
            st.info("INFO: Starting SIMULATED training process...")
            st.experimental_rerun() # Rerun to activate the training loop block below

    with col_cancel:
        cancel_disabled = not st.session_state.training_started or st.session_state.training_complete
        if st.button("⏹️ Cancel Training", disabled=cancel_disabled, key="cancel_training_button"):
            st.session_state.cancel_training_flag = True # Signal the simulation loop to stop
            st.session_state.training_logs += "\nINFO: Training cancellation requested by user (simulated).\n"
            st.warning("INFO: Training cancellation requested. Simulation will halt shortly.")
            # The simulation loop will see the flag and exit. UI will update on the next rerun.

    # Always display current logs and progress
    log_placeholder.text_area("Training Logs", value=st.session_state.training_logs, height=400, key="training_log_display_area")
    progress_bar_placeholder.progress(st.session_state.training_progress)

    # This block executes if 'Start Training' was clicked and training is not yet marked complete
    if st.session_state.training_started and not st.session_state.training_complete:
        with st.spinner("SIMULATED Training in progress..."):
            try:
                # Pass the Streamlit elements to the simulation function for updates
                run_simulated_training(log_placeholder, progress_bar_placeholder.progress)
            except Exception as e:
                st.session_state.training_logs += f"\n\n❌ SIMULATION ERROR: {str(e)}\n"
                st.error(f"An error occurred during simulated training: {str(e)}")
                st.session_state.training_complete = True # Mark as complete to stop further attempts
                st.session_state.training_started = False # Stop the training state
            finally:
                # Rerun to update UI based on completion or error state from run_simulated_training
                st.experimental_rerun()

    if st.session_state.training_complete:
        if st.session_state.get("cancel_training_flag", False): st.warning("Training (Simulated) was cancelled.")
        elif "error" in st.session_state.training_logs.lower(): st.error("Training (Simulated) seems to have encountered an issue.")
        else: st.success("Training (Simulated) has finished successfully.")

        if st.button("Proceed to Step 4: View Results & Save", key="training_to_results_btn"):
            st.session_state.app_mode = "4. View Results & Save (Simulated)"
            st.session_state.navigated_away_after_training = True # User is moving on
            st.experimental_rerun()

def run_simulated_training(log_placeholder, progress_bar_update_fn):
    """Simulates a training loop, updating logs and progress bar via passed Streamlit elements/methods."""
    # Initial log messages with config details
    st.session_state.training_logs += f"Model: {st.session_state.model_name}\n"
    st.session_state.training_logs += f"Dataset: {st.session_state.dataset_name or st.session_state.custom_dataset_file_name}\n"
    st.session_state.training_logs += f"Trainer: {st.session_state.trainer_config['selected_trainer']}\n"
    st.session_state.training_logs += "Simulated Parameters:\n"
    for key, value in st.session_state.trainer_config.items():
        st.session_state.training_logs += f"  {key}: {value}\n"

    st.session_state.training_logs += "\n--- SIMULATING: Model and Tokenizer Loading ---\n"
    log_placeholder.text_area("Training Logs", value=st.session_state.training_logs, height=400, key="log_sim_load_model")
    time.sleep(0.5)

    st.session_state.training_logs += "--- SIMULATING: Dataset Preparation ---\n"
    log_placeholder.text_area("Training Logs", value=st.session_state.training_logs, height=400, key="log_sim_prep_data")
    time.sleep(0.5)

    st.session_state.training_logs += f"--- SIMULATING: Initializing {st.session_state.trainer_config['selected_trainer']} ---\n"
    log_placeholder.text_area("Training Logs", value=st.session_state.training_logs, height=400, key="log_sim_init_trainer")
    time.sleep(0.2)

    st.session_state.training_logs += "*** SIMULATED Training Loop Started ***\n"
    log_placeholder.text_area("Training Logs", value=st.session_state.training_logs, height=400, key="log_sim_loop_start")

    total_steps = 100 # Simulate 100 training steps
    for i in range(total_steps):
        if st.session_state.get("cancel_training_flag", False):
            st.session_state.training_logs += "\nINFO: SIMULATED Training halted by cancellation.\n"
            log_placeholder.text_area("Training Logs", value=st.session_state.training_logs, height=400, key="log_sim_cancel_in_loop")
            break
        time.sleep(0.02)
        current_epoch = (i * st.session_state.trainer_config.get('per_device_train_batch_size', 4)) // 50
        mock_loss = 0.5 + (1 / (i + 10))
        mock_lr = st.session_state.trainer_config.get('learning_rate', 5e-5) * (1 - (i / total_steps))

        step_log = f"Epoch: {current_epoch} | Step: {i+1}/{total_steps} | Loss: {mock_loss:.4f} | LR: {mock_lr:.2e} (Simulated)\n"
        st.session_state.training_logs += step_log

        log_lines = st.session_state.training_logs.splitlines()
        display_log_content = "\n".join(log_lines[-100:])
        log_placeholder.text_area("Training Logs", value=display_log_content, height=400, key=f"log_sim_step_{i}")

        st.session_state.training_progress = (i + 1) / total_steps
        progress_bar_update_fn(st.session_state.training_progress) # Call the passed progress update method

    if not st.session_state.get("cancel_training_flag", False):
        st.session_state.training_logs += "\n*** SIMULATED Training Complete ***\n"
        st.session_state.training_logs += "Final metrics (simulated): eval_loss: 0.15, accuracy: 0.95\n"

    log_lines = st.session_state.training_logs.splitlines()
    display_log_content = "\n".join(log_lines[-100:])
    log_placeholder.text_area("Training Logs", value=display_log_content, height=400, key="log_sim_final")

    st.session_state.training_complete = True # Mark as complete
    st.session_state.training_started = False # Reset started flag

def render_results_page():
    """Renders Step 4: View Results and Save Model page (Simulated)."""
    st.header("Step 4: View Results & Save Model (Simulated)")
    st.markdown(SIMULATION_WARNING)

    if not st.session_state.get("training_complete", False):
        st.warning("Training has not been completed yet. Please complete Step 3 first.")
        if st.button("Go to Training (Step 3)", key="results_to_training_btn"):
            st.session_state.app_mode = "3. Start Training (Simulated)"
            st.experimental_rerun()
        return

    st.subheader("Simulated Training Results")
    final_metrics_log = "Metrics: eval_loss: 0.15, accuracy: 0.95 (Simulated Default)"
    if "training_logs" in st.session_state and st.session_state.training_logs:
        for line in reversed(st.session_state.training_logs.splitlines()):
            if "final metrics (simulated)" in line.lower(): final_metrics_log = line; break
    st.markdown(f"**{final_metrics_log}**")
    with st.expander("Show Full Simulated Training Log (from Step 3)", expanded=False):
        st.text_area("Full Log", value=st.session_state.get("training_logs", "No logs available."), height=200, disabled=True, key="results_full_log_display")

    st.subheader("Save Fine-Tuned Model (Simulated)")
    model_filename_default = st.session_state.get("model_name", "model").replace("/", "_") # Sanitize
    default_save_path = st.session_state.get("model_save_path") or f"./{model_filename_default}_finetuned_simulated"
    st.session_state.model_save_path = st.text_input(
        "Enter desired save path for the model (simulated):", value=default_save_path, key="model_save_path_input"
    )
    if st.button("Save Model (Simulated)", key="save_model_simulated_btn"):
        save_path = st.session_state.model_save_path
        st.info(f"Simulating: Saving model to directory: {save_path}")
        try:
            os.makedirs(save_path, exist_ok=True)
            dummy_config = {
                "base_model": st.session_state.model_name, "dataset": st.session_state.dataset_name or st.session_state.custom_dataset_file_name,
                "trainer_used": st.session_state.trainer_config.get("selected_trainer"), "status": "Simulated Fine-Tune and Save",
                "trl_version_simulated_with": "0.19.0 (example)",
                "training_args_summary": {k: v for k,v in st.session_state.trainer_config.items() if k in ["num_train_epochs", "learning_rate", "per_device_train_batch_size"]}
            }
            config_file_name = "adapter_config.json" if st.session_state.trainer_config.get("use_peft") else "config.json"
            if st.session_state.trainer_config.get("use_peft"):
                 dummy_config["peft_config"] = {k:v for k,v in st.session_state.trainer_config.items() if k in ["lora_r", "lora_alpha", "lora_target_modules"]}
            with open(os.path.join(save_path, config_file_name), "w") as f: json.dump(dummy_config, f, indent=2)
            with open(os.path.join(save_path, "README_SIMULATED_SAVE.md"), "w") as f:
                f.write(f"# Simulated Model Save: {st.session_state.model_name} fine-tuned.\nThis is a dummy save.\nConfiguration:\n{json.dumps(dummy_config, indent=2)}")
            st.success(f"Model (simulated by creating dummy files) 'saved' in directory: '{save_path}'")
        except Exception as e: st.error(f"Error during simulated save (e.g., creating directory/files): {e}")

    st.subheader("Push to Hugging Face Hub (Simulated)")
    model_fn_for_repo = st.session_state.get("model_name", "model").split("/")[-1]
    trainer_name_short = st.session_state.trainer_config.get('selected_trainer','sft').replace("Trainer","").lower()
    default_hf_repo_name = st.session_state.get("hf_repo_name") or f"simulated-{model_fn_for_repo}-{trainer_name_short}"
    st.session_state.hf_repo_name = st.text_input(
        "Enter desired Hugging Face Hub repo name (e.g., your-username/repo-name):",
        value=default_hf_repo_name, key="hf_repo_name_input"
    )
    if st.button("Push to Hub (Simulated)", key="push_to_hub_simulated_btn"):
        if not st.session_state.get("hf_token"): st.warning("Hugging Face Token not provided in Step 1. Cannot simulate push without token for authentication.")
        else: st.success(f"Model (simulated) 'pushed' to Hub repo: {st.session_state.hf_repo_name}. (UI confirmation only)")

    st.markdown("---")
    if st.button("🔄 Start Over / Fine-Tune Another Model", key="restart_app_button"):
        app_mode_bkp = st.session_state.app_mode # Preserve current mode for sidebar
        st.session_state.clear() # Clear all session state
        init_session_state() # Re-initialize with all defaults
        st.session_state.app_mode = "Home" # Explicitly navigate to Home
        st.experimental_rerun()

if __name__ == "__main__":
    main()
