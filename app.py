import streamlit as st
from huggingface_hub import HfApi, RepositoryNotFoundError, GatedRepoError, create_repo, HfFolder
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DPOTrainer
from peft import LoraConfig, PeftModel
import torch
import os
import json
import time
import threading
import queue
import sys
from io import StringIO
import logging
import shutil

# --- Application Constants ---
APP_TITLE = "TRL Model Fine-Tuner (Local MPS/CPU) 🚀"

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "app_mode": "Home", "model_name": "gpt2", "dataset_name": "imdb", "dataset_path": None,
        "custom_dataset_file_name": None, "train_split": "train", "validation_split": "test",
        "dataset_load_success": False, "model_load_success": False, "available_splits": [],
        "hf_token": HfFolder.get_token(),
        "trainer_config": {
            "selected_trainer": "SFTTrainer", "output_dir": "./results_actual_training",
            "num_train_epochs": 1, "per_device_train_batch_size": 1, "learning_rate": 5e-5,
            "logging_steps": 10, "save_steps": 0, "evaluation_strategy": "no",
            "eval_steps": 500, "gradient_accumulation_steps": 1, "warmup_steps": 0,
            "max_grad_norm": 1.0, "lr_scheduler_type": "linear", "optim": "adamw_torch",
            "use_peft": True, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,v_proj",
            "sft_packing": False, "sft_max_seq_length": 512, "sft_dataset_text_field": "text",
            "dpo_beta": 0.1, "dpo_loss_type": "sigmoid"
        },
        "training_started": False, "training_complete": False,
        "training_logs": "Training logs will appear here once training starts.",
        "cancel_event": None, "log_queue": None, "training_thread": None,
        "training_progress": 0.0, "navigated_away_after_training": False,
        "model_save_path": "",  "hf_repo_name": "",
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") and torch.__version__ >= "1.12" else False,
        "cpu_fallback_mode": False, "loaded_tokenizer_for_saving": None,
        "loaded_model_for_saving": None, "training_output_dir": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, dict): st.session_state[key] = value.copy()
            else: st.session_state[key] = value

# --- Helper Functions ---
def check_model_exists(model_name, token=None):
    if not model_name.strip(): return False, "Model name/path cannot be empty."
    try: AutoConfig.from_pretrained(model_name, token=token); return True, f"✅ Model '{model_name}' config accessible."
    except RepositoryNotFoundError: return False, f"❌ Model '{model_name}' not found on Hub."
    except GatedRepoError: return False, f"❌ Model '{model_name}' gated. Provide HF token or log in."
    except OSError: return False, f"❌ Model '{model_name}' not found locally or invalid path."
    except Exception as e: return False, f"❌ Error checking model '{model_name}': {str(e)}"

def check_dataset_exists(dataset_name, token=None):
    if not dataset_name.strip(): return False, "Dataset name cannot be empty."
    try: load_dataset_builder(dataset_name, token=token); return True, f"✅ Dataset '{dataset_name}' accessible on Hub."
    except Exception as e: return False, f"❌ Error checking dataset '{dataset_name}': {str(e)}"

def get_dataset_splits_info(dataset_name_or_path, is_custom_file=False, token=None):
    if is_custom_file: return ["train"], "For uploaded files, 'train' is assumed."
    if not dataset_name_or_path.strip(): return [], "Dataset name empty."
    try:
        split_names = get_dataset_split_names(dataset_name_or_path, token=token)
        if not split_names: return [], "No standard splits found."
        return split_names, f"Available splits: {split_names}"
    except Exception as e: return [], f"Could not retrieve splits for '{dataset_name_or_path}': {str(e)}"

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(self.format(record))

def main():
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    init_session_state()
    st.title(APP_TITLE)
    st.sidebar.title("Navigation")
    app_mode_options = ["Home", "1. Configure Model & Dataset", "2. Configure Trainer & Parameters", "3. Start Training", "4. View Results & Save"]
    if st.session_state.app_mode not in app_mode_options: st.session_state.app_mode = "Home"
    current_selection_index = app_mode_options.index(st.session_state.app_mode)
    st.session_state.app_mode = st.sidebar.selectbox("Choose the step:", app_mode_options, index=current_selection_index, key="sidebar_nav")
    if st.session_state.app_mode == "Home": render_home_page()
    elif st.session_state.app_mode == "1. Configure Model & Dataset": render_config_model_dataset_page()
    elif st.session_state.app_mode == "2. Configure Trainer & Parameters": render_config_trainer_parameters_page()
    elif st.session_state.app_mode == "3. Start Training": render_training_page()
    elif st.session_state.app_mode == "4. View Results & Save": render_results_page()

def render_home_page():
    st.header("Welcome!")
    st.markdown("Fine-tune Hugging Face models using TRL on your **local Apple Silicon (MPS) or CPU.**")
    if st.session_state.mps_available: st.success("✅ MPS (Apple Silicon GPU) backend available and preferred.")
    else: st.warning("⚠️ MPS not available. Training will use CPU (may be slow).")
    st.markdown("### Workflow Overview:")
    st.markdown("1. **Configure Model & Dataset**\n2. **Configure Trainer & Parameters** (LoRA supported)\n3. **Start Training** (Local execution)\n4. **View Results & Save** (Locally or to Hugging Face Hub)")
    st.info("💡 Start with small models/datasets and few epochs for initial tests.")

def render_config_model_dataset_page():
    st.header("Step 1: Configure Model & Dataset")
    with st.expander("Hugging Face Token (Optional)", expanded=False):
        st.session_state.hf_token = st.text_input("HF Token:", type="password", value=st.session_state.hf_token or "", key="hf_token_input_s1")
    token_to_use = st.session_state.hf_token if st.session_state.hf_token.strip() else None
    st.subheader("Model Selection")
    st.session_state.model_name = st.text_input("Model Name/Path:", value=st.session_state.model_name, key="model_name_input_s1")
    if st.button("Check Model", key="check_model_btn_s1"):
        st.session_state.model_load_success, msg = check_model_exists(st.session_state.model_name, token=token_to_use)
        if st.session_state.model_load_success: st.success(msg)
        else: st.error(msg)
    st.subheader("Dataset Selection")
    ds_options = ("Hugging Face Hub", "Upload Custom File"); ds_idx = ds_options.index(st.session_state.get("dataset_source_radio_s1_val", "Hugging Face Hub"))
    ds_source = st.radio("Source:", ds_options, key="dataset_source_radio_s1", index=ds_idx, horizontal=True); st.session_state["dataset_source_radio_s1_val"] = ds_source
    if ds_source == "Hugging Face Hub":
        if st.session_state.custom_dataset_file_name: st.session_state.custom_dataset_file_name = None; st.session_state.dataset_path = None; st.session_state.dataset_load_success = False
        st.session_state.dataset_name = st.text_input("Dataset Name (Hub):", value=st.session_state.dataset_name, key="dataset_name_input_s1")
        if st.button("Check Dataset & Splits", key="check_dataset_btn_s1"):
            st.session_state.dataset_load_success, msg = check_dataset_exists(st.session_state.dataset_name, token=token_to_use)
            if st.session_state.dataset_load_success: st.success(msg); splits, s_msg = get_dataset_splits_info(st.session_state.dataset_name, token=token_to_use); st.session_state.available_splits = splits; st.info(s_msg)
            else: st.error(msg); st.session_state.available_splits = []
    else:
        if st.session_state.dataset_name: st.session_state.dataset_name = ""; st.session_state.available_splits = []; st.session_state.dataset_load_success = False
        up_file = st.file_uploader("Upload (JSONL, CSV, TXT):", type=['jsonl', 'csv', 'json', 'txt'], key="custom_file_uploader_s1")
        if up_file:
            st.session_state.dataset_path = up_file; st.session_state.custom_dataset_file_name = up_file.name; st.success(f"Uploaded '{up_file.name}'."); st.session_state.dataset_load_success = True
            splits, s_msg = get_dataset_splits_info(None, is_custom_file=True); st.session_state.available_splits = splits; st.info(s_msg)
            if "train" in splits: st.session_state.train_split = "train"; st.session_state.validation_split = None
        elif st.session_state.custom_dataset_file_name: st.session_state.custom_dataset_file_name = None; st.session_state.dataset_path = None; st.session_state.dataset_load_success = False
    with st.expander("Dataset Format Guidance", expanded=False): st.markdown("SFT: `text` or `messages`. DPO: `prompt`, `chosen`, `rejected`.")
    c1,c2=st.columns(2); av_sp = st.session_state.available_splits
    with c1: st.session_state.train_split=st.selectbox("Train Split:",av_sp,index=av_sp.index(st.session_state.train_split) if st.session_state.train_split in av_sp else 0,key="ts_s1") if av_sp else st.text_input("Train Split:",value=st.session_state.train_split,key="tsi_s1")
    with c2: st.session_state.validation_split=st.selectbox("Eval Split:",["None"]+av_sp,index=(["None"]+av_sp).index(st.session_state.validation_split if st.session_state.validation_split else "None"),key="vs_s1") if av_sp else st.text_input("Eval Split:",value=st.session_state.validation_split or "",key="vsi_s1")
    if st.session_state.validation_split=="None":st.session_state.validation_split=None
    if not (st.session_state.model_load_success and st.session_state.dataset_load_success and st.session_state.train_split): st.error("Config incomplete for Step 1.")

def render_config_trainer_parameters_page():
    st.header("Step 2: Configure Trainer & Parameters")
    if not st.session_state.get("model_load_success") or not st.session_state.get("dataset_load_success"): st.warning("Complete Step 1 first."); return
    tc = st.session_state.trainer_config; st.subheader("Trainer & Device")
    c1,c2=st.columns(2)
    with c1:tc["selected_trainer"]=st.selectbox("TRL Trainer:",["SFTTrainer","DPOTrainer"],index=["SFTTrainer","DPOTrainer"].index(tc["selected_trainer"]),key="tsel_s2")
    with c2:
        dev_opts=["mps","cpu"] if st.session_state.mps_available else ["cpu"]; dev_idx=0 if not st.session_state.cpu_fallback_mode and st.session_state.mps_available else dev_opts.index("cpu")
        chosen_dev=st.selectbox("Device:",dev_opts,index=dev_idx,key="devsel_s2",help="MPS for Apple GPU, CPU otherwise."); st.session_state.cpu_fallback_mode = chosen_dev == "cpu"
        if chosen_dev=="cpu" and st.session_state.mps_available: st.caption("CPU selected (MPS available).")
    st.subheader("Training Arguments"); c1,c2=st.columns(2)
    with c1: tc["output_dir"]=st.text_input("Output Dir:",value=tc["output_dir"],key="od_s2"); tc["num_train_epochs"]=st.number_input("Epochs:",min_value=1,step=1,value=tc["num_train_epochs"],key="ep_s2")
    with c2: tc["per_device_train_batch_size"]=st.number_input("Batch Size:",min_value=1,step=1,value=tc["per_device_train_batch_size"],key="bs_s2"); tc["learning_rate"]=st.number_input("Learning Rate:",value=tc["learning_rate"],format="%e",key="lr_s2")
    with st.expander("Advanced Args"):
        c1,c2=st.columns(2)
        with c1: tc["logging_steps"]=st.number_input("Log Steps:",min_value=1,step=10,value=tc["logging_steps"],key="ls_s2"); tc["gradient_accumulation_steps"]=st.number_input("Grad. Accum.:",min_value=1,step=1,value=tc["gradient_accumulation_steps"],key="ga_s2"); tc["optim"]=st.selectbox("Optimizer:",["adamw_torch","adamw_hf","adafactor"],index=["adamw_torch","adamw_hf","adafactor"].index(tc["optim"]),key="opt_s2")
        with c2: tc["save_steps"]=st.number_input("Save Steps (0=none):",min_value=0,step=100,value=tc["save_steps"],key="ss_s2"); tc["warmup_steps"]=st.number_input("Warmup Steps:",min_value=0,step=50,value=tc["warmup_steps"],key="ws_s2"); tc["max_grad_norm"]=st.number_input("Max Grad Norm:",min_value=0.0,value=tc["max_grad_norm"],format="%.1f",key="mgn_s2"); tc["lr_scheduler_type"]=st.selectbox("LR Scheduler:",["linear","cosine","constant"],index=["linear","cosine","constant"].index(tc["lr_scheduler_type"]),key="lrs_s2")
        tc["evaluation_strategy"]=st.selectbox("Eval Strategy:",["no","steps","epoch"],index=["no","steps","epoch"].index(tc["evaluation_strategy"]),key="es_s2")
        if tc["evaluation_strategy"]=="steps": tc["eval_steps"]=st.number_input("Eval Steps:",min_value=1,step=50,value=tc["eval_steps"],key="evs_s2")
    st.subheader("PEFT (LoRA) Configuration"); tc["use_peft"]=st.checkbox("Enable LoRA",value=tc["use_peft"],key="upl_s2")
    if tc["use_peft"]:
        c1,c2=st.columns(2)
        with c1: tc["lora_r"]=st.number_input("LoRA r:",min_value=1,step=1,value=tc["lora_r"],key="lr_r_s2"); tc["lora_alpha"]=st.number_input("LoRA alpha:",min_value=1,step=1,value=tc["lora_alpha"],key="lr_a_s2")
        with c2: tc["lora_dropout"]=st.slider("LoRA Dropout:",0.0,1.0,tc["lora_dropout"],0.01,key="lr_d_s2"); tc["lora_target_modules"]=st.text_input("Target Modules (comma-sep):",value=tc["lora_target_modules"],key="lr_tm_s2",help="E.g. q_proj,v_proj for Llama. Check model docs.")
    st.caption("QLoRA (4-bit) is not available for MPS.")
    st.subheader(f"{tc['selected_trainer']} Specifics")
    if tc["selected_trainer"]=="SFTTrainer": tc["sft_packing"]=st.checkbox("Enable Packing",value=tc["sft_packing"],key="sft_p_s2"); tc["sft_max_seq_length"]=st.number_input("Max Seq Len:",min_value=32,step=32,value=tc["sft_max_seq_length"],key="sft_msl_s2"); tc["sft_dataset_text_field"]=st.text_input("Dataset Text Field:",value=tc["sft_dataset_text_field"],key="sft_dtf_s2")
    elif tc["selected_trainer"]=="DPOTrainer": tc["dpo_beta"]=st.slider("DPO Beta:",0.0,1.0,tc["dpo_beta"],0.01,key="dpo_b_s2"); tc["dpo_loss_type"]=st.selectbox("DPO Loss:",["sigmoid","hinge","ipo","kpo_pair"],index=["sigmoid","hinge","ipo","kpo_pair"].index(tc["dpo_loss_type"]),key="dpo_l_s2"); st.info("DPO needs: prompt, chosen, rejected cols.")
    st.success("Trainer config updated.")

def run_actual_training(log_queue, cancel_event, training_args_dict, model_args, dataset_args, trainer_specific_args):
    thread_logger = logging.getLogger("training_thread"); thread_logger.handlers = [QueueLogHandler(log_queue)]; thread_logger.setLevel(logging.INFO)
    try:
        thread_logger.info("Training thread started."); device = torch.device(model_args.get('device', 'cpu')); thread_logger.info(f"Using device: {device}")
        thread_logger.info(f"Loading tokenizer: {model_args['model_name_or_path']}")
        tokenizer = AutoTokenizer.from_pretrained(model_args['model_name_or_path'], token=st.session_state.hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; thread_logger.info("Set pad_token to eos_token.")
        st.session_state.loaded_tokenizer_for_saving = tokenizer
        thread_logger.info(f"Loading base model: {model_args['model_name_or_path']}")
        model = AutoModelForCausalLM.from_pretrained(model_args['model_name_or_path'], token=st.session_state.hf_token, trust_remote_code=True).to(device)
        thread_logger.info("Base model loaded.")
        peft_config = None
        if model_args.get("use_peft"):
            target_modules_list = [s.strip() for s in model_args.get("lora_target_modules", "").split(',') if s.strip()]
            if target_modules_list: peft_config = LoraConfig(r=model_args.get("lora_r",8), lora_alpha=model_args.get("lora_alpha",16), lora_dropout=model_args.get("lora_dropout",0.05), target_modules=target_modules_list, bias="none", task_type="CAUSAL_LM"); thread_logger.info(f"PEFT LoRA configured for: {target_modules_list}")
            else: thread_logger.warning("LoRA target modules empty. PEFT may not be effective.")
        thread_logger.info(f"Loading dataset: {dataset_args['dataset_name_or_path']}")
        if dataset_args.get("is_custom_file"):
            up_file = dataset_args['dataset_name_or_path']; file_content = up_file.getvalue().decode("utf-8")
            if up_file.name.endswith(".jsonl"): data = [json.loads(line) for line in file_content.splitlines()]
            elif up_file.name.endswith(".csv"): import csv; reader = csv.DictReader(StringIO(file_content)); data = list(reader)
            else: data = [{"text": line} for line in file_content.splitlines()]
            raw_dataset = Dataset.from_list(data); thread_logger.info(f"Loaded custom dataset: {up_file.name}")
        else: raw_dataset = load_dataset(dataset_args['dataset_name_or_path'], split=dataset_args['train_split'], token=st.session_state.hf_token)
        eval_dataset = None
        if dataset_args.get('validation_split'):
            try: eval_dataset = load_dataset(dataset_args['dataset_name_or_path'], split=dataset_args['validation_split'], token=st.session_state.hf_token) if not dataset_args.get("is_custom_file") else None
            except Exception as e: thread_logger.warning(f"Could not load eval_dataset: {e}")
        thread_logger.info("Initializing TrainingArguments."); training_args = TrainingArguments(**training_args_dict); st.session_state.training_output_dir = training_args.output_dir
        thread_logger.info(f"Initializing {trainer_specific_args['trainer_class_name']}.")
        common_args = {"model":model, "args":training_args, "train_dataset":raw_dataset, "eval_dataset":eval_dataset, "tokenizer":tokenizer, "peft_config":peft_config}
        if trainer_specific_args['trainer_class_name'] == "SFTTrainer": trainer = SFTTrainer(**common_args, dataset_text_field=trainer_specific_args.get("sft_dataset_text_field","text"), max_seq_length=trainer_specific_args.get("sft_max_seq_length",512), packing=trainer_specific_args.get("sft_packing",False))
        elif trainer_specific_args['trainer_class_name'] == "DPOTrainer": trainer = DPOTrainer(**common_args, beta=trainer_specific_args.get("dpo_beta",0.1), loss_type=trainer_specific_args.get("dpo_loss_type","sigmoid"))
        else: thread_logger.error(f"Trainer {trainer_specific_args['trainer_class_name']} not implemented."); return
        thread_logger.info("Starting trainer.train()...");
        if cancel_event.is_set(): thread_logger.info("Cancelled before train start."); return
        trainer.train(); thread_logger.info("trainer.train() finished.")
        st.session_state.loaded_model_for_saving = model
        # Save final model if not saved by steps
        if training_args.save_strategy == "no" or training_args.save_steps == 0: # Check if model was saved by steps
            final_save_path = training_args.output_dir
            os.makedirs(final_save_path, exist_ok=True)
            if hasattr(model, "save_pretrained"): model.save_pretrained(final_save_path) # Saves adapter if PEFT, full if not
            else: torch.save(model.state_dict(), os.path.join(final_save_path, "pytorch_model.bin")) # Fallback
            tokenizer.save_pretrained(final_save_path)
            thread_logger.info(f"Final model and tokenizer explicitly saved to {final_save_path}.")

    except Exception as e: thread_logger.error(f"ERROR in training thread: {str(e)}"); import traceback; thread_logger.error(traceback.format_exc())
    finally: log_queue.put("TRAINING_THREAD_DONE")

def render_training_page():
    st.header("Step 3: Start Training")
    if not st.session_state.get("model_load_success") or not st.session_state.get("dataset_load_success") or "trainer_config" not in st.session_state: st.warning("Complete Steps 1 & 2 first."); return
    log_placeholder = st.empty(); progress_placeholder = st.empty()
    if st.session_state.log_queue is None: st.session_state.log_queue = queue.Queue()
    if st.session_state.cancel_event is None: st.session_state.cancel_event = threading.Event()
    c1,c2=st.columns(2)
    with c1:
        start_dis = st.session_state.training_started and not st.session_state.training_complete
        if st.button("▶️ Start Training",disabled=start_dis,key="start_train_btn_s3"):
            st.session_state.training_started=True; st.session_state.training_complete=False; st.session_state.cancel_event.clear(); st.session_state.training_logs="INFO: Preparing training...\n"; st.session_state.navigated_away_after_training=False
            tc=st.session_state.trainer_config; valid_ta_keys = TrainingArguments.__dataclass_fields__.keys()
            train_args_dict = {key: tc[key] for key in tc if key in valid_ta_keys and key in tc}
            train_args_dict.update({
                "output_dir": tc["output_dir"], "num_train_epochs": tc["num_train_epochs"], "per_device_train_batch_size": tc["per_device_train_batch_size"],
                "learning_rate": tc["learning_rate"], "logging_steps": tc["logging_steps"],
                "save_strategy": "steps" if tc.get("save_steps", 0) > 0 else "no",
                "save_steps": tc.get("save_steps", 500) if tc.get("save_steps", 0) > 0 else 500, # Default if strategy is steps
                "evaluation_strategy": tc.get("evaluation_strategy", "no") if tc.get("evaluation_strategy", "no") != "no" else "no",
                "eval_steps": tc.get("eval_steps") if tc.get("evaluation_strategy", "no") == "steps" else None,
                "gradient_accumulation_steps": tc["gradient_accumulation_steps"], "warmup_steps": tc["warmup_steps"],
                "max_grad_norm": tc["max_grad_norm"], "lr_scheduler_type": tc["lr_scheduler_type"], "optim": tc["optim"],
                "report_to": "none", "remove_unused_columns": True,
                "load_best_model_at_end": True if tc.get("evaluation_strategy", "no") != "no" and tc.get("save_steps", 0) > 0 else False,
                "save_total_limit": 1 if tc.get("save_steps", 0) > 0 else None,
            })
            if train_args_dict.get("evaluation_strategy") == "no": train_args_dict.pop("eval_steps", None)
            if train_args_dict.get("save_strategy") == "no": train_args_dict.pop("save_steps", None); train_args_dict.pop("save_total_limit", None); train_args_dict.pop("load_best_model_at_end", None)

            model_args = {"model_name_or_path":st.session_state.model_name, "use_peft":tc["use_peft"], **{k:v for k,v in tc.items() if "lora_" in k}, "device":"mps" if not st.session_state.cpu_fallback_mode and st.session_state.mps_available else "cpu"}
            dataset_args = {"dataset_name_or_path":st.session_state.dataset_path if st.session_state.dataset_path else st.session_state.dataset_name, "is_custom_file":st.session_state.dataset_path is not None, "train_split":st.session_state.train_split, "validation_split":st.session_state.validation_split}
            trainer_specific_args = {"trainer_class_name":tc["selected_trainer"], **{k:v for k,v in tc.items() if "sft_" in k or "dpo_" in k}}
            st.session_state.training_thread = threading.Thread(target=run_actual_training, args=(st.session_state.log_queue, st.session_state.cancel_event, train_args_dict, model_args, dataset_args, trainer_specific_args), daemon=True)
            st.session_state.training_thread.start(); st.info(f"INFO: Training started on device: {model_args['device']}. Logs below.")
    with c2:
        cancel_dis = not st.session_state.training_started or st.session_state.training_complete
        if st.button("⏹️ Cancel Training",disabled=cancel_dis,key="cancel_train_btn_s3"):
            if st.session_state.training_thread and st.session_state.training_thread.is_alive(): st.session_state.cancel_event.set(); st.warning("INFO: Cancellation requested...")
            else: st.info("INFO: No active training to cancel.")
    if st.session_state.training_started and not st.session_state.training_complete:
        with st.spinner("Training in progress..."):
            while True:
                try: msg = st.session_state.log_queue.get_nowait()
                except queue.Empty: time.sleep(0.2); msg = None
                if msg == "TRAINING_THREAD_DONE": st.session_state.training_complete=True; st.session_state.training_started=False; st.experimental_rerun(); break
                if msg: st.session_state.training_logs += msg
                if st.session_state.training_thread and not st.session_state.training_thread.is_alive() and not st.session_state.training_complete: st.session_state.training_complete=True; st.session_state.training_started=False; st.session_state.training_logs += "\nERROR: Training thread terminated unexpectedly.\n"; st.experimental_rerun(); break
                log_placeholder.text_area("Logs:",value="\n".join(st.session_state.training_logs.splitlines()[-200:]),height=400,key="logdisp_s3_active")
                if not st.session_state.training_started or st.session_state.training_complete: break
    else: log_placeholder.text_area("Logs:",value="\n".join(st.session_state.training_logs.splitlines()[-200:]),height=400,key="logdisp_s3_final")
    if st.session_state.training_complete:
        if st.session_state.cancel_event.is_set(): st.warning("Training was cancelled.")
        elif "error" in st.session_state.training_logs.lower(): st.error("Training FAILED. Check logs.")
        else: st.success("Training FINISHED.")
        if st.button("Proceed to Step 4",key="train_to_res_btn_s3"): st.session_state.app_mode="4. View Results & Save"; st.session_state.navigated_away_after_training=True; st.experimental_rerun()

def render_results_page():
    st.header("Step 4: View Results & Save Model")
    if not st.session_state.get("training_complete", False): st.warning("Training not completed. Go to Step 3."); return
    st.subheader("Training Summary"); final_log_lines = [line for line in st.session_state.training_logs.splitlines() if "epoch" in line.lower() or "step" in line.lower() or "loss" in line.lower()]
    st.text_area("Key Training Log Snippets:", value="\n".join(final_log_lines[-10:]), height=150, disabled=True, key="log_snippet_s4")
    st.subheader("Save Fine-Tuned Model Locally")
    default_save_path = st.session_state.get("training_output_dir", f"./{st.session_state.model_name.replace('/','_')}_finetuned")
    st.session_state.model_save_path = st.text_input("Local Save Path:", value=default_save_path, key="save_path_input_s4")
    if st.button("Save Model Locally", key="save_local_btn_s4"):
        save_path = st.session_state.model_save_path; output_dir_from_training = st.session_state.get("training_output_dir")
        if not output_dir_from_training or not os.path.exists(output_dir_from_training): st.error(f"Training output dir '{output_dir_from_training}' not found. Model should have been saved by trainer.")
        else:
            try:
                if save_path != output_dir_from_training:
                    if os.path.exists(save_path): shutil.rmtree(save_path)
                    shutil.copytree(output_dir_from_training, save_path); st.success(f"Model files copied to '{save_path}'.")
                else: st.success(f"Model already in '{save_path}' (trainer's output_dir).")
                if st.session_state.get("loaded_tokenizer_for_saving") and not os.path.exists(os.path.join(save_path, "tokenizer_config.json")): st.session_state.loaded_tokenizer_for_saving.save_pretrained(save_path); st.info(f"Tokenizer explicitly saved to '{save_path}'.")
            except Exception as e: st.error(f"Error saving/copying model: {e}")
    st.subheader("Push to Hugging Face Hub")
    default_repo_name = f"{st.session_state.model_name.split('/')[-1]}-{st.session_state.trainer_config['selected_trainer'].lower()}-ft"
    st.session_state.hf_repo_name = st.text_input("Hub Repo Name (YourUsername/MyModel):", value=st.session_state.get("hf_repo_name") or default_repo_name, key="hf_repo_input_s4")
    if st.button("Push to Hub", key="push_hub_btn_s4"):
        if not st.session_state.get("hf_token"): st.warning("HF Token not provided (Step 1).");
        elif not st.session_state.get("training_output_dir") or not os.path.exists(st.session_state.get("training_output_dir")): st.warning("Model output dir not found. Save/train model first.")
        else:
            try:
                repo_id = st.session_state.hf_repo_name; st.info(f"Creating repo '{repo_id}' and uploading files...")
                api = HfApi(token=st.session_state.hf_token); create_repo(repo_id, token=st.session_state.hf_token, exist_ok=True, private=True)
                api.upload_folder(folder_path=st.session_state.training_output_dir, repo_id=repo_id, token=st.session_state.hf_token, repo_type="model")
                st.success(f"Model pushed to Hub: [{repo_id}](https://huggingface.co/{repo_id})")
            except Exception as e: st.error(f"Error pushing to Hub: {e}")
    st.markdown("---")
    if st.button("🔄 Start Over", key="restart_app_btn_s4"): st.session_state.clear(); init_session_state(); st.session_state.app_mode = "Home"; st.experimental_rerun()

if __name__ == "__main__":
    main()
