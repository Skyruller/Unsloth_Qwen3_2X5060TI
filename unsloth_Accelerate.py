#!/usr/bin/env python
import os
import sys  # For flushing output
import gc

# --- Critical Environment Variables (set BEFORE torch import) ---
os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# --- Early debug prints ---
print(f"[PID {os.getpid()}] Script start. Python version: {sys.version}", flush=True)
print(f"[PID {os.getpid()}] Current PWD: {os.getcwd()}", flush=True)
print(f"[PID {os.getpid()}] TORCH_DISTRIBUTED_USE_DTENSOR: {os.environ.get('TORCH_DISTRIBUTED_USE_DTENSOR')}", flush=True)
print(f"[PID {os.getpid()}] CUDA_VISIBLE_DEVICES (from env): {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"[PID {os.getpid()}] ACCELERATE_USE_TP: {os.environ.get('ACCELERATE_USE_TP')}", flush=True)

LAUNCHER_RANK = os.environ.get('RANK', 'N/A_LAUNCHER_RANK')
LAUNCHER_LOCAL_RANK = os.environ.get('LOCAL_RANK', 'N/A_LOCAL_RANK')
LAUNCHER_WORLD_SIZE = os.environ.get('WORLD_SIZE', 'N/A_WORLD_SIZE')
print(f"[PID {os.getpid()}] Launcher Env: RANK={LAUNCHER_RANK}, LOCAL_RANK={LAUNCHER_LOCAL_RANK}, WORLD_SIZE={LAUNCHER_WORLD_SIZE}", flush=True)

# --- Import torch and apply aggressive DTensor patch ---
import torch
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported torch. Version: {torch.__version__}. CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] CUDA device count: {torch.cuda.device_count()}", flush=True)
    try:
        if LAUNCHER_LOCAL_RANK != 'N/A_LOCAL_RANK':
            local_rank = int(LAUNCHER_LOCAL_RANK)
            # Убедимся, что мы используем правильное устройство
            torch.cuda.set_device(local_rank)
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Set CUDA device to: cuda:{local_rank}", flush=True)
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Name of current CUDA device: {torch.cuda.get_device_name(local_rank)}", flush=True)
        else:
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] LOCAL_RANK not set.", flush=True)
    except Exception as e_cuda_print:
        print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Error setting CUDA device: {e_cuda_print}", flush=True)

# AGGRESSIVE DTENSOR PATCH
try:
    from torch.distributed.tensor import DTensor
    if hasattr(DTensor, "_op_dispatcher") and \
       hasattr(DTensor._op_dispatcher, "sharding_propagator") and \
       hasattr(DTensor._op_dispatcher.sharding_propagator, "propagate"):
        original_propagate = DTensor._op_dispatcher.sharding_propagator.propagate
        def _no_op_propagate(self_sharding_prop, op_info, *args, **kwargs):
            return op_info.output_sharding
        DTensor._op_dispatcher.sharding_propagator.propagate = _no_op_propagate
        print(f"✅ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Successfully patched DTensor._op_dispatcher.sharding_propagator.propagate.", flush=True)
except ImportError:
    print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] torch.distributed.tensor.DTensor not found. Patch skipped.", flush=True)
except Exception as e:
    print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Error during DTensor patching: {e}", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing accelerate...", flush=True)
from accelerate import Accelerator
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported accelerate.", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing Unsloth...", flush=True)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported Unsloth.", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing Transformers & Datasets...", flush=True)
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported Transformers & Datasets.", flush=True)

# --- Configuration ---
MODEL_PATH = "/media/skyruller/Новый том/models/Qwen3-14B"
MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ", 1024))  # Use 2048 for stability
LORA_R = 8
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
LOCAL_JSONL = "/media/skyruller/Новый том/dataset/dataset_clean.jsonl"
TEST_SPLIT_RATIO = 0.1238
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MAX_STEPS = 300
LEARNING_RATE = 2e-4


# --- Model Loading ---
def load_model(current_accelerator):
    rank_idx = current_accelerator.process_index
    local_rank = current_accelerator.local_process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In load_model()...", flush=True)

    # 🔥 ВАЖНО: УКАЗЫВАЕМ device_map ЯВНО!
    device_map = {"": f"cuda:{local_rank}"}

    model_kwargs = {
        "model_name": MODEL_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": True,
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16,
        "device_map": device_map,  # <-- КЛЮЧЕВОЙ ФИКС!
    }

    print(f"[PID {pid}, Rank {rank_idx}] model_kwargs: {model_kwargs}", flush=True)
    print(f"[PID {pid}, Rank {rank_idx}] Calling FastLanguageModel.from_pretrained...", flush=True)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        print(f"[PID {pid}, Rank {rank_idx}] FastLanguageModel.from_pretrained successful.")
        print(f"[PID {pid}, Rank {rank_idx}] Model device after load: {model.device}")
    except Exception as e_load:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during FastLanguageModel.from_pretrained: {e_load}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    return model, tokenizer


# --- LoRA Application ---
def apply_lora(base_model, current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In apply_lora()...", flush=True)
    try:
        lora_model = FastLanguageModel.get_peft_model(
            base_model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        print(f"[PID {pid}, Rank {rank_idx}] apply_lora successful.", flush=True)
        return lora_model
    except Exception as e_lora:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during apply_lora: {e_lora}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise


# --- Dataset Handling ---
def load_and_split_dataset(current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In load_and_split_dataset()...", flush=True)
    try:
        ds = load_dataset("json", data_files={"train": LOCAL_JSONL}, trust_remote_code=True)["train"]
        splits = ds.train_test_split(test_size=TEST_SPLIT_RATIO, seed=42)
        print(f"[PID {pid}, Rank {rank_idx}] load_and_split_dataset successful.", flush=True)
        return splits["train"], splits["test"]
    except Exception as e_dsload:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during load_and_split_dataset: {e_dsload}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise


# --- Main Function ---
def main():
    pid = os.getpid()
    print(f"[PID {pid}, Pre-Accelerator-Rank] In main(). Initializing Accelerator...", flush=True)
    accelerator = Accelerator()
    rank_idx = accelerator.process_index
    print(f"[PID {pid}, Rank {rank_idx}] Accelerator initialized. Distributed: {accelerator.distributed_type}, Device: {accelerator.device}, Num_processes: {accelerator.num_processes}", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Loading model and tokenizer...", flush=True)
    model, tokenizer = load_model(accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] Model and tokenizer loaded.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Applying LoRA...", flush=True)
    model = apply_lora(model, accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] LoRA applied.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Loading and splitting dataset...", flush=True)
    train_ds_raw, val_ds_raw = load_and_split_dataset(accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] Dataset loaded and split.", flush=True)

    if tokenizer.pad_token is None:
        print(f"[PID {pid}, Rank {rank_idx}] Setting pad_token to eos_token.", flush=True)
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds_raw
    val_ds = val_ds_raw

    print(f"[PID {pid}, Rank {rank_idx}] Getting chat template...", flush=True)
    if tokenizer.pad_token is None:
        print(f"[PID {pid}, Rank {rank_idx}] Re-setting pad_token to eos_token post chat_template.", flush=True)
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[PID {pid}, Rank {rank_idx}] Chat template applied.", flush=True)

    DATASET_MAP_NUM_PROC = 1
    print(f"[PID {pid}, Rank {rank_idx}] Passing through train_ds text as-is (num_proc={DATASET_MAP_NUM_PROC})...", flush=True)
    train_ds = train_ds.map(lambda ex: {"text": ex["text"]}, num_proc=DATASET_MAP_NUM_PROC, remove_columns=[col for col in train_ds.features if col != 'text'])

    print(f"[PID {pid}, Rank {rank_idx}] Passing through val_ds text as-is (batched=True)...", flush=True)
    val_ds = val_ds.map(lambda batch: {"text": batch["text"]}, batched=True, num_proc=DATASET_MAP_NUM_PROC, remove_columns=[col for col in val_ds.features if col != 'text'])

    print(f"[PID {pid}, Rank {rank_idx}] Datasets processed.", flush=True)

    training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=15,              # <-- Количество эпох
    learning_rate=LEARNING_RATE,
    logging_steps=30,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    bf16=True,
    fp16=False,
)
    print(f"[PID {pid}, Rank {rank_idx}] TrainingArguments initialized.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Initializing SFTTrainer...", flush=True)
    SFT_DATASET_NUM_PROC = 1
    trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=SFT_DATASET_NUM_PROC,
    packing=False,
    args=training_args,
    use_fused_cross_entropy=False,  # <-- Ключевая строка!
)
    print(f"[PID {pid}, Rank {rank_idx}] SFTTrainer initialized. Model is on: {trainer.model.device}", flush=True)

    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Pre-train check for model.config.use_cache.", flush=True)
        unwrapped_model_for_config = accelerator.unwrap_model(trainer.model)
        if hasattr(unwrapped_model_for_config, "config") and getattr(unwrapped_model_for_config.config, "use_cache", False):
            print(f"✅ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Forcing model.config.use_cache = False on unwrapped model.", flush=True)
            unwrapped_model_for_config.config.use_cache = False

    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] All processes ready. Calling trainer.train()...", flush=True)

    try:
        metrics = trainer.train()
        print(f"[PID {pid}, Rank {rank_idx}] trainer.train() completed.", flush=True)
    except Exception as e_train:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during trainer.train(): {e_train}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] All processes finished training and synchronized.", flush=True)

    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Training finished. Saving artifacts...", flush=True)
        lora_adapter_path = "lora_adapters_final"
        merged_model_16bit_path = "merged_model_16bit"
        full_merged_model_path = "full_merged_model"

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving LoRA adapters to {lora_adapter_path}...", flush=True)
        try:
            trainer.model.save_pretrained(lora_adapter_path)
            tokenizer.save_pretrained(lora_adapter_path)
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: LoRA adapters saved.", flush=True)
        except Exception as e_lora_save:
            print(f"⚠️ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Error saving LoRA adapters: {e_lora_save}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Unwrapping model for save...", flush=True)
        unwrapped_model_for_save = accelerator.unwrap_model(trainer.model)

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Merging LoRA into base model...", flush=True)
        try:
            unwrapped_model_for_save.merge_and_unload()
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Merge successful.", flush=True)
        except Exception as e_merge:
            print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] MAIN PROCESS: ERROR during merge_and_unload: {e_merge}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving merged model to {merged_model_16bit_path}...", flush=True)
        try:
            unwrapped_model_for_save.save_pretrained_merged(merged_model_16bit_path, tokenizer, save_method="merged_16bit")
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Merged 16-bit model saved.", flush=True)
        except Exception as e_save_16bit:
            print(f"⚠️ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Error saving merged 16-bit model: {e_save_16bit}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving GGUF to {full_merged_model_path}...", flush=True)
        try:
            unwrapped_model_for_save.save_pretrained(full_merged_model_path)
            tokenizer.save_pretrained(full_merged_model_path)
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Model saved to GGUF.", flush=True)
        except Exception as e_gguf:
            print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] MAIN PROCESS: FATAL ERROR during GGUF save: {e_gguf}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            sys.exit(1)

        del trainer
        del model
        del unwrapped_model_for_save
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[PID {pid}, Rank {rank_idx}] Script finished successfully for this process.", flush=True)
    return metrics if 'metrics' in locals() else None


if __name__ == "__main__":
    main_pid = os.getpid()
    print(f"[PID {main_pid}] Script __main__ started.", flush=True)
    try:
        results = main()
        try:
            temp_accelerator_check = Accelerator()
            if temp_accelerator_check.is_main_process:
                print(f"[PID {main_pid}, MainRank] __main__: Training complete. Metrics: {results}", flush=True)
        except Exception as e_temp_accel:
            print(f"⚠️ [PID {main_pid}] Could not create temp accelerator for final print: {e_temp_accel}", flush=True)
    except Exception as e_main_fatal:
        print(f"🔥🔥🔥 [PID {main_pid}] FATAL ERROR in __main__ execution: {e_main_fatal}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
    print(f"[PID {main_pid}] Script __main__ exiting normally.", flush=True)
