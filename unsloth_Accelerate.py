#!/usr/bin/env python
import os, sys, gc
import subprocess, shutil


# --- Настройка окружения (до импорта torch) ---
os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

print(f"[PID {os.getpid()}] Script start. Python version: {sys.version}", flush=True)
print(f"[PID {os.getpid()}] Current PWD: {os.getcwd()}", flush=True)

LAUNCHER_RANK = os.environ.get('RANK', 'N/A')
LAUNCHER_LOCAL_RANK = os.environ.get('LOCAL_RANK', 'N/A')
LAUNCHER_WORLD_SIZE = os.environ.get('WORLD_SIZE', 'N/A')
print(f"Launcher Env: RANK={LAUNCHER_RANK}, LOCAL_RANK={LAUNCHER_LOCAL_RANK}, WORLD_SIZE={LAUNCHER_WORLD_SIZE}", flush=True)

import torch
print(f"Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}", flush=True)
    if LAUNCHER_LOCAL_RANK != 'N/A':
        local_rank = int(LAUNCHER_LOCAL_RANK)
        torch.cuda.set_device(local_rank)
        print(f"Set CUDA device to cuda:{local_rank} - {torch.cuda.get_device_name(local_rank)}", flush=True)

# Патч DTensor
try:
    from torch.distributed.tensor import DTensor
    if hasattr(DTensor, "_op_dispatcher"):
        original_propagate = DTensor._op_dispatcher.sharding_propagator.propagate
        def _no_op_propagate(self_sharding_prop, op_info, *args, **kwargs):
            return op_info.output_sharding
        DTensor._op_dispatcher.sharding_propagator.propagate = _no_op_propagate
        print("✅ Patched DTensor propagate", flush=True)
except Exception as e:
    print(f"⚠️ DTensor patch skipped: {e}", flush=True)

from accelerate import Accelerator
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# --- Конфигурация ---
MODEL_PATH = "/home/skyruller/webui/user_data/models/Qwen3-4B-Instruct-2507"
MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ", 1024))
LORA_R = 256
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
LORA_ALPHA = 512
LORA_DROPOUT = 0.0
LOCAL_JSONL = "/media/skyruller/Новый том/dataset/dataset_unescaped.jsonl"
TEST_SPLIT_RATIO = 0.1238
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
MAX_STEPS = 30
LEARNING_RATE = 2e-4

def load_model(accel):
    device_map = {"": f"cuda:{accel.local_process_index}"}
    model_kwargs = {
        "model_name": MODEL_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": True,
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16,
        "device_map": device_map,
    }
    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    return model, tokenizer

def apply_lora(model):
    return FastLanguageModel.get_peft_model(
        model,
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

def load_split_dataset():
    ds = load_dataset("json", data_files={"train": LOCAL_JSONL}, trust_remote_code=True)["train"]
    splits = ds.train_test_split(test_size=TEST_SPLIT_RATIO, seed=42)
    return splits["train"], splits["test"]
def post_training_menu(accelerator, trainer, tokenizer):
    if not accelerator.is_main_process:
        return

    # Пути по умолчанию
    LORA_DIR = "lora_adapters_final"
    MERGED_DIR = "merged_qwen3_4b_instruct_fp16"
    GGUF_F16 = "qwen3-4b-finetuned-f16.gguf"
    GGUF_Q4KM = "qwen3-4b-finetuned.q4_k_m.gguf"

    # Путь к llama.cpp (ожидаем папку рядом с этим скриптом)
    llama_cpp_dir = os.path.join(os.getcwd(), "llama.cpp")
    convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
    quant_bin = os.path.join(llama_cpp_dir, "quantize")

    def _save_adapters():
        print("→ Сохраняю LoRA-адаптеры…")
        trainer.model.save_pretrained(LORA_DIR)
        tokenizer.save_pretrained(LORA_DIR)
        print(f"✓ Адаптеры сохранены в: {LORA_DIR}")

    def _merge_fp16():
        print("→ Делаю корректный merge в FP16 (Unsloth)…")
        peft_model = accelerator.unwrap_model(trainer.model)
        peft_model.save_pretrained_merged(
            MERGED_DIR,
            tokenizer,
            save_method="merged_16bit",
            safe_serialization=True,
        )
        print(f"✓ Слитая FP16-модель сохранена в: {MERGED_DIR}")

    def _make_gguf():
        if not os.path.isdir(MERGED_DIR):
            print(f"✗ Папка {MERGED_DIR} не найдена. Сначала сделай merge (п.2).")
            return
        if not os.path.isfile(convert_script):
            print(f"✗ convert-hf-to-gguf.py не найден: {convert_script}")
            return

        print(f"→ HF→GGUF f16: {MERGED_DIR} → {GGUF_F16}")
        cmd = [sys.executable, convert_script, MERGED_DIR, "--outfile", GGUF_F16]
        subprocess.run(cmd, check=True)
        print(f"✓ GGUF f16 готов: {GGUF_F16}")

        # спросим про квант
        ans = input("Квантануть в q4_k_m? [y/N]: ").strip().lower()
        if ans == "y":
            if not os.path.isfile(quant_bin):
                print(f"✗ quantize не найден: {quant_bin}")
                return
            print(f"→ Квантование: {GGUF_F16} → {GGUF_Q4KM}")
            subprocess.run([quant_bin, GGUF_F16, GGUF_Q4KM, "q4_k_m"], check=True)
            print(f"✓ GGUF q4_k_m готов: {GGUF_Q4KM}")

    while True:
        print("\n=== Post-training menu ===")
        print("1) Сохранить адаптеры LoRA")
        print("2) Слить LoRA→FP16 (merged)")
        print("3) Сделать GGUF f16 (и опционально q4_k_m)")
        print("4) Сохранить и выйти (ничего больше не делать)")
        choice = input("Выбор [1/2/3/4]: ").strip()

        if choice == "1":
            _save_adapters()
        elif choice == "2":
            _save_adapters()  # полезно иметь адаптеры отдельно
            _merge_fp16()
        elif choice == "3":
            _make_gguf()
        elif choice == "4":
            # На всякий случай сохраним адаптеры, если ещё не сохраняли
            if not os.path.isdir(LORA_DIR):
                _save_adapters()
            print("Выход.")
            break
        else:
            print("Не понял выбор. Повтори.")


def main():
    accelerator = Accelerator()
    model, tokenizer = load_model(accelerator)
    model = apply_lora(model)
    train_ds, val_ds = load_split_dataset()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(lambda ex: {"text": ex["text"]}, remove_columns=[c for c in train_ds.features if c != 'text'])
    val_ds = val_ds.map(lambda ex: {"text": ex["text"]}, remove_columns=[c for c in val_ds.features if c != 'text'])

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=1,
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
        use_fused_cross_entropy=False,
    )

       # --- Обучение ---
    trainer.train()
    print("[main] Обучение завершено.")

    # --- Меню пост-обработки ---
    post_training_menu(accelerator, trainer, tokenizer)

if __name__ == "__main__":
    main()
