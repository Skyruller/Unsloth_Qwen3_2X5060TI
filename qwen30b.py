#export CUDA_VISIBLE_DEVICES=1,0
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#python -u qwen30bDS.py \
#  --model "/home/skyruller/text-generation-webui/user_data/models/Qwen3-4B-Instruct-2507" \
#  --dataset "/media/skyruller/NovyTom/dataset/opensloth_powermill_dataset/dataset.jsonl" \
#  --max_seq 128 --lora_r 24 --ga 4 --lr 2e-4 \
#  --output_dir "outputs_30b" --merge_fp16 0 \
#  --max_memory "16GiB,16GiB"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, gc, json, re, types
from pathlib import Path
import torch

# бережно к памяти и без лишней компиляции
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

print(f"[PID {os.getpid()}] Python: {sys.version}", flush=True)
print(f"Torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}", flush=True)

from unsloth import FastLanguageModel
from transformers import TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback

# ----------------- аргументы -----------------
def parse_args():
    p = argparse.ArgumentParser("Qwen3-30B LoRA SFT (читает JSONL {'text': ...} и сырой ChatML)")
    p.add_argument("--model", required=True, help="путь/ID исходной модели (напр., Qwen3-30B-A3B-Instruct)")
    p.add_argument("--dataset", required=True, help="либо JSONL {'text': ...}, либо сырой ChatML-файл")
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--ga", type=int, default=8, help="gradient_accumulation_steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--output_dir", type=str, default="outputs_30b")
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--merge_fp16", type=int, default=0)  # зарезервирован
    p.add_argument(
        "--max_memory",
        type=str,
        default="16GiB,16GiB",
        help="по порядку CUDA_VISIBLE_DEVICES, напр. '16GiB,16GiB' или '14GiB,14GiB'",
    )
    return p.parse_args()

args = parse_args()

# ----------------- разбор max_memory -----------------
def _parse_max_memory(s: str):
    parts = [x.strip() for x in s.split(",") if x.strip()]
    out = {}
    for i in range(min(len(parts), torch.cuda.device_count())):
        val = parts[i]
        # поддержим формат '0:16GiB' → '16GiB'
        if ":" in val:
            _, val = val.split(":", 1)
            val = val.strip()
        out[i] = val  # accelerate сам преобразует '16GiB' → байты
    return out

# --------- FA2 детект ----------
def pick_attn_impl():
    # возвращаем корректное имя для transformers: "flash_attention_2" или "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"

# --------- quant: 4-bit bnb для веса, bf16 для вычислений ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

def load_model_and_tokenizer():
    max_memory = _parse_max_memory(args.max_memory)
    attn_impl = pick_attn_impl()
    print(f"Using device_map=auto, max_memory={max_memory}", flush=True)
    print(f"Attention impl = {attn_impl}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        offload_state_dict=True,
    )
    return model, tokenizer

# --- накрываем LoRA ---
def apply_lora(model):
    # dropout=0 — чтобы Unsloth патчил ВСЕ слои и не ругался на 0.05
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_rslora=True,
        loftq_config=None,
    )
    return model

# ---------- утилиты для чтения датасета ----------
_CHATML_BLOCK_SPLIT = re.compile(r'(?=^<\|im_start\|\>\s*system\s*$)', re.MULTILINE)
_CHATML_MSG = re.compile(r"<\|im_start\|\>\s*(system|user|assistant)\s*(.*?)<\|im_end\|\>", re.DOTALL | re.IGNORECASE)

def _looks_like_chatml_text(sample: str) -> bool:
    return "<|im_start|>" in sample and "<|im_end|>" in sample

def _try_read_json_line(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None

def _load_jsonl_text(ds_path: Path) -> Dataset:
    ds = load_dataset("json", data_files=str(ds_path), split="train", streaming=False)
    if "text" not in ds.column_names:
        candidate = None
        for c in ds.column_names:
            feat = ds.features.get(c)
            if str(feat).startswith("Value("):
                candidate = c
                break
        if candidate is None:
            raise ValueError("JSONL не содержит 'text' и не найдено строковых колонок.")
        ds = ds.rename_column(candidate, "text")
    return ds

def _load_raw_chatml(ds_path: Path) -> Dataset:
    raw = ds_path.read_text(encoding="utf-8", errors="replace")
    blocks = [b.strip() for b in _CHATML_BLOCK_SPLIT.split(raw) if b.strip()]
    records = []
    for b in blocks:
        parts = _CHATML_MSG.findall(b)
        if not parts:
            continue
        text = "".join(f"<|im_start|>{role}\n{content.strip()}\n<|im_end|>\n" for role, content in parts)
        records.append({"text": text})
    if not records:
        raise ValueError("Не удалось извлечь ни одного ChatML-блока из файла.")
    return Dataset.from_list(records)

def _smart_load_dataset(path_str: str) -> Dataset:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            first = s
            break
        else:
            raise ValueError("Файл датасета пуст.")
    parsed = _try_read_json_line(first)
    if parsed is not None:
        ds = _load_jsonl_text(p)
        print("Detected dataset format: JSONL {'text': ...} (или переименована первая строковая колонка → 'text')", flush=True)
        return ds
    if _looks_like_chatml_text(first) or _looks_like_chatml_text(p.read_text(encoding="utf-8", errors="replace")[:5000]):
        ds = _load_raw_chatml(p)
        print("Detected dataset format: RAW ChatML (преобразован в Dataset с колонкой 'text')", flush=True)
        return ds
    try:
        ds = _load_jsonl_text(p)
        print("Fallback: загружено как JSONL", flush=True)
        return ds
    except Exception as e:
        raise ValueError(f"Не удалось распознать формат датасета: {e}")

# --- загрузка и разбиение train/val ---
def load_text_dataset():
    ds = _smart_load_dataset(args.dataset)
    n = ds.num_rows
    n_val = max(50, int(n * 0.02))
    train_ds = ds.select(range(n - n_val)) if n_val < n else ds
    val_ds = ds.select(range(n - n_val, n)) if n_val < n else ds.select([])  # пустая валидация для маленьких наборов
    print(f"Dataset rows: {n} (train={len(train_ds)}, val={len(val_ds)})", flush=True)
    return train_ds, val_ds

# --- коллбек чистки VRAM на каждом шаге ---
class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args_t, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        return control

# --- оптимайзер: bnb 8-bit (при наличии), иначе AdamW ---
def make_optimizer(model, lr):
    params = (p for p in model.parameters() if p.requires_grad)
    try:
        from bitsandbytes.optim import PagedAdamW8bit
        print("Используем bitsandbytes.PagedAdamW8bit ✅", flush=True)
        return PagedAdamW8bit(params, lr=lr)
    except Exception as e:
        print(f"⚠️ BnB 8-bit недоступен ({e}). Переходим на torch.optim.AdamW.", flush=True)
        from torch.optim import AdamW
        return AdamW(params, lr=lr)

# --- совместимый конструктор TrainingArguments (эпохи решают) ---
def build_training_args():
    common_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.ga,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,        # ← управление длительностью через эпохи
        logging_steps=10,
        save_steps=50,          # сохраняем по твоему флагу
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
    )
    try:
        return TrainingArguments(
            evaluation_strategy="steps",
            logging_strategy="steps",
            eval_steps=max(50, args.save_steps),
            **common_kwargs,
        )
    except TypeError:
        print("⚠️ TrainingArguments без evaluation_strategy (совместимый режим).", flush=True)
        return TrainingArguments(
            do_eval=True,
            eval_steps=max(50, args.save_steps),
            **common_kwargs,
        )

def hook_create_optimizer(trainer):
    # Правильный хук: создаём и ПРИСВАИВАЕМ self.optimizer (иначе шедулер падает).
    def _create_optimizer(self):
        opt = make_optimizer(self.model, args.lr)
        self.optimizer = opt
        return opt
    trainer.create_optimizer = types.MethodType(_create_optimizer, trainer)

def main():
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train_ds, val_ds = load_text_dataset()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_args = build_training_args()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        packing=True,
    )

    trainer.add_callback(MemoryCleanupCallback())

    # фикс: корректно подменяем create_optimizer
    hook_create_optimizer(trainer)

    trainer.train()
    print(">>> Training done.", flush=True)

    # Сохранение LoRA-адаптеров
    lora_dir = os.path.join(args.output_dir, "lora_adapters")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"✓ LoRA adapters saved to: {lora_dir}", flush=True)

if __name__ == "__main__":
    main()
