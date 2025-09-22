#export CUDA_VISIBLE_DEVICES=1,0
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#python -u qwen30b_1proc_sharded_lora.py \
#  --model "/home/skyruller/webui/user_data/models/unsloth_Qwen3-30B-A3B-Instruct-2507" \
#  --dataset "/media/skyruller/Novy/dataset/dataset_unescaped.jsonl" \
#  --max_seq 256 \
#  --lora_r 16 \
#  --ga 128 \
#  --lr 2e-4 \
#  --epochs 25 \
#  --targets 7 \
#  --attn sdpa \
#  --max_memory "16GiB,16GiB" \
#  --output_dir "outputs_30b" \
#  --merge_fp16 0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse
import torch

# ── безопасные дефолты окружения (не мешают, если уже выставлено снаружи) ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

print(f"[PID {os.getpid()}] Python: {sys.version}", flush=True)
print(f"[PID {os.getpid()}] PWD: {os.getcwd()}", flush=True)
print(f"Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA devices visible: {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}", flush=True)

# ── импорты обучения ──
from unsloth import FastLanguageModel
from transformers import TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer

# ── аргументы CLI ──
def parse_args():
    p = argparse.ArgumentParser("Qwen3-30B 1-proc sharded LoRA SFT")
    p.add_argument("--model", type=str, required=True, help="Путь к HF модели (директория)")
    p.add_argument("--dataset", type=str, required=True, help="JSONL с полем 'text'")
    p.add_argument("--max_seq", type=int, default=512, help="Макс. длина контекста")
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    p.add_argument("--ga", type=int, default=8, help="gradient_accumulation_steps")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--epochs", type=int, default=5, help="количество эпох")
    p.add_argument("--output_dir", type=str, default="outputs_30b", help="куда писать")
    p.add_argument("--save_steps", type=int, default=200, help="шаги сохранения")
    p.add_argument("--merge_fp16", type=int, default=0, help="1 = сделать merged FP16")
    p.add_argument("--targets", type=int, choices=[4,7], default=7,
                   help="4 = [q,k,v,o], 7 = [q,k,v,o,gate,up,down]")
    p.add_argument("--attn", type=str, choices=["sdpa","flash_attention_2"],
                   default="sdpa", help="механизм вним.")
    p.add_argument("--max_memory", type=str, default="13GiB,14GiB",
                   help='Лимиты через запятую для GPU0,GPU1, напр. "13GiB,14GiB"')
    return p.parse_args()

args = parse_args()

def parse_max_memory(s: str):
    parts = [x.strip() for x in s.split(",") if x.strip()]
    d = {}
    for i, v in enumerate(parts):
        d[i] = v
    if not d:  # подстраховка
        d = {0: "16GiB", 1: "16GiB"}
    print(f"Using max_memory map: {d}", flush=True)
    return d

def target_modules_by_choice(n: int):
    if n == 4:
        mods = ["q_proj","k_proj","v_proj","o_proj"]
    else:
        mods = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]
    print(f"Target modules: {mods}", flush=True)
    return mods

# ── дабл-квант 4бит (bnb) ──
def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # << ДВОЙНАЯ КВАНТОВКА
    )

# ── загрузка модели ──
def load_model_and_tokenizer():
    quant_config = build_quant_config()
    mm = parse_max_memory(args.max_memory)

    def _try_load(attn_impl: str):
        return FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=mm,
            quantization_config=quant_config,
            attn_implementation=attn_impl,
        )

    # сначала пытаемся то, что попросил пользователь
    try:
        model, tokenizer = _try_load(args.attn)
        return model, tokenizer
    except Exception as e:
        if args.attn == "flash_attention_2":
            print(f"⚠️ Flash-Attention 2 недоступен/упал: {e}\n→ Переключаюсь на SDPA.", flush=True)
            model, tokenizer = _try_load("sdpa")
            return model, tokenizer
        else:
            raise

# ── навешиваем LoRA ──
def apply_lora(model):
    target = target_modules_by_choice(args.targets)
    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target,
        lora_alpha=max(2*args.lora_r, 16),
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

# ── датасет ──
def load_text_dataset():
    ds = load_dataset("json", data_files={"train": args.dataset}, trust_remote_code=True)["train"]
    # небольшой валидационный сплит (если прям не надо — можно убрать)
    try:
        splits = ds.train_test_split(test_size=0.02, seed=42)
        train_ds, val_ds = splits["train"], splits["test"]
    except Exception:
        train_ds, val_ds = ds, None

    def _pick_text(ex):
        if "text" in ex:
            return {"text": ex["text"]}
        # fallback — берём первое строковое поле
        for k, v in ex.items():
            if isinstance(v, str):
                return {"text": v}
        raise ValueError("Пример без текстового поля. Нужен ключ 'text'!")

    train_ds = train_ds.map(_pick_text, remove_columns=[c for c in train_ds.features if c != "text"])
    if val_ds is not None:
        val_ds = val_ds.map(_pick_text, remove_columns=[c for c in val_ds.features if c != "text"])
    return train_ds, val_ds

# ── сохранение ──
def save_adapters(trainer, tokenizer, out_dir):
    lora_dir = os.path.join(out_dir, "lora_adapters")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"✓ LoRA adapters saved to: {lora_dir}", flush=True)

def merge_fp16(trainer, tokenizer, out_dir):
    merged_dir = os.path.join(out_dir, "merged_fp16")
    os.makedirs(merged_dir, exist_ok=True)
    peft_model = trainer.model
    try:
        peft_model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
            safe_serialization=True,
        )
        print(f"✓ Merged FP16 saved to: {merged_dir}", flush=True)
    except AttributeError:
        print("⚠️ save_pretrained_merged недоступен в вашей связке Unsloth/PEFT. Пропускаю merge.", flush=True)

# ── main ──
def main():
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train_ds, val_ds = load_text_dataset()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,   # один процесс
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
        use_fused_cross_entropy=False,
    )

    print(">>> Start training...", flush=True)
    trainer.train()
    print(">>> Training done.", flush=True)

    save_adapters(trainer, tokenizer, args.output_dir)
    if int(args.merge_fp16) == 1:
        merge_fp16(trainer, tokenizer, args.output_dir)

if __name__ == "__main__":
    main()

