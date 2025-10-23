#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# ВАЖНО: Unsloth импортируем ПЕРВЫМ
# =========================
import os

# Безопасный режим: отключаем Triton RMSNorm (лечит "illegal memory access")
if os.environ.get("UNSLOTH_DISABLE_TRITON") is None:
    os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
# Чтобы не было неожиданной компиляции графов
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# Меньше болтовни токенайзеров
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Меньше фрагментации CUDA памяти
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from unsloth import FastLanguageModel  # noqa: E402

import io
import re
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW

# FSDP (чистый PyTorch)
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# =========================
# Утилиты
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0

def mp_print(*args, **kwargs):
    if is_main():
        print(*args, **kwargs, flush=True)

def find_decoder_layer_cls(model: nn.Module):
    """Ищем класс слоя декодера (например Qwen3DecoderLayer) для auto_wrap_policy."""
    for m in model.modules():
        name = m.__class__.__name__
        if name.endswith("DecoderLayer"):
            return m.__class__
    return None

def cast_model_dtype_(model: nn.Module, dtype: torch.dtype):
    """Выравниваем dtype у ВСЕХ float параметров и буферов (исключает смесь BF16/FP32)."""
    for p in model.parameters(recurse=True):
        if torch.is_floating_point(p):
            p.data = p.data.to(dtype)
    for b in model.buffers(recurse=True):
        if torch.is_floating_point(b):
            b.data = b.data.to(dtype)
    try:
        model.config.torch_dtype = dtype
    except Exception:
        pass
    return model

# =========================
# Парсинг ChatML-текстов
# =========================
_CHATML_START = re.compile(r"^\s*<\|im_start\|\>\s*([a-zA-Z_]+)\s*$")
_CHATML_END   = re.compile(r"^\s*<\|im_end\|\>\s*$")

def _parse_chatml_dialogs(text: str) -> List[List[Dict[str, str]]]:
    dialogs: List[List[Dict[str, str]]] = []
    cur: List[Dict[str, str]] = []
    role: Optional[str] = None
    buf: List[str] = []

    def flush_msg():
        nonlocal role, buf, cur
        if role is not None:
            content = "\n".join(buf).rstrip("\n")
            cur.append({"role": role, "content": content})
        role = None
        buf = []

    def flush_dialog():
        nonlocal cur
        if cur:
            dialogs.append(cur)
        cur = []

    f = io.StringIO(text)
    empty_run = 0
    for line in f:
        m_start = _CHATML_START.match(line)
        m_end   = _CHATML_END.match(line)

        if m_start:
            if role is not None and buf:
                flush_msg()
            role = m_start.group(1).strip().lower()
            empty_run = 0
            continue

        if m_end:
            flush_msg()
            empty_run = 0
            continue

        if line.strip() == "":
            empty_run += 1
            if empty_run >= 2:
                if role is not None or cur:
                    flush_msg()
                    flush_dialog()
            continue
        else:
            empty_run = 0

        if role is not None:
            buf.append(line.rstrip("\n"))

    if role is not None or cur:
        flush_msg()
        flush_dialog()

    cleaned = [d for d in dialogs if any(msg.get("role") for msg in d)]
    return cleaned

def load_texts_from_chatml_txt(path: str, tokenizer) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    dialogs = _parse_chatml_dialogs(raw)
    mp_print(f"[DATA] Диалогов (блоков): {len(dialogs)}")

    texts: List[str] = []
    for msgs in dialogs:
        norm = []
        for m in msgs:
            r = m["role"].lower()
            if r.startswith("sys"):
                r = "system"
            elif r.startswith("ass"):
                r = "assistant"
            elif r.startswith("usr") or r.startswith("user"):
                r = "user"
            norm.append({"role": r, "content": m["content"]})

        txt = tokenizer.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)
        texts.append(txt)
    return texts

# =========================
# Dataset + Collator
# =========================
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        att = torch.tensor(enc["attention_mask"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": att}

@dataclass
class DataCollator:
    pad_id: int
    label_pad: int = -100

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            ids = item["input_ids"]
            att = item["attention_mask"]
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_id, dtype=torch.long)], dim=0)
                att = torch.cat([att, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            lbl = ids.clone()
            lbl[ids == self.pad_id] = self.label_pad
            input_ids.append(ids)
            attention_mask.append(att)
            labels.append(lbl)
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

# =========================
# DDP/FSDP init/cleanup
# =========================
def ddp_setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.empty(1, device="cuda")
    return local_rank, dist.get_rank(), dist.get_world_size()

def ddp_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# =========================
# Построение модели
# =========================
def build_model_and_tokenizer(model_path: str, max_seq_len: int, dtype: torch.dtype, gc: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = model_path,
        dtype             = dtype,
        load_in_4bit      = False,
        device_map        = {"": "cpu"},  # FSDP сам управляет размещением
        low_cpu_mem_usage = True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_len

    FastLanguageModel.for_training(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing=("unsloth" if gc else False),
    )

    cast_model_dtype_(model, dtype)
    return model, tokenizer

# =========================
# Точка входа
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", type=str, default="0", help="GPU IDs, e.g. 0,1 (use torchrun for multi-GPU)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, default="./out_fsdp_lora")
    p.add_argument("--max_seq_len", type=int, default=896)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gc", action="store_true", help="Enable gradient checkpointing (unsloth)")

    args = p.parse_args()

    want_bf16 = args.dtype == "bf16"
    want_fp16 = args.dtype == "fp16"
    dtype = torch.bfloat16 if want_bf16 else (torch.float16 if want_fp16 else torch.float32)

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    distributed = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    
    if distributed:
        local_rank, rank, world_size = ddp_setup()
        if is_main():
            mp_print(f"[Init] procs={world_size} | out={args.out}")
            mp_print(f"[Paths] model={args.model}\n        data ={args.data}")
    else:
        rank = 0
        world_size = 1
        torch.cuda.set_device(int(args.devices.split(",")[0]))
        torch.empty(1, device="cuda")
        mp_print(f"[Init] single GPU | out={args.out}")

    # Загрузка модели и токенизатора
    model, tokenizer = build_model_and_tokenizer(args.model, args.max_seq_len, dtype, gc=args.gc)

    # Данные
    texts = load_texts_from_chatml_txt(args.data, tokenizer)
    if len(texts) == 0:
        raise RuntimeError("Датасет пуст после парсинга ChatML.")

    ds = TextDataset(texts, tokenizer, args.max_seq_len)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if distributed else None

    collate = DataCollator(pad_id=tokenizer.pad_token_id)
    dl = DataLoader(
        ds,
        batch_size=args.microbatch,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=False,
    )

    # Оптимизатор и LR scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    total_steps = math.ceil(len(ds) / (args.microbatch * world_size * max(1, args.grad_accum))) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: float(max(1, warmup_steps - step)) / max(1, warmup_steps) if step < warmup_steps
        else max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
    )

    # FSDP обёртка
    if distributed and world_size > 1:
        dec_cls = find_decoder_layer_cls(model)
        if dec_cls is None:
            ddp_cleanup()
            raise RuntimeError("Не найден класс DecoderLayer для auto_wrap_policy.")
        mp_print(f"[FSDP] auto_wrap по классу: {dec_cls.__name__}")

        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={dec_cls})
        mp_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            device_id=torch.device(f"cuda:{local_rank}"),
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
            mixed_precision=mp_policy,
        )
    else:
        model = model.cuda()

    # Функция сохранения
    def save_adapter(tag: str):
        save_dir = os.path.join(args.out, tag)
        
        # Все процессы должны войти в контекст
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            state_dict = model.state_dict()
            
            if is_main():
                os.makedirs(save_dir, exist_ok=True)
                # Извлекаем base_model из PEFT
                base_model = model._forward_module.base_model if hasattr(model, "_forward_module") else model.base_model
                base_model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=True)
                tokenizer.save_pretrained(save_dir)
                mp_print(f"💾 Сохранено: {save_dir}")

        # Синхронизация всех рангов
        if distributed:
            dist.barrier()

    # Обучение
    model.train()
    if distributed:
        dist.barrier()

    mp_print(f"🚀 Начинаем обучение ...")
    mp_print(f"[Config] dtype={args.dtype}, microbatch={args.microbatch}, grad_accum={args.grad_accum}, "
             f"steps_per_epoch={math.ceil(len(ds)/(args.microbatch*world_size))}, total_steps={total_steps}")

    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dl, start=1):
            # Убираем .cuda() — FSDP сам переместит данные
            input_ids = batch["input_ids"]          # уже на CPU, FSDP сам загрузит
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.grad_accum

            loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main() and (global_step <= 5 or global_step % args.log_every == 0):
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = loss.item() * args.grad_accum
                    mp_print(f"[step {global_step:>6}/{total_steps}] loss={avg_loss:.4f} lr={lr_now:.3e}")

                if args.save_every and global_step % args.save_every == 0:
                    save_adapter(f"checkpoint-step{global_step}")

        # Сохранение в конце эпохи
        save_adapter(f"checkpoint-epoch{epoch+1}")

    if is_main():
        dt = time.time() - t0
        mp_print(f"✅ Обучение завершено. Шагов: {global_step}, время: {dt/60:.1f} мин. Результат: {args.out}")

    if distributed:
        ddp_cleanup()

# =========================
# Запуск
# =========================
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
