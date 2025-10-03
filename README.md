
# Unsloth + Qwen3 Setup for Linux + 2 X NVIDIA RTX 5060 Ti + Google Colab (Local)



## ✅ Проверено и работает

- `Qwen3-30B`, `Qwen3-14B`, `Qwen3-8B`, `Qwen3-4B` (в int4, int8 и FP16)


# 🚀 Launch

### 1. `accelerate launch unsloth_Accelerate.py`  (< 14B)

### 2. `qwen30b_lora.py` ( For 30b mods = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] )*

### 3.  `Google Colab (Local)` https://colab.research.google.com/drive/1vpPA8bpQb0XdSn9_w_OJGZjHmJwEga8R?usp=sharing
```
export CUDA_VISIBLE_DEVICES=1,0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u qwen30bDS.py \
  --model "/home/skyruller/text-generation-webui/user_data/models/Qwen3-4B-Instruct-2507" \
  --dataset "/media/skyruller/NovyTom/dataset/opensloth_powermill_dataset/dataset.jsonl" \
  --max_seq 128 --lora_r 24 --ga 4 --lr 2e-4 \
  --output_dir "outputs_30b" --merge_fp16 0 \
  --max_memory "16GiB,16GiB"
```
* Если не хватает VRAM, то только mods = ["q_proj","k_proj","v_proj","o_proj"]
  тогда можно поднять RANK до 128 --max_seq 1024


## 🧠 Советы

- Для экономии VRAM используйте FlashAttention 2

 

### `Qwen3-14B accelerate launch unsloth_Accelerate.py`

```
[PID 21602] Script start. Python version: 3.13.7 | packaged by Anaconda, Inc. | (main, Sep  9 2025, 19:59:03) [GCC 11.2.0]
[PID 21602] Current PWD: /home/skyruller/unsloth-5090-multiple
[PID 21602] TORCH_DISTRIBUTED_USE_DTENSOR: 0
[PID 21602] CUDA_VISIBLE_DEVICES (from env): None
[PID 21602] ACCELERATE_USE_TP: false
[PID 21602] Launcher Env: RANK=0, LOCAL_RANK=0, WORLD_SIZE=2
[PID 21603] Script start. Python version: 3.13.7 | packaged by Anaconda, Inc. | (main, Sep  9 2025, 19:59:03) [GCC 11.2.0]
[PID 21603] Current PWD: /home/skyruller/unsloth-5090-multiple
[PID 21603] TORCH_DISTRIBUTED_USE_DTENSOR: 0
[PID 21603] CUDA_VISIBLE_DEVICES (from env): None
[PID 21603] ACCELERATE_USE_TP: false
[PID 21603] Launcher Env: RANK=1, LOCAL_RANK=1, WORLD_SIZE=2
[PID 21602, Rank 0] Imported torch. Version: 2.8.0+cu128. CUDA available: True
[PID 21603, Rank 1] Imported torch. Version: 2.8.0+cu128. CUDA available: True
[PID 21602, Rank 0] CUDA device count: 2
[PID 21603, Rank 1] CUDA device count: 2
[PID 21602, Rank 0] Set CUDA device to: cuda:0
[PID 21602, Rank 0] Name of current CUDA device: NVIDIA GeForce RTX 5060 Ti
[PID 21603, Rank 1] Set CUDA device to: cuda:1
[PID 21603, Rank 1] Name of current CUDA device: NVIDIA GeForce RTX 5060 Ti
✅ [PID 21602, Rank 0] Successfully patched DTensor._op_dispatcher.sharding_propagator.propagate.
[PID 21602, Rank 0] Importing accelerate...
✅ [PID 21603, Rank 1] Successfully patched DTensor._op_dispatcher.sharding_propagator.propagate.
[PID 21603, Rank 1] Importing accelerate...
[PID 21602, Rank 0] Imported accelerate.
[PID 21602, Rank 0] Importing Unsloth...
[PID 21603, Rank 1] Imported accelerate.
[PID 21603, Rank 1] Importing Unsloth...
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
........................................
==((====))==  Unsloth 2025.9.6: Fast Qwen3 patching. Transformers: 4.55.4.
   \\   /|    NVIDIA GeForce RTX 5060 Ti. Num GPUs = 2. Max memory: 15.444 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
==((====))==  Unsloth 2025.9.6: Fast Qwen3 patching. Transformers: 4.55.4.
   \\   /|    NVIDIA GeForce RTX 5060 Ti. Num GPUs = 2. Max memory: 15.444 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:28<00:00, 26.04s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:28<00:00, 26.05s/it]
[PID 21602, Rank 0] FastLanguageModel.from_pretrained successful.
..........................................
Unsloth is running with multi GPUs - the effective batch size is multiplied by 2
Unsloth is running with multi GPUs - the effective batch size is multiplied by 2
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 362 | Num Epochs = 15 | Total steps = 2,715
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 2 | Total batch size (1 x 1 x 2) = 2
 "-____-"     Trainable parameters = 32,112,640 of 8,194,391,040 (0.39% trained)
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 362 | Num Epochs = 15 | Total steps = 2,715
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 2 | Total batch size (1 x 1 x 2) = 2
 "-____-"     Trainable parameters = 32,112,640 of 8,194,391,040 (0.39% trained)
  0%|▍                                                                                                                                                 | 9/2715 [00:08<37:08,  1.21it/s]Unsloth: Will smartly offload gradients to save VRAM!
  1%|▉                                                                                                                                                | 18/2715 [00:15<36:41,  1.23it/s]Unsloth: Will smartly offload gradients to save VRAM!
{'loss': 2.6813, 'grad_norm': 1.4616115093231201, 'learning_rate': 0.00019786372007366484, 'epoch': 0.17}                                                                               
{'loss': 0.9447, 'grad_norm': 1.287734866142273, 'learning_rate': 0.00019565377532228362, 'epoch': 0.33}                     
```


### `Unsloth_Qwen3-30B-A3B-Instruct-2507 qwen30b_lora.py`


```
export CUDA_VISIBLE_DEVICES=1,0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u qwen30b.py \
  --model "/home/skyruller/webui/user_data/models/unsloth_Qwen3-30B-A3B-Instruct-2507" \
  --dataset "/media/skyruller/Novy/dataset/dataset_unescaped.jsonl" \
  --max_seq 256 \
  --lora_r 8 \
  --ga 8 \
  --lr 3e-4 \
  --epochs 25 \
  --targets 7 \
  --attn sdpa \
  --max_memory "13GiB,14GiB" \
  --output_dir "outputs_30b" \
  --merge_fp16 0
[PID 711757] Python: 3.13.7 | packaged by Anaconda, Inc. | (main, Sep  9 2025, 19:59:03) [GCC 11.2.0]
[PID 711757] PWD: /home/skyruller/unsloth-5090-multiple
Torch 2.8.0+cu128, CUDA available: True
CUDA devices visible: 2
  cuda:0 -> NVIDIA GeForce RTX 5060 Ti
  cuda:1 -> NVIDIA GeForce RTX 5060 Ti
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Using max_memory map: {0: '13GiB', 1: '14GiB'}
==((====))==  Unsloth 2025.9.6: Fast Qwen3_Moe patching. Transformers: 4.56.2.
   \\   /|    NVIDIA GeForce RTX 5060 Ti. Num GPUs = 2. Max memory: 15.477 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.4.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.33+115df95.d20250921. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.38it/s]
Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']
Unsloth: Making `model.base_model.model.model` require gradients
num_proc must be <= 9. Reducing num_proc to 9 for dataset of size 9.
[2025-09-21 22:04:00,088] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-09-21 22:04:00,340] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
>>> Start training...
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151654}.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 2
   \\   /|    Num examples = 416 | Num Epochs = 25 | Total steps = 1,300
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 8 x 1) = 8
 "-____-"     Trainable parameters = 283,508,736 of 30,815,631,360 (0.92% trained)
  0%|▍                                                                                                                                            | 4/1300 [02:41<14:17:29, 39.70s/it]Unsloth: Will smartly offload gradients to save VRAM!
{'loss': 1.5998, 'grad_norm': 0.5639712810516357, 'learning_rate': 0.0002956153846153846, 'epoch': 0.38}                                                                                
{'loss': 0.5247, 'grad_norm': 0.44476211071014404, 'learning_rate': 0.00029099999999999997, 'epoch': 0.77}                                                                              
{'loss': 0.318, 'grad_norm': 0.2609693109989166, 'learning_rate': 0.0002863846153846154, 'epoch': 1.15}                                                                                 
{'loss': 0.2046, 'grad_norm': 0.22140665352344513, 'learning_rate': 0.00028176923076923073, 'epoch': 1.54}                                                                              
........................................
........................................                                                                         
{'loss': 0.0832, 'grad_norm': 0.06220075488090515, 'learning_rate': 6.023076923076922e-05, 'epoch': 20.0}                                                                               
{'loss': 0.0728, 'grad_norm': 0.061242613941431046, 'learning_rate': 5.5615384615384614e-05, 'epoch': 20.38}                                                                            
 83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                       | 1077/1300 [12:27:51<2:42:15, 43.66s/it]
```


## Colab (Local) https://colab.research.google.com/drive/1vpPA8bpQb0XdSn9_w_OJGZjHmJwEga8R?usp=sharing

<img width="3398" height="1282" alt="image" src="https://github.com/user-attachments/assets/62cced4e-4d98-498a-afc8-482a4eac7f3a" />
<img width="1839" height="1325" alt="image" src="https://github.com/user-attachments/assets/761c91c3-50fa-4fcf-91ba-da3d1b1aa684" />


#### ⚡  Thanks "unsloth-5090-multiple"




**@Skyruller** — тестирование и отладка Qwen3 на RTX 5060 Ti  

