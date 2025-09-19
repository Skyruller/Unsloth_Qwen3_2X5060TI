# Unsloth_Qwen3_2X5060TI


### ✅ Confirmed Working:

*  < `Qwen3-14B`
  

## 🚀 Launch

```bash
accelerate launch unsloth_Accelerate.py
```

---

## ⚡ Benchmarks (GTX 5060 Ti)

### Qwen3-14B

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

## 📁 Example `requirements.txt`

You can use this sample:

```
tokenizers==0.21.4
torch==2.8.0
torchao==0.13.0
torchvision==0.23.0
tornado==6.5.2
tqdm==4.67.1
traitlets @ file:///work/perseverance-python-buildout/croot/traitlets_1728385099292/work
transformers==4.55.4
triton==3.4.0
trl==0.22.2
typeguard==4.4.4
types-python-dateutil==2.9.0.20250822
typing-inspection==0.4.1
typing_extensions @ file:///croot/typing_extensions_1756280817316/work
tyro==0.9.31
tzdata==2025.2
unsloth @ git+https://github.com/unslothai/unsloth.git@7c59a9b63fab13cdfe3fe0f2d6b10c59bcd83ef4
unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git@8b312c089a92b5bcde02beb6d68f92e9d81c95fd
uri-template==1.3.0
```
## ⚡  Thanks "unsloth-5090-multiple"
