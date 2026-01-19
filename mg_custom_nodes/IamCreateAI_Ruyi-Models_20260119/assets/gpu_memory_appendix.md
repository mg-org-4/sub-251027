# GPU Memory Appendix

On this page, we will display the required memory and execution time under different combinations of memory optimization parameters. We would also like to thank the community for their support in helping us enrich the table data.

Please note that due to variations in the running environment, the content in the table may differ from actual conditions and is for reference only.

## RTX 3090 Results

### Results from [bmgjet](https://github.com/bmgjet)

- Environments

| CPU | Ram | Disk | GPU |
| :---: | :---: | :---: | :---: |
| 7950X | 64GB 6200mhz, 30,32,32 | NVME 990 Pro | 3090 (Water Cooled 500W PL, +1000mhz Vram) Sits about 2.1ghz on core |

Memory usage reported from GPUz logs.

Torch was not compiled with flash attention.

Windows 11 Using Python Environment from Forge (webui_forge_cu124_torch24.7z) Feb 5 2024.

- Resolution of 512

| Num frames | Parameters | GPU memory | Time |
| :---: | :---: | :---: | :---: |
| 24 frames  | normal 0  | 18117mb |  1:32s -  3.80s/it |
| 48 frames  | normal 0  | 19995mb |  4:06s -  9.98s/it |
| 72 frames  | normal 0  | 22133mb |  9:57s - 23.88s/it |
| 96 frames  | normal 10 | 22200mb | 12:51s - 30.84s/it |
| 120 frames | normal 7  | 23195mb | 15:57s - 38.26s/it |

- Resolution of 768

| Num frames | Parameters | GPU memory | Time |
| :---: | :---: | :---: | :---: |
| 24 frames  | normal 0 | 22105mb |    4:07s - 9.94s/it   |
| 48 frames  | normal 7 | 22195mb |   14:34s - 34.98s/it  |
| 72 frames  | lowmem 0 | 17998mb |   38:29s - 91.99s/it  |
| 96 frames  | lowmem 0 | 23000mb | 1:04:16s - 153.74s/it |
| 120 frames | lowmem 0 | 28195mb | 5:29:54s - 797.28s/it <br> (froze system during vram overflows for a few sec each pass) |

## RTX 4090

### Results from CreateAI

> The GPU memory reported below is the `max_memory_allocated()` value. The values read from nvidia-smi may be higher than the reported values because CUDA occupies some GPU memory (usually between 500 - 800 MiB), and PyTorch's caching mechanism also requests additional GPU memory.

- Resolution of 384

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 15387MiB <br> _00:42s_ | 15052MiB <br> _00:50s_ | 14941MiB <br> _00:50s_ | 14867MiB <br> _00:57s_ | 14718MiB <br> _00:58s_ |
| 48 <br> frames  | 16686MiB <br> _01:15s_ | 16018MiB <br> _01:33s_ | 15795MiB <br> _01:35s_ | 15647MiB <br> _01:38s_ | 15350MiB <br> _01:40s_ |
| 72 <br> frames  | 17996MiB <br> _01:51s_ | 16993MiB <br> _02:12s_ | 16659MiB <br> _02:21s_ | 16436MiB <br> _02:22s_ | 15990MiB <br> _02:31s_ |
| 96 <br> frames  | 19297MiB <br> _02:31s_ | 17959MiB <br> _02:58s_ | 17514MiB <br> _03:09s_ | 17216MiB <br> _03:13s_ | 16621MiB <br> _03:27s_ |
| 120 <br> frames | 20599MiB <br> _03:17s_ | 18928MiB <br> _03:50s_ | 18371MiB <br> _04:09s_ | 17999MiB <br> _04:14s_ | 17256MiB <br> _04:29s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 4096MiB <br> _02:51s_ | 4096MiB <br> _03:25s_ | 4096MiB <br> _03:35s_ | 4096MiB <br> _02:45s_ | 4095MiB <br> _03:13s_ |
| 48 <br> frames  | 4148MiB <br> _03:37s_ | 4148MiB <br> _04:01s_ | 4148MiB <br> _04:19s_ | 4148MiB <br> _03:43s_ | 4147MiB <br> _04:52s_ |
| 72 <br> frames  | 4200MiB <br> _05:01s_ | 4200MiB <br> _05:40s_ | 4200MiB <br> _07:51s_ | 4200MiB <br> _05:01s_ | 4200MiB <br> _06:18s_ |
| 96 <br> frames  | 5248MiB <br> _05:59s_ | 4249MiB <br> _06:49s_ | 4249MiB <br> _08:43s_ | 4249MiB <br> _06:03s_ | 4249MiB <br> _07:09s_ |
| 120 <br> frames | 6549MiB <br> _07:24s_ | 4876MiB <br> _08:12s_ | 4321MiB <br> _10:45s_ | 4299MiB <br> _08:09s_ | 4299MiB <br> _08:32s_ |

- Resolution of 512

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 16366MiB <br> _01:18s_ | 15805MiB <br> _01:26s_ | 15607MiB <br> _01:37s_ | 15475MiB <br> _01:36s_ | 15211MiB <br> _01:39s_ |
| 48 <br> frames  | 18720MiB <br> _02:21s_ | 17532MiB <br> _02:49s_ | 17136MiB <br> _02:55s_ | 16872MiB <br> _02:58s_ | 16344MiB <br> _03:01s_ |
| 72 <br> frames  | 21036MiB <br> _03:41s_ | 19254MiB <br> _04:25s_ | 18660MiB <br> _04:34s_ | 18264MiB <br> _04:36s_ | 17472MiB <br> _04:51s_ |
| 96 <br> frames  | -----MiB <br> _--:--s_ | 20972MiB <br> _06:18s_ | 20180MiB <br> _06:24s_ | 19652MiB <br> _06:36s_ | 18596MiB <br> _06:56s_ |
| 120 <br> frames | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 21704MiB <br> _08:50s_ | 21044MiB <br> _08:53s_ | 19724MiB <br> _09:08s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  4212MiB <br> _04:31s_ | 4212MiB <br> _04:44s_ | 4211MiB <br> _05:12s_ | 4211MiB <br> _04:51s_ | 4212MiB <br> _04:14s_ |
| 48 <br> frames  |  4666MiB <br> _05:17s_ | 4401MiB <br> _06:11s_ | 4400MiB <br> _06:33s_ | 4399MiB <br> _08:29s_ | 4401MiB <br> _06:18s_ |
| 72 <br> frames  |  6981MiB <br> _07:27s_ | 5199MiB <br> _08:00s_ | 4605MiB <br> _08:57s_ | 4598MiB <br> _11:02s_ | 4598MiB <br> _08:46s_ |
| 96 <br> frames  |  9298MiB <br> _09:33s_ | 6922MiB <br> _10:13s_ | 6130MiB <br> _11:31s_ | 5602MiB <br> _13:46s_ | 4794MiB <br> _11:29s_ |
| 120 <br> frames | 11613MiB <br> _11:33s_ | 8643MiB <br> _13:11s_ | 7653MiB <br> _14:18s_ | 6993MiB <br> _14:23s_ | 5673MiB <br> _14:16s_ |

- Resolution of 640

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 17671MiB <br> _01:47s_ | 16739MiB <br> _02:09s_ | 16429MiB <br> _02:17s_ | 16234MiB <br> _02:23s_ | 15850MiB <br> _02:28s_ |
| 48 <br> frames  | 21324MiB <br> _03:49s_ | 19468MiB <br> _04:23s_ | 18849MiB <br> _04:55s_ | 18437MiB <br> _04:59s_ | 17612MiB <br> _05:14s_ |
| 72 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 21230MiB <br> _07:57s_ | 20611MiB <br> _08:13s_ | 19373MiB <br> _08:25s_ |
| 96 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 21134MiB <br> _11:57s_ |
| 120 <br> frames | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  4301MiB <br> _03:49s_ |  4301MiB <br> _04:31s_ |  4301MiB <br> _04:55s_ |  4301MiB <br> _06:33s_ | 4301MiB <br> _04:52s_ |
| 48 <br> frames  |  7271MiB <br> _07:05s_ |  5416MiB <br> _08:04s_ |  4798MiB <br> _08:07s_ |  4589MiB <br> _09:42s_ | 4589MiB <br> _09:18s_ |
| 72 <br> frames  | 10889MiB <br> _10:38s_ |  8106MiB <br> _11:54s_ |  7179MiB <br> _11:58s_ |  6560MiB <br> _12:52s_ | 5322MiB <br> _13:13s_ |
| 96 <br> frames  | 14509MiB <br> _14:13s_ | 10795MiB <br> _15:50s_ |  9557MiB <br> _16:10s_ |  8732MiB <br> _17:08s_ | 7082MiB <br> _17:39s_ |
| 120 <br> frames | 18128MiB <br> _18:26s_ | 13487MiB <br> _19:57s_ | 11942MiB <br> _21:12s_ | 10910MiB <br> _21:55s_ | 8847MiB <br> _26:20s_ |

- Resolution of 768

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 19223MiB <br> _02:38s_ | 17900MiB <br> _03:06s_ | 17448MiB <br> _03:18s_ | 17153MiB <br> _03:23s_ | 16624MiB <br> _03:34s_ |
| 48 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 20946MiB <br> _07:28s_ | 20352MiB <br> _07:35s_ | 19164MiB <br> _08:04s_ |
| 72 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |
| 96 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |
| 120 <br> frames | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  5251MiB <br> _05:19s_ |  4398MiB <br> _05:55s_ |  4398MiB <br> _06:52s_ |  4398MiB <br> _07:24s_ |  4399MiB <br> _07:35s_ |
| 48 <br> frames  | 10457MiB <br> _09:38s_ |  7786MiB <br> _11:09s_ |  6896MiB <br> _12:43s_ |  6304MiB <br> _12:52s_ |  5114MiB <br> _13:20s_ |
| 72 <br> frames  | 15671MiB <br> _15:14s_ | 11661MiB <br> _17:50s_ | 10325MiB <br> _18:22s_ |  9433MiB <br> _19:32s_ |  7652MiB <br> _19:19s_ |
| 96 <br> frames  | -----MiB <br> _--:--s_ | 15534MiB <br> _24:30s_ | 13752MiB <br> _24:28s_ | 12564MiB <br> _25:54s_ | 10188MiB <br> _29:19s_ |
| 120 <br> frames | -----MiB <br> _--:--s_ | 19406MiB <br> _33:04s_ | 17179MiB <br> _33:14s_ | 15694MiB <br> _35:01s_ | 12724MiB <br> _39:11s_ |

- Resolution of 896

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 21085MiB <br> _03:46s_ | 19265MiB <br> _04:25s_ | 18659MiB <br> _04:39s_ | 18248MiB <br> _04:43s_ | 17543MiB <br> _04:59s_ |
| 48 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 20994MiB <br> _11:34s_ |
| 72 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |
| 96 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |
| 120 <br> frames | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  7128MiB <br> _07:26s_ |  5309MiB <br> _07:45s_ |  4703MiB <br> _07:50s_ |  4517MiB <br> _08:29s_ |  4517MiB <br> _09:31s_ |
| 48 <br> frames  | 14220MiB <br> _14:05s_ | 10582MiB <br> _15:01s_ |  9370MiB <br> _15:43s_ |  8562MiB <br> _16:34s_ |  6946MiB <br> _17:36s_ |
| 72 <br> frames  | -----MiB <br> _--:--s_ | 15857MiB <br> _24:53s_ | 14038MiB <br> _25:06s_ | 12826MiB <br> _26:49s_ | 10401MiB <br> _27:30s_ |
| 96 <br> frames  | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 17087MiB <br> _38:52s_ | 13853MiB <br> _45:19s_ |
| 120 <br> frames | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | -----MiB <br> _--:--s_ | 17310MiB <br> _55:50s_ |

## A100

### Results from CreateAI

> The GPU memory reported below is the `max_memory_allocated()` value. The values read from nvidia-smi may be higher than the reported values because CUDA occupies some GPU memory (usually between 500 - 800 MiB), and PyTorch's caching mechanism also requests additional GPU memory.

- Resolution of 384

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 15387MiB <br> _00:36s_ | 15052MiB <br> _00:43s_ | 14941MiB <br> _00:50s_ | 14867MiB <br> _00:50s_ | 14717MiB <br> _00:56s_ |
| 48 <br> frames  | 16686MiB <br> _00:57s_ | 16018MiB <br> _01:10s_ | 15795MiB <br> _01:28s_ | 15647MiB <br> _01:16s_ | 15350MiB <br> _01:21s_ |
| 72 <br> frames  | 17996MiB <br> _01:29s_ | 16993MiB <br> _01:43s_ | 16659MiB <br> _01:53s_ | 16436MiB <br> _01:53s_ | 15990MiB <br> _01:57s_ |
| 96 <br> frames  | 19297MiB <br> _02:01s_ | 17959MiB <br> _02:21s_ | 17514MiB <br> _02:29s_ | 17216MiB <br> _02:33s_ | 16621MiB <br> _02:54s_ |
| 120 <br> frames | 20599MiB <br> _02:33s_ | 18928MiB <br> _03:04s_ | 18371MiB <br> _03:12s_ | 17999MiB <br> _03:16s_ | 17256MiB <br> _03:25s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 4096MiB <br> _03:56s_ | 4096MiB <br> _04:24s_ | 4096MiB <br> _04:16s_ | 4096MiB <br> _04:14s_ | 4095MiB <br> _04:41s_ |
| 48 <br> frames  | 4148MiB <br> _04:45s_ | 4148MiB <br> _05:27s_ | 4148MiB <br> _05:29s_ | 4148MiB <br> _05:32s_ | 4147MiB <br> _06:06s_ |
| 72 <br> frames  | 4200MiB <br> _06:10s_ | 4200MiB <br> _07:10s_ | 4200MiB <br> _07:13s_ | 4200MiB <br> _07:19s_ | 4200MiB <br> _07:24s_ |
| 96 <br> frames  | 5248MiB <br> _07:53s_ | 4249MiB <br> _08:16s_ | 4249MiB <br> _08:30s_ | 4249MiB <br> _08:28s_ | 4249MiB <br> _09:05s_ |
| 120 <br> frames | 6547MiB <br> _08:43s_ | 4876MiB <br> _09:54s_ | 4321MiB <br> _10:16s_ | 4299MiB <br> _10:20s_ | 4299MiB <br> _11:26s_ |

- Resolution of 512

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 16399MiB <br> _00:59s_ | 15805MiB <br> _01:07s_ | 15607MiB <br> _01:13s_ | 15475MiB <br> _01:18s_ | 15211MiB <br> _01:16s_ |
| 48 <br> frames  | 18720MiB <br> _01:47s_ | 17532MiB <br> _02:10s_ | 17136MiB <br> _02:25s_ | 16872MiB <br> _02:22s_ | 16344MiB <br> _02:24s_ |
| 72 <br> frames  | 21036MiB <br> _02:55s_ | 19254MiB <br> _03:27s_ | 18660MiB <br> _03:36s_ | 18264MiB <br> _03:41s_ | 17472MiB <br> _03:51s_ |
| 96 <br> frames  | 23348MiB <br> _04:08s_ | 20972MiB <br> _04:46s_ | 20180MiB <br> _05:05s_ | 19652MiB <br> _05:12s_ | 18596MiB <br> _05:38s_ |
| 120 <br> frames | 25664MiB <br> _05:42s_ | 22694MiB <br> _06:28s_ | 21704MiB <br> _06:47s_ | 21044MiB <br> _06:50s_ | 19724MiB <br> _07:06s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  4212MiB <br> _05:03s_ | 4213MiB <br> _05:31s_ | 4212MiB <br> _05:23s_ | 4212MiB <br> _05:50s_ | 4212MiB <br> _05:51s_ |
| 48 <br> frames  |  4666MiB <br> _06:47s_ | 4401MiB <br> _07:13s_ | 4401MiB <br> _07:34s_ | 4400MiB <br> _08:21s_ | 4400MiB <br> _08:19s_ |
| 72 <br> frames  |  6981MiB <br> _08:41s_ | 5199MiB <br> _09:44s_ | 4605MiB <br> _10:10s_ | 4598MiB <br> _11:13s_ | 4598MiB <br> _12:00s_ |
| 96 <br> frames  |  9298MiB <br> _10:59s_ | 6922MiB <br> _12:10s_ | 6130MiB <br> _12:16s_ | 5602MiB <br> _13:27s_ | 4794MiB <br> _15:17s_ |
| 120 <br> frames | 11613MiB <br> _13:27s_ | 8643MiB <br> _14:55s_ | 7653MiB <br> _15:22s_ | 6993MiB <br> _16:04s_ | 5673MiB <br> _18:04s_ |

- Resolution of 640

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 17707MiB <br> _01:26s_ | 16778MiB <br> _01:45s_ | 16470MiB <br> _01:48s_ | 16263MiB <br> _01:51s_ | 15850MiB <br> _01:53s_ |
| 48 <br> frames  | 21324MiB <br> _03:09s_ | 19468MiB <br> _03:36s_ | 18849MiB <br> _03:53s_ | 18437MiB <br> _03:53s_ | 17612MiB <br> _04:03s_ |
| 72 <br> frames  | 24943MiB <br> _05:06s_ | 22158MiB <br> _05:54s_ | 21230MiB <br> _06:12s_ | 20612MiB <br> _06:18s_ | 19374MiB <br> _06:35s_ |
| 96 <br> frames  | 28562MiB <br> _07:35s_ | 24848MiB <br> _08:36s_ | 23611MiB <br> _09:04s_ | 22785MiB <br> _09:07s_ | 21134MiB <br> _09:35s_ |
| 120 <br> frames | 32182MiB <br> _10:32s_ | 27541MiB <br> _11:54s_ | 25994MiB <br> _12:20s_ | 24962MiB <br> _12:37s_ | 22899MiB <br> _13:05s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  4301MiB <br> _05:40s_ |  4301MiB <br> _06:07s_ |  4301MiB <br> _06:27s_ |  4301MiB <br> _06:33s_ | 4301MiB <br> _06:57s_ |
| 48 <br> frames  |  7271MiB <br> _09:18s_ |  5416MiB <br> _09:51s_ |  4798MiB <br> _10:26s_ |  4589MiB <br> _10:36s_ | 4589MiB <br> _11:52s_ |
| 72 <br> frames  | 10889MiB <br> _13:03s_ |  8106MiB <br> _14:07s_ |  7179MiB <br> _14:35s_ |  6561MiB <br> _17:30s_ | 5322MiB <br> _16:33s_ |
| 96 <br> frames  | 14509MiB <br> _16:57s_ | 10795MiB <br> _18:12s_ |  9557MiB <br> _18:29s_ |  8732MiB <br> _18:55s_ | 7082MiB <br> _21:10s_ |
| 120 <br> frames | 18128MiB <br> _22:05s_ | 13487MiB <br> _22:49s_ | 11942MiB <br> _23:47s_ | 10910MiB <br> _23:50s_ | 8847MiB <br> _32:46s_ |

- Resolution of 768

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 19300MiB <br> _02:09s_ | 17962MiB <br> _02:32s_ | 17517MiB <br> _02:37s_ | 17219MiB <br> _02:39s_ | 16624MiB <br> _02:46s_ |
| 48 <br> frames  | 24510MiB <br> _04:46s_ | 21837MiB <br> _05:32s_ | 20946MiB <br> _05:51s_ | 20352MiB <br> _05:59s_ | 19164MiB <br> _06:10s_ |
| 72 <br> frames  | 29725MiB <br> _08:23s_ | 25715MiB <br> _09:35s_ | 24378MiB <br> _09:56s_ | 23486MiB <br> _10:05s_ | 21703MiB <br> _10:40s_ |
| 96 <br> frames  | 34925MiB <br> _12:52s_ | 29579MiB <br> _14:33s_ | 27797MiB <br> _15:01s_ | 26609MiB <br> _15:13s_ | 24233MiB <br> _15:53s_ |
| 120 <br> frames | 40134MiB <br> _18:29s_ | 33451MiB <br> _20:39s_ | 31223MiB <br> _21:31s_ | 29737MiB <br> _21:42s_ | 26766MiB <br> _22:51s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  5251MiB <br> _08:04s_ |  4398MiB <br> _08:15s_ |  4398MiB <br> _08:21s_ |  4398MiB <br> _08:48s_ |  4399MiB <br> _09:14s_ |
| 48 <br> frames  | 10457MiB <br> _12:42s_ |  7786MiB <br> _13:54s_ |  6896MiB <br> _14:17s_ |  6304MiB <br> _14:37s_ |  5114MiB <br> _16:21s_ |
| 72 <br> frames  | 15671MiB <br> _18:18s_ | 11661MiB <br> _19:44s_ | 10325MiB <br> _20:53s_ |  9435MiB <br> _20:47s_ |  7652MiB <br> _23:42s_ |
| 96 <br> frames  | 20880MiB <br> _24:55s_ | 15534MiB <br> _27:07s_ | 13752MiB <br> _28:13s_ | 12564MiB <br> _28:44s_ | 10188MiB <br> _30:30s_ |
| 120 <br> frames | 26092MiB <br> _31:21s_ | 19406MiB <br> _34:09s_ | 17179MiB <br> _35:13s_ | 15694MiB <br> _35:15s_ | 12724MiB <br> _32:55s_ |

- Resolution of 896

| Num <br> frames | normal_mode + 0 steps | normal_mode + 10 steps | normal_mode + 7 steps | normal_mode + 5 steps | normal_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  | 21181MiB <br> _02:58s_ | 19362MiB <br> _03:32s_ | 18755MiB <br> _03:48s_ | 18351MiB <br> _03:47s_ | 17543MiB <br> _03:54s_ |
| 48 <br> frames  | 28271MiB <br> _07:18s_ | 24633MiB <br> _08:24s_ | 23420MiB <br> _08:49s_ | 22611MiB <br> _08:53s_ | 20994MiB <br> _09:10s_ |
| 72 <br> frames  | 35360MiB <br> _13:22s_ | 29902MiB <br> _14:57s_ | 28083MiB <br> _15:33s_ | 26870MiB <br> _15:46s_ | 24445MiB <br> _16:13s_ |
| 96 <br> frames  | 42447MiB <br> _21:08s_ | 35169MiB <br> _23:48s_ | 32744MiB <br> _24:41s_ | 31126MiB <br> _25:08s_ | 27891MiB <br> _26:33s_ |
| 120 <br> frames | 49538MiB <br> _30:55s_ | 40442MiB <br> _34:03s_ | 37410MiB <br> _35:11s_ | 35388MiB <br> _35:31s_ | 31345MiB <br> _36:40s_ |

| Num <br> frames | low_gpu_mode + 0 steps | low_gpu_mode + 10 steps | low_gpu_mode + 7 steps | low_gpu_mode + 5 steps | low_gpu_mode + 1 steps |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 24 <br> frames  |  7128MiB <br> _08:52s_ |  5309MiB <br> _09:30s_ |  4702MiB <br> _10:37s_ |  4517MiB <br> _10:55s_ |  4517MiB <br> _12:09s_ |
| 48 <br> frames  | 14220MiB <br> _15:56s_ | 10582MiB <br> _17:04s_ |  9370MiB <br> _18:49s_ |  8562MiB <br> _19:08s_ |  6946MiB <br> _21:50s_ |
| 72 <br> frames  | 21315MiB <br> _24:58s_ | 15857MiB <br> _26:54s_ | 14038MiB <br> _29:10s_ | 12826MiB <br> _29:42s_ | 10401MiB <br> _30:58s_ |
| 96 <br> frames  | 28407MiB <br> _31:16s_ | 21130MiB <br> _34:22s_ | 18704MiB <br> _36:45s_ | 17087MiB <br> _36:34s_ | 13853MiB <br> _43:48s_ |
| 120 <br> frames | 35504MiB <br> _45:00s_ | 26405MiB <br> _48:05s_ | 23373MiB <br> _49:27s_ | 21352MiB <br> _50:34s_ | 17310MiB <br> _59:38s_ |
