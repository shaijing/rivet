# Rivet Phase 0 性能基线（commit `023ed68`）

测量日期：2026-09-23。该报告覆盖 HEAD `023ed68c0b91d33cdf75618c31c2451218a7c1a8`。

为得到有效数据，对两个 benchmark harness 做了仅限测量逻辑的本地修改：`rivet-core/src/bench.rs` 将构造完成的 bytes 传给 `black_box`，避免 release 优化消除分配/写入；`pipeline_regression_bench.rs` 增加 plan state 与 first/p50/p95 输出。产品实现没有因此改变。这些测量 harness 修改与报告一起提交；被测产品实现的基准点仍是父提交 `023ed68`，复现时使用本提交中的 harness 版本。

## 环境

| 项目 | 值 |
| --- | --- |
| OS / kernel | Fedora 44, Linux 7.2.6, x86_64 |
| CPU | Intel Core Ultra 7 270K Plus，24 physical cores，24 online CPUs |
| GPU | NVIDIA GeForce RTX 4070 Ti，12,282 MiB；bench 未启用 GPU path |
| Rust | rustc 1.98.1 / cargo 1.98.0 |
| Python | CPython 3.13.12 (.venv) |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| CIFAR data | `/data/datasets/rivet/cifar10/train.lance` (50,000 rows, PNG); PyTorch CIFAR-10 cache at `/data/datasets/pytorch` |
| Criterion | 0.8.2；Rust microbench sample size 20，warmup 0.5s，measurement 1.5s |

CIFAR pipeline matrix 使用 release build、256 synthetic encoded samples、batch 32、8 batches、workers 0/1/4/24、prefetch 0/1/2；完整矩阵连续测量 4 次。CIFAR decoded comparison 使用 AGENTS.md 的命令，完整训练 split、batch 128、workers 4、3 epochs，独立运行 5 次。operator/dtype/normalize/fusion 各运行 3 次。pipeline matrix 重复 4 次；latency probe 重复 5 次。

## 关键基线

- CIFAR-10 decoded Rivet：均值 **723,429 img/s**，总体标准差 **9,020 img/s**，范围 **710,259–736,701 img/s**；TorchVision 当前均值见原始日志。相对 AGENTS.md 参考均值 731,303 img/s，本轮低 1.08%，且运行区间有重叠。
- Pipeline latency probe（5 次）：first batch 中位数 **0.703 ms**（0.689–0.834）；steady batch p50 中位数 **0.114 ms**（0.102–0.117）；steady batch p95 中位数 **0.119 ms**（0.106–0.123）；throughput 中位数 **1,065,427 img/s**。该 probe 是 128-image batch、40 个 steady batches 的 decoded-memory workload。
- `pipeline_baseline_bench` 全矩阵峰值 RSS：**193.3 MiB**（`/usr/bin/time -v`）；小型 latency probe 峰值 RSS 中位数 **11.5 MiB**。

### Pipeline workload 矩阵

下表对 4 次全矩阵中每个 workload 选择平均 throughput 最高的配置。

| workload | workers / prefetch | throughput mean ± σ (img/s) | first batch mean (ms) | steady batch mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| `decode` | 24 / 2 | 955,565 ± 83,551 | 0.071 | 0.034 |
| `full_cifar` | 24 / 2 | 559,904 ± 50,475 | 0.208 | 0.058 |
| `imagenet_random_resized_crop_flip_color_jitter_normalize_chw` | 24 / 2 | 8,354 ± 1,261 | 15.831 | 3.907 |
| `random_crop` | 24 / 2 | 1,070,986 ± 82,432 | 0.080 | 0.030 |
| `random_crop_flip` | 24 / 2 | 1,018,987 ± 70,820 | 0.074 | 0.032 |
| `random_crop_normalize` | 24 / 2 | 770,515 ± 31,470 | 0.109 | 0.042 |
| `random_crop_normalize_chw` | 24 / 2 | 775,624 ± 5,606 | 0.102 | 0.041 |

四轮矩阵中 24-worker 配置波动明显，尤其 ImageNet-style workload；解释时应以矩阵的重复分布为准，不应只拿单次峰值比较。

### CIFAR-10 数据集端到端对比

| benchmark | 配置 | Rivet img/s | Torch img/s | ratio / 其他 |
| --- | --- | ---: | ---: | --- |
| decoded cache vs torchvision | batch 128, workers 4, 3 epochs, 5 runs | 723,429 mean (σ 9,020) | 51,859 mean | mean ratio 13.95× |
| Lance lazy mode A (decode) | batch 128, resize 64, workers 4, 3 epochs | 84,224 | 29,710 | 2.83× |
| Lance lazy mode B (decode + resize) | same | 84,761 | 46,696 | 1.82× |
| Lance lazy mode C (+ float CHW) | same | 68,264 | 32,547 | 2.10× |
| Lance scaling, decoded cache, mode A/B/C | batch 128, 2 epochs | see extended raw output | — | cache later throughput A 101,430 / B 102,339 / C 88,646 img/s; peak RSS ≈1,261 MiB |

### 单算子 / batch 路径 benchmark

Normalize/fusion/dtype 是 wall-clock 小程序，表内为 3 次运行中逐次 speedup（sample or unfused ÷ batch or fused），短时首轮受热身影响。

| benchmark | configuration | run speedups | median |
| --- | --- | --- | ---: |
| Normalize sample/batch | batch 128, 32×32×3, 50 iterations | 0.84×, 0.86×, 0.85× | 0.85× |
| Normalize + Layout unfused/fused | batch 128, 32×32×3, 50 iterations | 0.88×, 1.05×, 1.03× | 1.03× |
| dtype sample/batch | batch 128, 32×32×3, 50 iterations | 1.45×, 1.00×, 1.00× | 1.00× |

`operator_bench` 使用 Arrow CIFAR-10（50k rows），batch 128、workers 4、prefetch 2、30 iterations；表内是 3 次运行中位数。`dense_cache.get_batch` 与 `hwc_to_chw_view` 是 view/metadata 快路径，计时精度显示为 0.000 ms，其换算吞吐只作量级参考。

| operation | latency (ms) | throughput (img/s) | alloc calls | allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| `decoded_cache.get_many` | 0.009 (0.009–0.009) | 14,754,476 | 390 | 22,464 |
| `dense_cache.get_batch` | 0.000 (0.000–0.000) | 1,021,276,596 | 0 | 0 |
| `crop_view` | 0.004 (0.004–0.004) | 30,945,531 | 0 | 0 |
| `hwc_to_chw_view` | 0.000 (0.000–0.000) | 1,662,337,662 | 4 | 164 |
| `stack_contiguous` | 0.014 (0.014–0.016) | 9,015,016 | 13 | 400,520 |
| `stack_non_contiguous` | 0.340 (0.334–0.388) | 376,704 | 141 | 403,592 |
| `sample_normalize` | 0.160 (0.155–0.161) | 798,254 | 896 | 1,599,488 |
| `batch_normalize` | 0.193 (0.160–0.195) | 661,628 | 7 | 1,573,088 |
| `normalize_plus_layout` | 0.178 (0.141–0.180) | 718,699 | 11 | 1,573,252 |
| `normalize_to_chw_fused` | 0.178 (0.175–0.179) | 718,889 | 7 | 1,573,088 |
| `full_pipeline` | 0.495 (0.494–0.496) | 258,492 | 2,146 | 6,576,996 |

### Criterion 微基准

Criterion 的 time/throughput 是 20 样本估计区间（下/中/上）；修改后的 aligned-buffer benchmark 消费输出 slice，避免前一版在 release 下把写入优化掉。

#### `rivet-core/aligned_buffer`

| case | time estimate | throughput estimate |
| --- | --- | --- |
| `construction/sequential/Vec/3 KiB` | 1.1645 µs 1.1695 µs 1.1746 µs | 2.4358 GiB/s 2.4463 GiB/s 2.4568 GiB/s |
| `construction/sequential/AlignedBufferBuilder/3 KiB` | 51.278 ns 51.470 ns 51.659 ns | 55.383 GiB/s 55.586 GiB/s 55.794 GiB/s |
| `construction/sequential/Vec/12 KiB` | 4.6325 µs 4.6522 µs 4.6746 µs | 2.4481 GiB/s 2.4599 GiB/s 2.4704 GiB/s |
| `construction/sequential/AlignedBufferBuilder/12 KiB` | 110.13 ns 110.60 ns 111.20 ns | 102.92 GiB/s 103.48 GiB/s 103.92 GiB/s |
| `construction/sequential/Vec/384 KiB` | 173.15 µs 173.29 µs 173.43 µs | 2.1116 GiB/s 2.1133 GiB/s 2.1150 GiB/s |
| `construction/sequential/AlignedBufferBuilder/384 KiB` | 7.0049 µs 7.0395 µs 7.0701 µs | 51.797 GiB/s 52.023 GiB/s 52.279 GiB/s |
| `construction/sequential/Vec/1.5 MiB` | 698.12 µs 698.54 µs 699.00 µs | 2.0956 GiB/s 2.0970 GiB/s 2.0983 GiB/s |
| `construction/sequential/AlignedBufferBuilder/1.5 MiB` | 30.824 µs 30.850 µs 30.890 µs | 47.422 GiB/s 47.483 GiB/s 47.522 GiB/s |
| `construction/sequential/Vec/12 MiB` | 6.0445 ms 6.0571 ms 6.0740 ms | 1.9293 GiB/s 1.9347 GiB/s 1.9388 GiB/s |
| `construction/sequential/AlignedBufferBuilder/12 MiB` | 516.43 µs 524.89 µs 537.70 µs | 21.794 GiB/s 22.326 GiB/s 22.692 GiB/s |
| `construction/extend_from_slice/Vec/3 KiB` | 45.991 ns 46.205 ns 46.479 ns | 61.555 GiB/s 61.921 GiB/s 62.209 GiB/s |
| `construction/extend_from_slice/AlignedBufferBuilder/3 KiB` | 57.976 ns 58.610 ns 59.076 ns | 48.430 GiB/s 48.815 GiB/s 49.348 GiB/s |
| `construction/extend_from_slice/Vec/12 KiB` | 87.340 ns 87.672 ns 88.070 ns | 129.94 GiB/s 130.53 GiB/s 131.03 GiB/s |
| `construction/extend_from_slice/AlignedBufferBuilder/12 KiB` | 99.966 ns 100.17 ns 100.31 ns | 114.08 GiB/s 114.24 GiB/s 114.48 GiB/s |
| `construction/extend_from_slice/Vec/384 KiB` | 4.9906 µs 4.9961 µs 5.0011 µs | 73.226 GiB/s 73.299 GiB/s 73.380 GiB/s |
| `construction/extend_from_slice/AlignedBufferBuilder/384 KiB` | 5.1440 µs 5.1450 µs 5.1462 µs | 71.161 GiB/s 71.178 GiB/s 71.192 GiB/s |
| `construction/extend_from_slice/Vec/1.5 MiB` | 26.828 µs 26.878 µs 26.917 µs | 54.421 GiB/s 54.499 GiB/s 54.601 GiB/s |
| `construction/extend_from_slice/AlignedBufferBuilder/1.5 MiB` | 24.959 µs 25.014 µs 25.055 µs | 58.466 GiB/s 58.562 GiB/s 58.691 GiB/s |
| `construction/extend_from_slice/Vec/12 MiB` | 515.20 µs 516.00 µs 516.68 µs | 22.681 GiB/s 22.711 GiB/s 22.746 GiB/s |
| `construction/extend_from_slice/AlignedBufferBuilder/12 MiB` | 517.50 µs 521.18 µs 527.35 µs | 22.222 GiB/s 22.485 GiB/s 22.645 GiB/s |
| `access/per_element_get/3 KiB` | 4.1174 µs 4.1238 µs 4.1288 µs | 709.57 MiB/s 710.43 MiB/s 711.53 MiB/s |
| `access/validated_strided/3 KiB` | 4.1673 µs 4.1705 µs 4.1742 µs | 701.85 MiB/s 702.48 MiB/s 703.02 MiB/s |
| `access/contiguous_slice/3 KiB` | 288.07 ns 288.43 ns 288.83 ns | 9.9056 GiB/s 9.9194 GiB/s 9.9316 GiB/s |
| `access/per_element_get/12 KiB` | 16.667 µs 16.726 µs 16.784 µs | 698.21 MiB/s 700.62 MiB/s 703.10 MiB/s |
| `access/validated_strided/12 KiB` | 16.636 µs 16.655 µs 16.676 µs | 702.72 MiB/s 703.62 MiB/s 704.44 MiB/s |
| `access/contiguous_slice/12 KiB` | 1.1570 µs 1.1611 µs 1.1651 µs | 9.8227 GiB/s 9.8562 GiB/s 9.8913 GiB/s |
| `access/per_element_get/384 KiB` | 523.55 µs 523.78 µs 524.04 µs | 715.60 MiB/s 715.95 MiB/s 716.26 MiB/s |
| `access/validated_strided/384 KiB` | 541.25 µs 542.80 µs 544.80 µs | 688.32 MiB/s 690.86 MiB/s 692.84 MiB/s |
| `access/contiguous_slice/384 KiB` | 36.801 µs 36.926 µs 37.143 µs | 9.8594 GiB/s 9.9173 GiB/s 9.9511 GiB/s |
| `access/per_element_get/1.5 MiB` | 2.0848 ms 2.0866 ms 2.0887 ms | 718.13 MiB/s 718.88 MiB/s 719.49 MiB/s |
| `access/validated_strided/1.5 MiB` | 2.1312 ms 2.1327 ms 2.1341 ms | 702.86 MiB/s 703.32 MiB/s 703.81 MiB/s |
| `access/contiguous_slice/1.5 MiB` | 146.04 µs 146.05 µs 146.05 µs | 10.029 GiB/s 10.030 GiB/s 10.030 GiB/s |
| `access/per_element_get/12 MiB` | 16.791 ms 16.871 ms 16.950 ms | 707.95 MiB/s 711.26 MiB/s 714.69 MiB/s |
| `access/validated_strided/12 MiB` | 17.230 ms 17.302 ms 17.375 ms | 690.65 MiB/s 693.54 MiB/s 696.47 MiB/s |
| `access/contiguous_slice/12 MiB` | 1.2468 ms 1.2643 ms 1.2949 ms | 9.0502 GiB/s 9.2693 GiB/s 9.3989 GiB/s |

#### `rivet-data/mmap_arrow_rows`

| case | time estimate | throughput estimate |
| --- | --- | --- |
| `mmap_arrow_row_access/row/sequential/128` | 6.0066 µs 6.0159 µs 6.0240 µs | 21.248 Melem/s 21.277 Melem/s 21.310 Melem/s |
| `mmap_arrow_row_access/rows_grouped/sequential/128` | 2.3437 µs 2.3443 µs 2.3447 µs | 54.590 Melem/s 54.602 Melem/s 54.614 Melem/s |
| `mmap_arrow_row_access/row/random/128` | 4.6315 µs 4.6480 µs 4.6609 µs | 27.463 Melem/s 27.538 Melem/s 27.637 Melem/s |
| `mmap_arrow_row_access/rows_grouped/random/128` | 6.3267 µs 6.3368 µs 6.3452 µs | 20.173 Melem/s 20.200 Melem/s 20.232 Melem/s |
| `mmap_arrow_row_access/row/repeated/128` | 4.6505 µs 4.6675 µs 4.6808 µs | 27.346 Melem/s 27.424 Melem/s 27.524 Melem/s |
| `mmap_arrow_row_access/rows_grouped/repeated/128` | 3.3579 µs 3.3648 µs 3.3715 µs | 37.965 Melem/s 38.040 Melem/s 38.119 Melem/s |
| `mmap_arrow_row_access/row/sequential/1024` | 48.791 µs 49.014 µs 49.191 µs | 20.817 Melem/s 20.892 Melem/s 20.988 Melem/s |
| `mmap_arrow_row_access/rows_grouped/sequential/1024` | 17.678 µs 17.746 µs 17.872 µs | 57.295 Melem/s 57.702 Melem/s 57.924 Melem/s |
| `mmap_arrow_row_access/row/random/1024` | 37.623 µs 37.871 µs 38.107 µs | 26.872 Melem/s 27.039 Melem/s 27.218 Melem/s |
| `mmap_arrow_row_access/rows_grouped/random/1024` | 31.769 µs 31.861 µs 31.934 µs | 32.066 Melem/s 32.139 Melem/s 32.233 Melem/s |
| `mmap_arrow_row_access/row/repeated/1024` | 37.663 µs 37.887 µs 38.078 µs | 26.892 Melem/s 27.028 Melem/s 27.188 Melem/s |
| `mmap_arrow_row_access/rows_grouped/repeated/1024` | 21.046 µs 21.080 µs 21.117 µs | 48.491 Melem/s 48.577 Melem/s 48.654 Melem/s |
| `mmap_arrow_row_access/row/sequential/8192` | 324.52 µs 325.69 µs 326.55 µs | 25.086 Melem/s 25.153 Melem/s 25.243 Melem/s |
| `mmap_arrow_row_access/rows_grouped/sequential/8192` | 137.06 µs 137.13 µs 137.23 µs | 59.694 Melem/s 59.738 Melem/s 59.771 Melem/s |
| `mmap_arrow_row_access/row/random/8192` | 296.04 µs 297.01 µs 297.87 µs | 27.502 Melem/s 27.581 Melem/s 27.672 Melem/s |
| `mmap_arrow_row_access/rows_grouped/random/8192` | 164.38 µs 164.62 µs 164.94 µs | 49.667 Melem/s 49.764 Melem/s 49.834 Melem/s |
| `mmap_arrow_row_access/row/repeated/8192` | 298.14 µs 299.44 µs 300.42 µs | 27.268 Melem/s 27.357 Melem/s 27.477 Melem/s |
| `mmap_arrow_row_access/rows_grouped/repeated/8192` | 142.25 µs 142.35 µs 142.45 µs | 57.507 Melem/s 57.550 Melem/s 57.590 Melem/s |

批量 Arrow 行读取在 18 种请求规模/访问模式中均通过顺序和重复项正确性断言；随机请求为 128 行时 grouped 版本较慢（6.34 µs vs row 4.65 µs），随机 1,024/8,192 行和顺序/重复索引则更快。

## 测试与可观测性

- 定向测试通过：`rivet-core --lib` 9/9，`rivet-data --lib` 20/20，`rivet-vision --lib` 130/130。
- 当前性能测量为 CPU pipeline；工作站有 NVIDIA RTX 4070 Ti，但这些 benchmark 没有实际跑 CUDA vision kernel。没有 trainer wait ratio、CUDA H2D/D2H bytes、kernel launch/sync 等数据，因当前 bench 没有训练循环或 GPU pipeline instrumentation。`nvidia-smi` 的一次快照显示桌面 GPU 有负载，不能当作 benchmark 的 GPU utilization。
- pipeline matrix 所在进程的 `/usr/bin/time -v` 报告平均 CPU 占用 190%（约 1.9 个逻辑 CPU 核，单次 3.26 s wall / 6.20 s user）；这不是整机利用率。`operator_bench` 记录 allocations；CIFAR extended comparison 记录 peak RSS。当前没有跨所有 workload 的 allocation/profiler 统一采集。
- `pipeline_regression_bench` 保存了当前代表 pipeline 编译摘要：input U8/HWC → pre-batch F32/HWC → output F32/CHW；2 sample ops、1 batch op（Resize / Layout）。

## 日志

按要求已删除 `docs/perf` 下的原始 `.log` 文件；本报告保留汇总性能数据和复现命令。

## 复现命令

```bash
cargo bench -j 12 -p rivet-core --bench aligned_buffer --features bench-internals -- --sample-size 20 --warm-up-time 0.5 --measurement-time 1.5
cargo bench -j 12 -p rivet-data --bench mmap_arrow_rows -- --sample-size 20 --warm-up-time 0.5 --measurement-time 1.5
cargo run -j 12 -p rivet-vision --release --no-default-features --example pipeline_baseline_bench
RIVET_NORMALIZE_BENCH_ITERS=50 cargo run -j 12 -p rivet-vision --release --no-default-features --example normalize_bench
RIVET_FUSION_BENCH_ITERS=50 cargo run -j 12 -p rivet-vision --release --no-default-features --example fusion_bench
RIVET_DTYPE_BENCH_ITERS=50 cargo run -j 12 -p rivet-vision --release --no-default-features --example dtype_bench
cargo run -j 12 -p rivet-vision --release --no-default-features --example pipeline_regression_bench
RIVET_OPERATOR_BENCH_ARROW=<cifar10-train.arrow> RIVET_OPERATOR_BENCH_ITERS=30 cargo run -j 12 -p rivet-vision --release --example operator_bench
maturin develop --release -j 12
.venv/bin/python bench/compare_cifar10_torch_rivet.py --modes A,B,C --batch 128 --batches 0 --resize 64 --rivet-workers 4 --torch-workers 4 --epochs 3
.venv/bin/python bench/compare_cifar10_torch_rivet.py --modes A,B,C --batch 128 --batches 0 --resize 64 --rivet-workers 4 --torch-workers 4 --epochs 2 --sweep --cache-compare --cache-level decoded
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py --torch-root /data/datasets/pytorch --rivet-root /data/datasets/rivet/cifar10 --batch 128 --workers 4 --epochs 3
cargo test -j 12 -p rivet-core --lib
cargo test -j 12 -p rivet-data --lib
cargo test -j 12 -p rivet-vision --lib
```
