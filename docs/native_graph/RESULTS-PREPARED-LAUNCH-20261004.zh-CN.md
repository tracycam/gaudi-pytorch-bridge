# Prepared launch 与真实提交路径：续行检查点

2026-10-04，用户已要求继续，构建和实验恢复。远端系统时区为 UTC+8，部分日志为 10-05。
目标仍是可复用的 bridge 执行能力；权重、计算 kernel 和精度未改，应用继续使用
普通 `HPUGraph`。两个外置执行器保留为对照，尚未完成弃用条件。

**后续探针身份纠正：** 旧单卡 supervisor 只设置 `HABANA_VISIBLE_MODULES`，没有设置
bridge 实际用于选卡的 `HLS_MODULE_ID`。硬件 trace 证明一次请求 module5 的旧
探针实际运行在 module0/PCI43，且 CPU 绑定在 NUMA1。因此本文历史单卡数字不能
作为“module5、NUMA 本地”的证据；功能结果与原始计时保留，但旧微基准的本地性
与归因需重新验证。TP2 和完整模型 worker 已在 import 前设置 `HLS_MODULE_ID`，
不受这项探针 bug 影响。新 supervisor 同时设置两者，并检查实际 accel 文件描述符、
PCI 和 CPU 本地性。后续结果见
[依赖批处理检查点](RESULTS-DEPENDENCY-BATCH-20261004.zh-CN.md)。

## 本轮实现

1. `PreparedLaunchBindings` 在准备时保存抽象地址、allocator 槽和标量修正。
   replay 只恢复地址字段，不再复制整张 launch table、构造地址 vector 或
   HOST_TO_DEVICE 索引 map。不可变 Tensor ID、尺寸、类型与稳定名称保持原值。
   原 8 参数 `graph::launch` ABI 保留，新 overload 共用其底层提交和错误处理。
2. 地址仍每次经原 allocator 翻译并取得租约。没有缓存裸物理指针，也没有省掉
   workspace、producer/event 或异步保活。源码是减少静态准备，不是绕过资源语义。
3. 关闭 TOWL 时不再构造物理地址日志 vector；关闭 memory log 时不再先计算
   `Post-Tensors` 的 allocator 元数据。后者在 stringent pool 模式下会扫描和加锁。
   必需的内存预算与地址锁仍执行；未把重复扫描的删除包装成已证实的模型收益。
4. 修复并验收 CPU-only profiler：补 pybind STL 转换和 typed 默认参数，区分
   bridge import 自动设置的 marker 与外部 profiler 配置。CPU-only 探针不初始化
   HPU；模型诊断使用普通 Torch CPU profiler，放在性能请求之后。
5. 复用现有 RAII trace 增加计划入口、绑定预检、提交线程，以及内存预算/
   workspace、地址租约、Synapse 提交三个范围。没有新增 profiler 框架。

核心源码：[NativeExecutionPlan.cpp](../../backend/habana_device/NativeExecutionPlan.cpp)、
[prepared_launch_bindings.h](../../backend/synapse_helpers/prepared_launch_bindings.h)、
[graph.cpp](../../backend/synapse_helpers/graph.cpp)。

## 编译与指令证据

构建均使用上游 `.devops/build.py`，`-j 32`，独立源码/build/验证 venv；不修改共享
Torch 或系统 SDK。环境：Gaudi2、SDK 1.24.1、Python 3.12、Torch
`2.11.0a0+git009b5f6`。具体 wheel 身份与源码 HEAD 分开记录。

| wheel commit | 用途 | 进展 |
|---|---|---|
| `503b5c5f8` | 新鲜旧候选对照 | 完整 70 层 A1 已完成 |
| `3d2111584` | prepared launch + CPU profiler | 单卡、TP2、完整 70 层 C1 已通过 |
| `a444c1787` | 关闭日志时省 allocator 扫描 | 正常/启用日志、32 项 FP8 graph、小图 ABBA 已通过 |
| `487b3889c` | 上述改动 + 三个低层 trace 范围 | 构建、变输入/下游消费、完整 70 层 C2 通过 |

反汇编保留在 `build-records/prepared-launch-a/`：旧 503 的 replay lambda 调用
`memmove` 复制绑定表；新 loop 按 `0xe8`（232 B 的 `synLaunchTensorInfo`）步长
只读写 8 B 地址字段。这证明生成代码改变，不能单凭它宣称模型加速。

新加 3 个 prepared binding CPU 测试覆盖 alias/静态元数据、HOST 地址/零尺寸，
以及错误绑定数量在修改前拒绝；与原 Storage/队列测试合计 11 项通过。

## 新鲜完整模型配对

冻结应用 `7eb4e41c0e9277da0bd11921dc0620fc7e865753`；该版只在已有普通应用上
增加计时后诊断，计算策略相同。MiMo-V2.6-Pro-RL，70 层、TP8、无 MTP、4K，
B1/B2 各 128 输出 token、2 轮。没有外置 executor 库。C1 完成后跑新鲜 A1。

| 指标 | 503 A1 | 3d C1 | 相对变化 |
|---|---:|---:|---:|
| B1 round0 resident TPS | 75.465 | 76.754 | +1.71% |
| B1 round1 resident TPS | 75.682 | 76.880 | +1.58% |
| B2 round0 聚合 resident tokens/s | 118.920 | 119.404 | +0.41% |
| B2 round1 聚合 resident tokens/s | 118.854 | 119.438 | +0.49% |

每臂 768 个 token、6 条输出序列完全一致；全部 8 rank 的 B1/B2 图使用新计划。
每份计划 284 命令、141 collective。一条命令可能包含 compute 和 collective，
因此不是 284 个设备算子，也不等于一次硬件 launch。

这是 fresh-process 配对，每臂两轮，不是同进程 ABBA。小幅变化不能直接当成
稳定收益保证。resident 区间保留正常逐步延迟，不按快慢筛选；包含 prefill 的
generate wall 另报：A1 B1 21.979/21.907、C1 22.101/22.099 tokens/s；B2 两臂
均约 23.8 聚合 tokens/s。**约 77 TPS 不是含 prefill 的端到端吞吐。**

重复文本用于速度与执行等价；重复输出不是独立语言质量门。
与此前上游约 49 TPS 对照相比，bridge 核心收益已成立；本轮增量较小。
**90 TPS 未达到，约 13.0 ms → 11.11 ms 的缺口尚未消除。**

最终 487 C2 同条件验收：B1 76.035/76.471 TPS，B2 聚合 117.652/119.204 tokens/s。
对新鲜 A1 的 B1 变化 +0.75/+1.04%，B2 −1.07/+0.29%；仍没有稳定新增模型
收益的结论。含 prefill 的 C2 generate wall：B1 21.963/21.987、B2 聚合
23.813/23.877 tokens/s。768 token 全部相同、8 rank 覆盖通过，90 门明确未过。

## 小图 ABBA：不挑最好数字

正常 `HPUGraph`，FP32/BF16 × inputs 1/8/32 × recipes 1/8/32，共 18 点；每点先
过 4 次变输入，再取 3 块 × 256 replay。脚本记录实际安装 wheel、native 命令数和
replay 覆盖。它测热静态图提交扩展性，不测模型 TPS 或 HBM 带宽。

- 503 → 3d，四进程 A/C/C/A：多数点变化 0–2%。几个表面上 10–12% 的点伴随
  对照臂显著波动，例如 BF16/32 recipe/1 input 对照 456.60/574.86 µs；不能当成
  可复现两位数收益。
- 3d → a444，另一组完整 A/C/C/A：变化约 −1.7% 到 +1.4%，未证明稳定新增收益。
  保留省重复工作的实现，但不为这项填毫秒级收益预算。

## Trace 已经定位到哪一层

C1 在计时后采 8 rank CPU trace，每 rank 63 次完整计划；分析后 55 次。B/E 配对
无缺失。下表为 CPU wall 诊断，包含 profiler 开销，不是设备时间；parent/child
inclusive 范围不能相加。

| rank | 函数入口→提交线程，µs | NativePlanSubmit，ms | parent Launch 合计，ms |
|---:|---:|---:|---:|
| 0 | 7.60 | 7.586 | 7.389 |
| 1 | 8.00 | 7.769 | 7.572 |
| 2 | 7.68 | 7.518 | 7.317 |
| 3 | 8.78 | 7.608 | 7.401 |
| 4 | 9.99 | 7.476 | 7.275 |
| 5 | 19.16 | 7.663 | 7.434 |
| 6 | 9.76 | 7.654 | 7.429 |
| 7 | 14.18 | 9.519 | 9.227 |

rank0 绑定预检中位约 90.9 µs。函数入口到 worker 包含 promise/入队开销，不能
严格命名为纯 wakeup；但量级明确不是 2 ms。

按 284 条命令的序号配对，rank7 相对 rank0 的内部累计延迟从第 0 条约 9 µs，
增长到第 63 条约 558 µs、第 127 条约 987 µs、末尾约 1.94 ms。
144 个有 compute child 的 parent 合计中位：rank0 4.348 ms、rank7 5.433 ms；
140 个没有 compute child 的 parent：2.949/3.782 ms。这只是 trace 结构分类，
不是按 recipe 身份给全部 collective 逐个定案。

因此这次较慢 rank 的额外延迟是在逐命令提交中累积，不是某一次排队停顿。
之前晚到 rank 曾是 1/6，不能据本次宣布物理卡7坏了。C1 的 `launch` 范围仍
包含 workspace、allocator 和 SDK；C2 新范围已经进一步分开三者。

C2 同样每 rank 63 次计划、取后 55 次，每次三个阶段各命中 144 次，无缺失事件。
CPU profiling 的范围增加了，不能把 C1/C2 CPU duration 的差异称为性能回退。

| rank | 内存预算/workspace，µs | 地址租约，µs | Synapse API 提交，µs | 整计划提交，µs |
|---:|---:|---:|---:|---:|
| 0 | 243.54 | 233.62 | 3382.03 | 9867.27 |
| 1 | 195.77 | 213.27 | 2718.43 | 8081.61 |
| 2 | 196.03 | 214.46 | 2761.30 | 8267.93 |
| 3 | 206.69 | 223.51 | 2855.66 | 8516.24 |
| 4 | 240.44 | 220.30 | 3317.80 | 10471.58 |
| 5 | 187.47 | 217.14 | 2665.76 | 8000.64 |
| 6 | 191.65 | 207.09 | 2680.55 | 7979.37 |
| 7 | 194.18 | 212.38 | 2710.23 | 8088.59 |

每格是每次计划内对应阶段合计的中位数。前三列只覆盖计算 recipe，通信路径的
地址/事件处理不在其中。计算侧 allocator/workspace 合计约 0.4–0.48 ms，不能
单独解释剩余约 2 ms；也不能据此跳过必须的内存预算、defrag 和 OOM 保护。
SDK API 返回耗时最大，但包含描述符处理、排队/锁及设备背压的可能性，不能全部
当作可删除的 CPU 税。C2 晚到 rank 转为 0/4，也不支持“物理卡7有问题”的判断。

## CPU/NUMA 与退出

C2 加载阶段只读 `/proc/<worker>/task/*/status`：rank0–3 的所有线程允许
`0-31,64-95`，rank4–7 为 `32-63,96-127`，符合 4 卡/NUMA 的本地绑定。
这是 NUMA 本地性，不是每 rank 独占核心；约 266 个线程多半在等待，不能仅凭
线程总数认定 CPU 争用。当前 session 和 user 父 cgroup 均 `cpu.max=max 100000`，
`nr_throttled=0`；只说明这份可读快照未发现 quota 节流，不排除其他等待。

A1、C1 模型验收均先 PASS，退出阶段仍有 8 worker 超时后 SIGTERM、EngineCore
强制收尾和 device 剩余引用警告。两个 arm 都有；更早未改上游也有 EngineCore
强制收尾，但没有相同 device 引用警告。自有进程已回收，设备释放；不能由此
宣称优雅退出或分布式在途生命周期已完全验收。此债务继续保留。

## 接下来做什么，为什么

1. 下一轮先对齐有界的 host/设备时间线，把计算/通信交接、SDK 提交/返回和实际
   引擎执行对应起来；区分 host 延迟必要提交和设备背压。CPU 记录本身不能回答
   所有引擎为何空闲。当前证据不支持优先做复杂的全图物理地址缓存。
2. 对可证明静态、顺序区域做编译依赖和执行资源租约，减少逐命令的 map、holder、
   callback、重复管理；共用正常 launcher helper。terminal 回收不等于把所有
   Tensor 可见完成推迟到末尾：partial/external readiness 与输入 WAR 必须保留。
3. 再处理公开 wrapper 的输入重绑/拷贝与异步 replay。当前显式新输入 adapter、
   async、动态 shape、RNG/训练、多个 recipe stream、DMA 等仍走已有回退；不能
   把快路径的窄覆盖说成整个 PyTorch 图 API 已被替换。
4. 生命周期列为同等交付项：在途 reset/销毁、communicator 释放、异常/混合
   rank、内存压力和多图共存。暂不删除旧 executor/graph；达到性能与语义门后
   从 serving 生产依赖移除，历史对照归档，不再长期运行三套执行核心。

输入、通信算法和 kernel 融合是不同问题。此轮没有改归约精度、吞掉正常 Lazy
写入、永久 pin allocator，也没有把 GIL 局部释放到持锁的 `future.get()` 处。
通用方向及反例见 [OPTIMIZATION-DIRECTIONS](OPTIMIZATION-DIRECTIONS-20261004.zh-CN.md)。

## 本地资产与失败留存

用户“完成后整理资产，资产和成果留在本地”的要求持续执行。
根目录：`/home/rate/llm/Gaudi-LLM/gaudi-kernels/artifacts/builds/bridge-integration-20261004/`。
源码 bundle、冻结 source export、构建命令/返回、wheel/SHA、CPU 反汇编、单卡/TP2/
整模型原始结果、trace 和失败都在此目录；早期资产没有覆盖或删除。

CPU trace 分析器加入线程/时间索引；同一 rank7 的 63 条 command 记录与旧分析
逐字段相同，分析用时约 6.6 秒；4 个有意义 CPU 测试覆盖线程、嵌套、缺失与
非包含范围。诊断工具不进入模型热路径。

本轮保留失败：旧 profiler probe 的 import marker 检查位置错误；CPU profiler
pybind 缺 STL converter；以及 stages probe 初次导出选错 supervisor 源码路径。
后者未取得设备，已纠正后跑新 case；不计为设备或模型失败。暂停前中止的构建
仍标明用户暂停，不伪装为编译器错误。主 Graph 测试套件缺独立
`intel_transformer_engine` 的历史收集问题仍未解决，不声称通过全套。

wheel SHA256（另有 `final-wheel/wheels-sha256.json`）：

| commit | SHA256 |
|---|---|
| 3d | `552897df673e9ea854c7fe3e7152c46965fea831b8bd6c686a6802daab82a83f` |
| a444 | `8a5438935a1caedd1557bd42612b2676efc918241883f2730c547fa57afd2a7d` |
| 487 | `75fa9a94bf391be5e3773261ac07867fdc6278f311e40cf0d64993b8f6d0cb21` |

源码 HEAD 后续文档提交不改变这些已构建 wheel 的版本。

最后全部自有进程已回收，8 卡均为 768 MiB/0% 的空闲基线；C2 supervisor 返回 0，
`remaining_owned_processes=[]`。退出警告债务仍保留，设备释放不等于优雅生命周期过门。
