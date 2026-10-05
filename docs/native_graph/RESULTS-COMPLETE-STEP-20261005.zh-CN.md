# 完整 step 图：实现、真实收益与反证

## 裁决

完整 decode 设备计算已经运行在现有 bridge 的 NativeExecutionPlan 内；没有
加入新执行器，没有切换内核或精度。70 层 TP8 B1 从约 76 TPS 到约 78 TPS，
三次同进程 ABBA 均有小幅收益。**尚未达到 90 TPS，也没有收回先前的全部
2.8–2.9ms 差距。** 两层省出的约 1.1–1.2ms 不能线性外推到完整模型。

本轮性能实现位于 gaudi-kernels 的 vLLM 前端适配器，复用已编译 bridge
`77d8490270`，没有新编译一版 bridge C++ 或改变 NativeExecutionPlan 提交逻辑。
这必须与“bridge 逐命令资源处理已优化”区别开。

## 改了什么

- `runtime.decode_graph=step` 声明 forward → 选行 → logits → 原 sampler 的
  完整计算范围。内层 forward 图 bypass，避免嵌套捕获。
- 同步 decode 按角色/形状/dtype/device 复用输入 Tensor；普通 `copy_` 继续
  负责转换、依赖和在途 Tensor 所有权，没有缓存裸物理地址。
- 第一合同为同步、无约束、无 penalty、无 logprobs 的 greedy。MTP、异步、
  混合 prefill/decode、LoRA、KV transfer、自定义 processor 等回退原路径。
- 请求状态提交、scheduler、CPU 输出处理仍由原 vLLM runner 负责。
- 算术策略变化时清理完整图；缓存键使用策略 fingerprint，避免对象 ID 复用。
  图闭包弱引用 runner，退出时先 drain 再释放缓存，避免新增 runner 引用环。
- 默认仍是 `forward`。没有宣布旧 SDK 执行器已可删除。

最终性能相关源码为 gaudi-kernels `941c3e5`；该版本重新通过两层 B1/B2、
完整 logits 页边界检查。70 层性能/质量对应冻结版本见下表与本地 source-identity。
末尾生命周期清理没有重新跑 70 层；不能把它说成已完成整模退出验收。

## 无 profiler 的实际结果

同一 LLM 实例、同一模型权重/内核/精度，forward / step / step / forward。
每臂预热完整生成范围，4K 输入，70 层输出 160 token。时间采用完全驻留 decode
窗口的连续 coordinator wall，含 engine.step 之间的间隙。B2 是聚合吞吐。

| 实验 | batch | forward ms/步 | step ms/步 | 节省 | TPS / 聚合 tokens/s |
|---|---:|---:|---:|---:|---:|
| model70-b (`7b144ac`) | 1 | 13.034 | 12.797 | 0.237ms | 76.72 → 78.15 |
| model70-b | 2 | 16.917 | 16.449 | 0.468ms | 118.23 → 121.59 |
| profile70-d 的计时阶段 (`0fbc847`) | 1 | 13.105 | 12.782 | 0.323ms | 76.31 → 78.24 |
| sdk70-f 的计时阶段 (`f76358b`) | 1 | 13.073 | 12.833 | 0.240ms | 76.49 → 77.92 |

所有 ABBA 输出序列一致。性能请求没有 profiler；名称带 profile 的实验是在
计时结束后另行采集。这里的窗口从 emitted_min≥16 开始；先前旧 SDK 对照用
32≤emitted<160，不能将不同运行的数字拼成严格的三臂因果对照。

以当前约 12.8ms 为起点，到 90 TPS 还需约 1.7ms，到 100 TPS 还需约 2.8ms。
这只是所需差额，不是已经找到、保证能获得的收益。

## 覆盖与数值

- 8 rank 均实际 ready/replay，没有失败或外部执行器库。完整 70 层图为
  145 个 compute recipe、142 次 collective，共 286 条命令，含混合命令。
- 两层图 8/6，对比旧完整 SDK 10/6。图扩大后末段编译合并，不能机械要求
  recipe 个数与旧链相同。
- 两层 B1/B2：首个 decode、128-token 页边界前后，输入、位置和完整词表
  logits 均逐位一致；最后生命周期修订后重新通过。
- 70 层 B1：同样三个检查点的完整 logits 逐位一致。
- 70 层 B2：完整 ABBA 生成一致，尚未补同等完整 logits 检查。
- 本轮未验收 MTP、随机/约束采样、长上下文、prefill 性能或广泛语言质量。

## 为什么两层省 1.2ms，70 层只省约 0.25ms

另跑无设备 profiler 的有界 host 观察，rank0 后 32 个 decode 的中位数：

| 嵌套范围 | forward | step |
|---|---:|---:|
| 输入准备 | 0.787ms | 0.629ms |
| forward + logits 的调用范围 | 8.695ms | 8.040ms |
| sampler 调用范围 | 0.569ms | 0.0013ms |
| 输出 `.cpu()` | 2.530ms | 3.671ms |
| sample_tokens 总范围 | 12.798ms | 12.508ms |

CPU 工作确实减少，但末尾等待设备增加约 1.14ms。`.cpu()` 包含设备完成等待，
不是 4 字节复制花了 3.67ms；表内 parent/child 也不能相加。
因此“把完整范围迁入图就能拿回全部缺口”被本轮实验否定了。完整图是必要
基础设施和已有小幅收益，不是目前已证明的主要性能根因解决方案。

## profiler 反证

本轮保存了 8 rank × 两路径的联合 trace，以及不经过 Torch profiler 回调的
直接 Synapse TEF 采集。直采复用已有 exporter，仅生成诊断库，不记录或重放模型。

rank0 示例（均为诊断 sample_tokens 范围，不能作为 TPS）：

| 观察方式 | forward | step |
|---|---:|---:|
| 普通轻量 host 观察 | 12.80ms | 12.51ms |
| Torch + bridge + HPU profiler | 19.45ms | 15.60ms |
| 直接 Synapse profile | 16.68ms | 13.97ms |

两类 profiler 都改变了差值，直采也不是零开销。不能从 trace 中数毫秒的差值
宣布优化收益。两路径观测到的 TPC/MME 时间并集都约 5.7ms，未见算术主体变慢；
不同的是计算之间的空隙。但这仍不足以把每个空隙定性为可删除的 host 开销。
NIC 轨道没有有效 span，不能把未覆盖时间称作“所有引擎空闲”。分析器已显式
限定口径，按 interval union 去重，不累加 24 个 TPC 的并行时间。

## 下一步：把提交晚到与设备固有执行成本分开

优先复用已有门控实验办法做同一捕获产物的最小对照：

1. 正常提交；以及在设备头部加有界门控，让 host 有机会提前提交同一条命令链。
   计时仅取门控之后的设备事件区间，门控不用于生产或宣称 TPS。
2. 必须证明所有待测命令在门控结束前已经提交；若 SDK 背压导致未提交完，这个
   样本不能证明“已排除 host 饥饿”。保留失败与提交/门控时间。
3. 若提前提交能恢复旧链执行时间，再在现有 NativeExecutionPlan 中做完整计划
   的资源持有和依赖发布：入口处理跨计划 RAW/WAR/WAW，区内复用已证明的同流
   依赖，尾部统一发布完成、释放资源。继续复用 allocator/communicator。
4. 若提前提交没有收益，则查设备端命令/事件语义及 recipe 边界。不能先删除
   生命周期保护，或再把 CPU enqueue 降幅当成 token 收益。

源码需要处理的具体边界是 `NativeExecutionPlan::replay` 仍逐命令进入
`RecipeLauncher::Launch`，随后分别做 producer/readiness、地址租约、resource
holder 和完成登记。这是明确的实现对象，**其可获收益尚未定量证明**。
异常提交前缀、图销毁、别名写入和并发访问必须有验收，不能直接复制旧 recorder
对整个进程固定地址/串行调用的假设。

## 资产与退出

所有代码用 Git；原始结果、8 rank trace、完整 logits、冻结源码、诊断 .so、
manifest、源码身份与校验和留本地：
`gaudi-kernels/artifacts/builds/bridge-step-20261005/`。

诊断 helper 以 `make -j32` 构建。没有改动其他人的权重/资产，没有设备 reset。
现有 vLLM 退出仍可能强制结束 EngineCore/worker 并报 DeviceReference 警告；本轮
不能宣布一般退出问题已经解决。每次实验的进程组清理与设备空闲检查单独保存。
