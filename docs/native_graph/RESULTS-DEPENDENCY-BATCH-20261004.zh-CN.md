# 依赖批处理与设备关键路径检查点

2026-10-04 续行，远端部分日志为 10-05。目标是普通 PyTorch 应用仅替换 bridge，
达到 90–100 resident TPS；目前尚未达到。权重、kernel、精度、通信算法均不变。

## 实现和边界

编译候选 `0b236aa72`：多个输入先完成原有 future 检查，再一次取得 producer-map
锁、保存 event owners，并在释放 map 锁后去重。同一 producer 的多个输入只等待
一次。不同 producer 与跨 stream 的真实依赖保留；单输入保留原路径。

`event::stream_wait_event` 把已有的同 stream 判断提前到 GIL/互斥锁操作之前。
同 stream 原本就不提交 SDK wait，这项不是新消除一种硬件依赖。

没有缓存裸物理地址，没有删除输入 WAR、partial/external readiness、allocator
租约或异步保活。仍逐条 recipe/HCCL 提交，尚未实现全图依赖索引或资源完成 frame。

后续源码 `a480db0b0` 增加真实 producer/consumer graph：同一图两个输出共享完成
事件，以及另一 stream 的独立 producer。测试脚本版本与已编译 wheel 分开记录。

## 选卡缺陷与重新测量

旧单卡 supervisor 只设置 `HABANA_VISIBLE_MODULES=5`，bridge 的实际选卡代码还读取
`HLS_MODULE_ID`。一次硬件采集观察到实际 accel3/PCI43/module0，CPU 却在 NUMA1。
原始结果保留，不能继续用作 module5/NUMA 本地证据。

修复后在 Torch autoload 前同时设置两个变量。探针读取实际 `/proc/self/fd` 的 accel
设备、sysfs PCI/NUMA/local CPU，并核对预期 PCI 与进程 affinity。重新采集明确为
accel6/PCIcd/module5/NUMA1。完整 TP2/模型 worker 原本已有 `HLS_MODULE_ID`，不受影响。

## 已完成验收

- 32 线程、777 个对象构建通过，616.925 秒；独立 venv，未修改共享 Torch/SDK。
- 11 项 C++ Storage/队列/prepared bindings 测试通过。
- 真实 HPU stream/view/churn/reset 与共享/独立 producer 共 7 情形、112 个最终输出
  检查通过。reset 仅末轮一次无显式同步释放，不冒充完整分布式在途压力测试。
- FP32/BF16 变化输入、普通 wrapper M1/2/8/64/512、显式新输入 adapter 与非法输入
  验证通过。新输入 adapter 仍明确使用回退。
- vendor FP8 graph 32 项通过；第一次漏传 lazy/dut 参数导致 32 项跳过，原失败资产
  保留，不计为通过。
- TP2 的 FP32/BF16 all-reduce、all-gather、混合 native/V3 链通过。其短通信链时间
  没有明显新增收益；与历史结果非 ABBA，不据此填收益预算。

## 正确选卡后的微基准

普通 HPUGraph，FP32/BF16 × 输入数 1/8/32 × recipe 数 1/8/32，18 点，A/C/C/A。
A 为 `487b3889c`，C 为 `0b236aa72`。每点 4 次变化输入、3 块 × 256 replay，全部
通过实际 PCI、本地 CPU、输出和 native 覆盖检查。

| 范围 | FP32 延迟降低 | BF16 延迟降低 |
|---|---:|---:|
| 1 输入、1/8/32 recipe | 1.41–2.37% | 2.83–3.44% |
| 8 输入、8/32 recipe | 7.17–8.43% | 8.28–8.44% |
| 32 输入、8/32 recipe | 10.79–11.24% | 8.69–9.21% |

这是热静态 add 图的提交扩展性，不是 HBM 测量，更不能直接乘到模型 TPS。

## 完整模型与下一步

新鲜 A/C 完整 70 层模型已经通过执行等价验收。冻结应用 `3f569b3`，
A 使用 wheel `487b3889c`，C 使用 `0b236aa72`；4K、无 MTP、TP8、B1/B2、128 输出
token、每点两轮。

| 指标 | A | C | TPS 相对变化 |
|---|---:|---:|---:|
| B1 round0 | 75.089 | 76.349 | +1.68% |
| B1 round1 | 75.680 | 76.594 | +1.21% |
| B2 round0，聚合 | 120.012 | 118.821 | −0.99% |
| B2 round1，聚合 | 119.695 | 118.532 | −0.97% |

两臂 768 token、6 条序列逐一完全一致；8 rank 的两份计划均 ready，284 命令/
141 collective，无外置执行器。实际设备 FD/PCI 与全部线程 CPU 本地性验收通过。
这是一个 fresh-process 对，每臂两轮，不是同进程 ABBA。不能宣称 B2 获得收益；
90 TPS 明确未达到。B1 的约 1% 增量仍不足以解决约 2 ms 的差距。

设备 profiler 仅在性能请求之后的额外 decode 启动，不采加载/
prefill/capture，不从诊断区间计算 TPS。各 rank 在计时后记录实际 PCI/CPU 位置。

设备时间线用于区分 CPU 供给不及时、必要的计算/通信依赖与设备背压。按各引擎
执行区间取 union，不把 24 个 TPC 的时间相加，也不把不同引擎的重叠算两遍。
缺少某类执行事件不代表该引擎确实空闲。CPU SDK 调用时间不是可删除开销的证明。

A 的 rank0 steady 采样窗口中位为 21.732 ms，TPC 3.534 ms、MME 2.234 ms、二者
union 5.705 ms；与普通请求约 13.2 ms 不同，证明硬件 profiler 扰动明显。全部
8 rank 的 compute union 约 5.70–5.75 ms。NIC 只有元数据、没有导出执行事件，
因此不能据此声称网络传输为零，也不能把其余约 15 ms 叫全部引擎空闲。使用
rank0 的采样窗口并按整数 `baseTimeNanoseconds` 对齐其余 rank，避免逐 rank 的
host 返回点或 epoch float 误差改变窗口。

纯通信候选 `b81fa4a45` 删去没有 compute recipe 的计算流等待与额外 compute
资源 holder/counter。完整模型 B1 75.933/75.579、B2 聚合 118.715/118.712；较 0b
对照无可确认新增收益，768 token 与覆盖完全一致。数值/覆盖 PASS 不等于生命周期
安全：后续源码复审发现该版本空 callback 漏掉了 Tensor Storage 保活。

当 `PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS=1` 时，collective helper 依赖上层 callback
持有 Tensor，其局部 in/out Tensor 为空。地址锁只防 defrag，不持有 Storage。
`b81` 不作为可发布候选；原始结果和纠正记录保留。修复 `d8f717434` 在 done callback
中保留 `PreparedTensorBindings`，实际通信的地址租约与 communicator 保持原管理。
删除的仅是没有计算工作的额外 compute holder/counter。

加强的 TP2 `collective-lease-c1` 和旧 0b 对照均通过：128 个数值检查、图外消费、
连续两次通信；额外让 rank1 在 Gloo 屏障等待，rank0 提交后 reset 图并删除输出
Tensor，仅用弱 Storage 句柄观察。FP32/BF16 输出 Storage 在通信被对端阻塞期间
存活，完成后已释放。这没有覆盖所有 communicator 释放、异常和内存压力情形。
修复后的完整模型 `collective-lease-model-c1` 已完成：B1 75.626/76.209 TPS，B2 聚合
118.618/118.855 tokens/s；与 0b 的四组输出 768 token 全部一致，8 rank 覆盖通过，
全部 worker 实际 PCI/CPU 本地性通过。相对 0b，B1 −0.95/−0.50%，B2 −0.17/+0.27%，
一个 fresh-process 对不支持稳定增量收益。修复改变保活，未改变数值算法；90 未达到。

b81 的轻量诊断没有启用任何 profiler：rank0 `runner.sample_tokens` 中位 12.797 ms，
`model.forward` 8.345 ms，输入准备 0.806 ms、sampling parent 0.554 ms、输出 D2H
2.439 ms。它们是嵌套 host wall，不能相加；D2H 包含等待未完成的设备工作。
HCCL/API 小图收益也没有自动出现在整模，不再用微调金额拼出 90 TPS。

后续按关键路径证据决定是否做编译依赖和区域资源回收。不能把 compute/通信盲目
合到一个物理 stream：partial Tensor 可见完成可能早于 recipe 完成，合流会损失
有效 overlap。旧执行器仍保留对照，退出引用警告与分布式生命周期债务尚未解决。

## 下一项结构性实验

旧 `e1_replay.cpp` 把命令放到固定 serial stream，seal 明确拒绝 partial/external
事件。当前 bridge 对这类事件有完整处理，不能拿旧录制器的窄假设推广到所有图。
约 76.5 TPS 对应 13.07 ms，90/100 TPS 分别需要 11.11/10.00 ms，即省约 1.96/3.07 ms。
这是目标缺口，不是已发现可无条件删除的开销。

下一实验复用已有 `NativeExecutionPlan`、launcher、ProcessGroup 与 stream helper：

1. 从冻结 Tensor/collective 描述符生成每条命令的读写集合，区分真实写输出与输入
   WAR 登记；先盘点实际模型的 external-event 数量。没有证明的链继续原路径。
2. 只选择相邻命令存在必需 RAW 边、单 recipe stream、无 partial/external 提前
   readiness 的区域。原桥的 Storage/alias 边界必须一致，不根据 Tensor 数值猜依赖。
3. 区域内将通信临时提交到相同物理 stream；入口等待原 communicator stream 的
   既有工作，出口把完成关系接回原 communicator FIFO。保留普通队列顺序，禁止
   全局改写 ProcessGroup stream 或影响图外/其他线程的 collective。
4. 原 Tensor/地址租约、输入 WAR 和完成语义照常执行。先独立评估 stream 交接，
   再考虑区域资源 frame，避免一次修改两种生命周期机制导致不可归因。
5. 最小链需同时测试正常/混合 fallback rank、alias、跨流图外读写、连续 replay、
   peer-blocked reset。先取得对当前 0b/d8 的同口径收益，再上 70 层；确认 M>1/
   prefill 的 overlap 不退化。变化输出/覆盖门和真实计时分别报，不以 profile TPS 验收。

这项还未实现或实测。既有 90 TPS 参考为目标提供依据；普通应用 90–100 TPS 仍需
证明，不能凭两套执行器有相同 SDK 调用就保证结果。

## 资产

用户“资产和成果留本地”的要求持续执行。目录：
`/home/rate/llm/Gaudi-LLM/gaudi-kernels/artifacts/builds/bridge-integration-20261004/`。
`batch-matrix-local-abba.json`、`probe-device-identity-correction.json`、TP2、全部原始
探针/构建记录、源码 bundle 与 wheel 均保留。代码已提交本地 Git，不能等同于已推送。

wheel：`1.24.1+git0b236aa72`；SHA256：
`4db97d3c4ae3d5d5757d835317035cdbe680d27189a7201a2f7132ccfe0796c2`。

最后候选 wheel `1.24.1+gitd8f717434`，SHA256：
`17ff00d1d00ceef6205e80c4dd118979f13d4518978991a9fa90015fbbb991bd`。
冻结应用/验证脚本 `385b8e7`。新增 Chrome 窗口 CLI 与整数时钟对齐测试已经通过；
现有 TEF 路径保留。CPU 反汇编是 launcher 路径证据，计算 kernel ISA 未改。

最后 supervisor 返回 0、`remaining_owned_processes=[]`。即时 postflight 显示旧的
93898 MiB/卡计数，后续只读核查回到 768 MiB/卡、0% 和无本用户设备 FD；未 reset
设备。完整模型退出仍有 device 引用警告、worker 超时 SIGTERM 和 EngineCore 强制
收尾。设备释放不等于优雅退出过门。
