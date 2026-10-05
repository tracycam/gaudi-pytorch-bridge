# Bridge 内部值得继续做的通用优化

2026-10-04，续中场诊断。用户要求从第一性原理检查 bridge 自身及两套原生
执行器中可合入的能力，要求根本、通用、清晰、可靠。本轮继续源码分析；没有
恢复设备实验，没有把下列候选的收益写成实测。
随后用户要求继续，已实现和验收部分 prepared launch、关闭日志的参数构造及 CPU
诊断，见 [续行结果](RESULTS-PREPARED-LAUNCH-20261004.zh-CN.md)。以下其余项仍是候选。

审计源码：bridge `72b41f885`，kernel/历史执行器仓库 `cdbbde9`。
已验证模型性能仍为约 49 → 75 TPS，见 [中场记录](MIDPOINT-20261004.zh-CN.md)。

## 第一性原理：重放应重复什么

静态图已经知道算什么、Tensor 布局、资源依赖和哪些对象必须存活。重放变化的
通常只有输入内容，部分应用还有输入分配或形状桶。因此，不变信息应在 capture/
prepare 时处理；运行时只处理变化和真实的外部依赖。内部每个算子不必再次作为
一次全新、未知的框架任务进行管理。

设备执行、host 提交、通信可能重叠。整步取决于依赖图的关键路径，不能把 CPU
函数时间、设备时间、D2H 等待相加。少做 CPU 工作只有在它延迟必要提交、竞争资源
或位于返回关键路径上时，才会降低步延迟。

理想的 bridge 重放路径接近：验证/更新公开输入绑定并接上图外 producer；取得
本次地址和 workspace 租约；按编译好的依赖计划提交计算/通信/拷贝；发布图外
完成依赖并统一回收本次资源。这仍可能有多次 SDK 调用，不要求先有单次硬件提交。

## 三套代码中真正应当合并的能力

| 能力 | Bridge 当前快路径 | 旧 executor | 外置 native_graph | 应吸收的部分 |
|---|---|---|---|---|
| 正式捕获、PyTorch Tensor 语义 | 有 | 拦截调用 | 显式构图 / 受限拦截 | 保留 bridge 正式导出 |
| 紧凑 SDK 提交循环 | 循环内仍调用通用 RecipeLauncher | 有 | 有 | 编译好的命令、预绑定表 |
| stream 规划 | 保留 compute / communicator stream | 扁平到 serial stream | 自有单 stream | 对可证明的顺序区域减少交接 |
| 资源有效期 | 逐 recipe/collective holder、锁、回收 | 原捕获资源与专用保活 | pool、owner 契约、flight ticket | 每次执行的正式租约与完成回收 |
| 地址稳定性 | 每次 allocator 翻译/锁定 | 重用记录地址 | 私有分配固定、外部地址借用 | allocator 内可审计的稳定地址契约 |
| 主机输入 | wrapper 拷贝、Lazy StepMarker | pinned staging、角色补丁 | 每 flight staging | 通用拷贝计划与异步生命周期 |

旧 `native_step.inc` 理解 token、position、slot、128-token 页面，不能进入 bridge。
外置 graph 的独立 ABI、独立 stream/allocator 也不应整包复制。要合入的是所有权、
预编译命令和完成语义，接上现有 allocator、stream、ProcessGroup 和 Graph API。

## 1. 完成 prepared submission，而不止准备 Tensor 列表

**优先级高。** 503 只前移了 Tensor 引用、依赖地址列表和通信描述的准备。
[NativeExecutionPlan::replay](../../backend/habana_device/NativeExecutionPlan.cpp)
仍逐条调用 `RecipeLauncher::Launch`，后者进入
[graph::launch](../../backend/synapse_helpers/graph.cpp)。低层仍重复：

- 遍历绑定表、修正静态尺寸、构造地址 vector 和 HOST_TO_DEVICE 索引 map。
- 查询未物化地址的内存需求，检查 workspace，取得 workspace mutex。
- 锁定/翻译地址，再遍历写回 launch table，组装 event handle 数组。

把静态修正、tensor ID/offset、别名组、workspace 要求、地址和 event 索引编译为
不可变提交描述。本次只解析必要地址、填更新槽；共用现有低层 helper 和错误处理，
不复制整份 launcher。

**正确性：** 持有 Storage 不等于物理地址不动，allocator 可物化/defrag。只有
有效地址租约下才能使用物理绑定，不能直接缓存裸指针。

最小实验：扫描 recipe 数与每 recipe 绑定数，测 host 提交、分配、锁等待和设备
完成；验证输入变化、alias/offset、回收及内存压力。已测“小图约 2%”只是 503 的
Tensor 列表优化，完整 prepared submission 尚无收益数据。

## 2. 把内部依赖编译成计划，减少全局地址事件管理

**结构性机会最大之一，与第 3 项一起设计。** 当前每条 recipe 查 producer event、
登记输出；HCCL 在 communicator stream 准备依赖并登记完成。见
[RecipeLauncher](../../backend/kernel/hpu_habana_cache.cpp)、
[collective](../../habana_kernels/hccl_kernels.cpp)、
[stream_event_manager](../../backend/synapse_helpers/stream_event_manager.cpp)。

capture 时按存储范围建立依赖，区分内部边和图外边界：同 stream 内部有序边可
省反复的全局 map/排序/登记；跨 stream 真实依赖保留 event，按编译索引执行；
图外输入、输出及被修改输入在 entry/exit 接上现有事件系统。

严格 compute→collective→compute 顺序区域可评估同一物理 stream；真正独立的
分支保留多 stream。所有 rank 的 collective 顺序必须一致，不能全局强制串行。

两个源码反例必须记住：

- [event::stream_wait_event](../../backend/synapse_helpers/event.cpp) 已跳过相同
  stream 的等待，不能把每次依赖查询都算成硬件 wait。
- [event_handle_cache](../../backend/synapse_helpers/event_handle_cache.cpp) 已复用
  SDK event handle。可减少的是 wrapper、producer map、record、GC 排队和完成处理，
  不是假设每次都 `synEventCreate`。

**正确性：** RAW/WAR/WAW、alias 范围、图外访问、partial/external event、SFG、
多个 communicator、并发 replay 都需要明确语义。先接纳可证明区域，其余保留
现有处理。命令有序不等于跨 stream 自动有序。

最小实验分开比较“保留 stream、编译依赖”和“顺序区域合流”，包含 TP2
producer→collective→consumer 与独立分支两类图；验证普通 PyTorch 在图前后和
另一 stream 读写输入输出。不能以失去有效 overlap 换取较少事件数。

进一步核对源码后，需要保留两种容易遗漏的行为。external/partial event 可能在
整条 recipe 结束前就允许通信消费某个 Tensor；把通信搬到同一 stream、等整条
recipe 执行完，可能丢掉这段真实 overlap。现有 producer 表还登记输入的使用结束，
用于防止别的 stream 提前覆写它，不能只处理输出写完的 RAW 依赖。

## 3. 每次执行一个资源租约，合并回收与 workspace 管理

**是高效重放的基础。** 每条命令目前都有 holder、active recipe 计数、callback；
[stream::register_pending_event / gc_thread_proc](../../backend/synapse_helpers/stream.cpp)
再逐事件回收。

借鉴 flight，在 bridge 内用一个执行实例保留本次 Storage、recipe、communicator、
地址锁、workspace、staging、事件和失败状态。只用于保活的逐 recipe callback
可合并为区域/执行实例完成回调。单 stream 可用 terminal event，多 stream 必须
join；不能只等 compute 完成就释放仍被通信使用的资源。

workspace 在 prepare 时预定容量。顺序 recipe 可按最大需求复用；并行 recipe/
执行实例必须分配互不冲突或受依赖保护的 workspace。
[workspace_alloc](../../backend/synapse_helpers/device_memory.cpp) 已复用足够大的
缓冲，并非每个 launch 都分配；可省的是重复管理和增长/回收等待。

**代价：** 资源持有更久可能增加显存峰值、限制 defrag，多 bucket 可能积累保留
空间。只读权重不复制，私有中间内存有预算和释放策略，不永久关闭 allocator 安全。
原 active-recipe 背压与 profiler 语义也要适配，不能简单少减几次计数。

最小实验：连续/并发 replay、跨 stream 消费、在途 reset/销毁、中途异常、内存
压力、多 graph 共存；同时记录显存高水位与对象/事件数量。

资源统一在区域末尾释放，不意味着所有可见 Tensor 都只能在末尾发布完成。
输入最后一次被读、某个输出可以被图外消费的时间，可能早于区域结束。保活回收
与对外依赖发布应分别描述，才能保留现有输入拷贝和通信 overlap。

## 4. 输入绑定与拷贝计划：通用性最大的缺口之一

显式 `replay_with_inputs` / V3 仍回退；replay 先执行 StepMarker。
[graphs.py](../../python_packages/habana_frameworks/torch/hpu/graphs.py) 每次递归
计算输入 hash、展开结构，再逐 Tensor `copy_`。

应在捕获时保存结构和 slot 映射，分别处理：

| 情况 | 处理 |
|---|---|
| 相同 Storage、offset、shape/stride/dtype 的精确同一数据视图 | 验证依赖后可省物理拷贝；不能只比 Python 对象身份 |
| 新分配、相同布局和 alias 契约 | 更新 slot；图会修改输入或要求缓存隔离时保留 staging copy |
| shape/布局/非 Tensor 控制值变化 | 选另一 bucket、重新捕获或回退 |

小 H2D/D2D 可编译成批量拷贝计划，复用 pinned staging，完成前不能改写 host 源。
旧 executor 的 staging 可借鉴，token/页表规则仍在上层。这服务 batch、prefill、
verify 及其他模型，不需要 bridge 认识这些业务标签。

不能删除 StepMarker 而漏掉 Lazy 写入；不能按内容“似乎没变”跳过更新。
数据 version counter 也不是纯绑定版本号。验收新分配、精确/非精确 alias、
in-place、pinned/non-pinned、0D scalar、容器变化和动态 shape，测整个公开 wrapper。

## 5. 异步提交与有界并发实例

[SubmitCollectivePlan](../../habana_kernels/hccl_kernels.cpp) 整图入队一次后仍
`future.get()` 等 SDK 提交完；`async=True` 则回退。可以借第 3 项的执行实例
保留资源，提前发布输出的提交/完成依赖，沿已有 Graph 异步 API 与 HCCL 队列实现。
slot 满才背压，异常经现有同步点回传，不另起线程池或第二套业务 ticket API。

后续普通算子、另一 stream、`.cpu()`、reset 必须正确等本次执行。Python 提前
返回不等于 TPS 增长；自回归下一 token 有真实依赖，只有重叠独立工作或避免提交
空洞才省整步。多请求和 prefill pipeline 可能更受益，待测。

另一个可独立验证点：`hpu/csrc/bindings.cpp` 的 replay 入口没有整体释放 GIL，
整计划 `future.get()` 也没局部释放，底层部分函数已有 AutoNoGIL。审计 C++ 调用链
与 Python 回调/析构后可在合适范围释放。它影响同一进程的 Python 线程，不能
声称它直接解释 8 个独立 rank 的差距。

尤其不能直接在持有 HPUGraph/plan mutex 的 `future.get()` 附近局部释放 GIL：
另一个 Python 线程可能持 GIL 等同一 mutex，而当前线程仍持 mutex 等重新取得
GIL，形成死锁。应先审核全部入口的锁/GIL 顺序、fallback Python 回调与析构；
尚未实现，不把局部 AutoNoGIL 当作现成优化。

## 6. 压缩我们自己的重复校验

[HPUGraph::replay](../../backend/habana_device/HPUGraph.cpp) 遍历子图边界，逐槽
调用带 mutex 的 matches_boundary；内部又遍历所有 lease。相邻子图会重复出现
中间 Tensor。

把描述扁平化，按 Storage+布局去重，持一次 plan lock 批量检查，同时保留精确
slot→lease 映射。只有能证明 graph 私有、外部不可重绑的中间对象才省重复检查。
不能把“图里某个合法 Tensor”当作该 slot，也不能在缺少失效机制时删除所有检查。
此项属于 prepared plan 整理，不另建缓存系统；是否是热点待测。

## 7. 沿已有编译链减少分图、冗余布局转换和算子边界

runtime 变短不会自动减少 recipe 内的算子或 HBM 中间写回。编译层应保留更大
区域、传播布局、选择已有融合算子。Dynamo 链已经有 attention/allreduce/view
fusion、reinplace、partition 合并和 collective 重排，见
[passes.py](../../python_packages/habana_frameworks/torch/dynamo/compile_backend/passes.py)。

先查目标应用走哪条编译链、为何某个具体边界未融合；当前 Lazy 模型不一定经过
这些 Dynamo pass。补正规 lowering/alias/meta，复用 Synapse 与 kernel 库，
不能在执行器按 MiMo 层名打补丁。

同组且语义兼容的独立通信可融合/提早提交；跨层有消费依赖的 141 次通信不能
机械合成一次。普通 replay 不改归约算法与精度；编译分块改变累加顺序时采用
正常 FP32 误差验收，不追求 FP64 或无条件逐位一致。

长期让 Lazy 导出和 compile backend 产物进入同一 bridge 执行计划，才是复用
基础设施。当前快路径只接了已验证 Lazy HPUGraph，不代表覆盖了所有前端。

## 8. 小而明确的工程整理

- 关闭日志时避免预先构造参数：`graph::launch` 的 `locked_addresses` vector
  在 TOWL 调用外构造，传给 `log_dev_mem_stats` 的额外内存扫描也在其函数内
  disabled 判断之前发生。让参数构造受启用条件保护。
- 静态 producer/read 地址提前去重，减少每次 `add_producer` 的 sort/unique。
- 统计/诊断使用有界记录，默认避免高频格式化和分配；执行 frame 复用现有池。

不能重复宣称已有优化：RecipeRegistry 在 profiler 关联关闭时直接返回，并非
每条普通 launch 都持 map 锁；workspace 本就复用；event 本就缓存。
`check_and_limit_recipe_execution` 的等待有内存不足条件，看见 wait 不代表
当前驻留模型走到了该分支。这些小项都不预估毫秒级收益。

## 实施顺序与验收

第一批是同一设计，拆成小提交：

1. 用有界 trace 分清提交、事件/回收、地址/workspace 在关键路径上的占比。
2. 建不可变 prepared command 与每次执行 frame，前移静态绑定/workspace 准备，
   共享现有低层 SDK helper 和错误处理。
3. 对可证明顺序区域编译依赖、合并保活回收；分别检验 stream 合流与保留多
   stream 的策略，保留真正有利的 overlap。

第二批补输入重绑/拷贝计划、异步 replay；第三批按未融合证据补编译链覆盖。
第 6、8 项随第一批整理。CPU profiler 是诊断工具，不发展成另一项长期工程。

期望是把反复管理的成本，从“每个内部命令/绑定”降到“变化绑定、图外边界和
真正跨 stream 依赖”。SDK 提交和部分地址补丁仍随图大小增长，不能承诺任意图
O(1)，也不能承诺一项优化就从 75 到 90 TPS。

验收同时看：公开 PyTorch 语义；异步生命周期与内存压力；host/SDK/设备实际
工作量；完整应用步延迟、prefill、多请求表现。性能计时关闭 profiler，保留
matched baseline 和失败。新增模型不应改执行核心。

进一步合入的核心是：**预编译提交描述、可证明的区域依赖、每次执行的资源
生命周期。** 这是原生路径短提交循环和明确所有权中可复用的部分，把它们接到
PyTorch 语义上，才是在完成同一套基础设施。
