# Bridge 内部编译计划与重放实现

应用继续使用原有 `HPUGraph` / `wrap_in_hpu_graph` API。执行计划是 bridge 的内部对象，
没有新增 Python 执行器、模型专用重放入口或部署时改源码的脚本。普通路径尚未完成
70 层性能验收，不能据小图成绩宣称达到 90 TPS。

## 捕获：把编译结果交给计划

`HPUGraph` 拥有 `NativeExecutionPlan`。`HabanaLaunchOpPT::ExportCapturedExecution`
在正式 fresh/cache 编译结果接点导出 recipe、collective 描述、Tensor 和抽象地址绑定。
导出发生在 SDK 将抽象地址改成物理地址之前。公共 `RecipeLauncher::Launch` 不再
查询 Lazy 捕获状态；它只负责已有的底层提交与资源管理。

每条计划命令拥有 recipe launcher 快照、stream、输入/中间/输出 Tensor、
抽象 launch bindings、稳定名称及外部事件索引。seal 要求命令数等于捕获子图数，
并检查静态 recipe。没有 compute recipe 的纯 collective 命令也可接纳。

## 绑定：输入值可变，分配和槽位不能偷偷变化

`RetainedTensorBinding` 独立保留 `c10::Storage`，避免 TensorImpl 的原地重绑释放
旧分配。每个租约记录 storage 身份、抽象地址、分配字节数、offset、dtype、shape、stride。
捕获的每个公开输入/输出槽都有其确定的租约索引；不能拿图内另一个合法 Tensor
冒充当前槽。值变化与精确 alias 允许，绑定变化在任何提交前拒绝新路径。

重放使用分配器原有的地址锁，每次从抽象绑定重新产生物理绑定。它不会复用捕获时
的物理指针。暂只支持无梯度、连续的后端 HPU Tensor 和冻结标量。部分公开 view
在捕获时已被 lowering 转成支持的后端绑定，不等于任意 alias 都受支持。

## 通信：复用既有 HCCL 队列

FP32/BF16 all-reduce 和 all-gather-out 提供 typed `SnapshotForReplay`：冻结算子参数、
保留 communicator、复制并保留 patched Tensor 描述。不支持的 collective 使整个计划
在提交前回退。recipe 与 collective 原有 producer/consumer/external-event 语义保留。

含通信的整张计划只向既有 `JobThreadLazyHCCL` 队列加入一个 job。该 job 内顺序提交
全部 compute 与 collective；typed collective 在队列线程内 inline 提交，省掉每个
collective 独立的 job/promise/future。整计划完成 future 表示 SDK 已接受提交，
不是设备执行完成。复用既有队列保证与普通 collective 的 host 提交顺序一致。
compute-only 计划仍在调用线程提交。

这不是一次硬件 launch：仍有逐 recipe 的 `RecipeLauncher::Launch` 和逐 collective
的 SDK 提交。地址锁、workspace、事件、producer 登记和异步资源回收仍复用 bridge。
尚未实现 all-rank 快路径协商。两卡实验中的模式一致性检查在诊断 harness 中完成，
不能称为运行时自动协商。

## 重放与回退

普通同步 `replay()` 检查全部绑定，直接遍历编译计划，跳过逐子图的
`ExecuteCachedGraph` / `HlExec::Launch`。stream guard 保留捕获 stream 并恢复调用方 stream。
提交前不支持的条件走旧路径；提交后异常会 poison 整计划，所有 replay 入口须拒绝
重跑，避免重复写入。队列 job 捕获异常并回传调用者，队列能继续处理后续工作。

显式新输入重绑定仍使用既有 V3 adapter，暂不走新快路径。本轮修复了原 bridge
`replay_with_inputs` 遇到普通 HPU Tensor 会直接返回而不计算的问题；现在非空输入
都会进入 V3。marked input 的数量、defined/shape 在首个子图提交前检查；reset 清理
输入元数据。RNG、dry-run、多个 recipe stream、DMA 输入、async replay、输入释放、
输出重写等目前仍明确回退。fast-path 支持范围与公共 API 功能支持范围应分别报告。

## 应用接入与诊断

serving 的 typed `executor=pytorch` 只注册相同计算算子，跳过旧 `libe1` preload、
外置 executor 启动及模型专用传输 hook；原生 runner 在没有专用传输对象时调用
原 PyTorch superclass。因此模型 A/C 可使用同一应用、权重、算子及配置，只换 wheel。

`_hpu_C.native_replay_stats` 只用于诊断命令数、collective 数、重放数、queued_replays
和拒绝原因。模型 snapshot 在计时区间外记录所有 rank 的统计及实际加载库。
部分层数测试只验接线和等价性，不验语言质量或完整模型 TPS。

## 已验与未验

最终计算/绑定候选 `413e535ce` 从源码构建，编译使用 `-j 32`。CPU 的 4 个真实 Storage
绑定测试和 2 个队列异常/嵌套测试已在 `6169def46` 构建上通过。普通 wrapper 覆盖
FP32/BF16 的 M=1/2/8/64/512；输入 adapter、cross-stream、lowered view、churn/reset
在最终候选上通过。两卡 compute→all-reduce/all-gather→consumer 在 FP32/BF16 下
通过，包括 rank0 新路径、rank1 V3 回退。两层 TP8 每 rank 11 条命令、5 个 collective，
45 次新路径重放；32 个 token 与原 bridge 完全一致。

70 层同应用 A/C 正在执行。分布式在途 communicator 销毁、全异常矩阵、训练/RNG、
动态 shape、正式快速输入重绑和全部模型场景仍欠验收。保留现有外置执行器作为
对照资产，尚不能宣布弃用。原始日志、失败、wheel 和源码包留在本地私有资产目录。
