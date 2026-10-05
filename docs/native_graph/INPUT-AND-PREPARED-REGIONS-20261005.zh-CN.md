# 从输入入口到区段提交

目标是现有 70 层 MiMo、TP8、B1、4K、无 MTP 的真实推理 ≥90 TPS。
原始证据统一留在本地 `gaudi-kernels/artifacts/builds/bridge-event-batch-20261005/`。
这个目录含被否定的 event-only 方案，也含后续有效输入优化，不能只看目录名
就把所有实验解释成事件合并。

## 已验收的输入优化

bridge `a41e4ca2fb`：推理模式跳过同一个 Tensor 的自拷贝；普通 replay 不再
构建只用于重绑的输入 Tensor 列表；autocast 状态每次输入遍历只读取一次。
保持非推理模式下的 copy/version-counter 行为，不跳过不同对象的别名拷贝。

bridge `3102c66ced`：常见内建容器和普通 Tensor 的输入哈希移入 C++，保留
原 Python 缓存键协议；自定义容器、Tensor 子类和 dataclass 走原语义。
设置递归保护，循环输入抛 Python RecursionError，不耗尽 C++ 栈。
边界处遇到同一个 TensorImpl 时，将重复元数据检查合并到提交前的完整租约
检查；并未省略存储、布局和 shape 变更校验。

同驻留模型内旧包装/新包装/新包装/旧包装，计时窗口为连续 coordinator wall：

| 实验 | batch | control ms | candidate ms | 吞吐 |
|---|---:|---:|---:|---:|
| input70-f，a41 | 1 | 12.802 | 12.166 | 78.12 → 82.19 TPS |
| input70-f，a41 | 2 | 16.291 | 15.685 | 122.77 → 127.51 tokens/s |
| input70-g，310 | 1 | 12.734 | 11.995 | 78.53 → 83.37 TPS |
| input70-g，310 | 2 | 16.276 | 15.493 | 122.88 → 129.09 tokens/s |

两轮都通过 B1/B2 的完整生成序列对比，以及首个 decode、KV 页边界前后的
完整 logits、位置和输入 token 逐位一致检查。8 rank 都走 bridge 计划，无
外置执行器。每轮 control 和 candidate 使用同一 wheel 的 C++，仅切换 SHA
固定的旧 Python 包装和该 wheel 自带包装；因此不能把两轮 candidate 差值
全部归因于某一项 C++ 改动。

83.37 TPS 距 90 TPS 还差约 0.884ms/步，未达目标。

## API 分账纠正了什么

`api70-e` 使用只转发原调用一次的 CPU API observer，没有启动设备 profiler。
rank0 正常完整 step 的执行调用约 8.170ms，其中 SDK launch 2.478ms、HCCL
all-gather 1.811ms、event record 0.361ms、stream wait 0.009ms；余下约
3.502ms 在包装、检查和 bridge 资源处理等范围内。

旧格式 launch 转 Ext 格式合计约 0.039ms，不是毫秒级差距的主因。
后台 GC 线程的 event synchronize 与提交重叠，不能加到上述总时长中。
只合并事件将事件数 291 降到 12，但整模回退；该方案保持关闭，见
[PREQUEUE-20261005.zh-CN.md](PREQUEUE-20261005.zh-CN.md)。

旧 trace 中 Python 进入 C++ replay 之前约 1.74ms。这个有 profiler 的绝对数
不能当预期收益；真正被证明的是上表未启用 profiler 的约 0.64–0.74ms 收益。

## 区段提交：正确性通过，尚无整模收益

bridge `bd63f0d40fa` 在现有 NativeExecutionPlan 的已证明同流区段中，预先
形成 recipe 参数和类型化 collective 调用，不再在每次重放时重复走通用
RecipeLauncher 的整套适配过程。它没有添加 SDK 录制器或第二套 allocator。

- 入口仍通过 bridge 的依赖表处理外部访问，HCCL 原队列及入口/出口 FIFO
  保护保留；不改变 collective 算法、次数和数值运算。
- 每次重放通过现有 allocator 批量获取实际地址锁，不跨重放缓存裸地址。
- 一个在途 frame 持有区段 program、独立 Storage owner、Tensor、recipe、
  communicator 和地址锁。完成事件覆盖所有读写地址，完成后释放资源。
- 工作空间按该区段最大需求取得。编译期不支持的区段继续走普通路径；
  compile-only、同步 launch 和 memory reporter 模式不走这个快路径。
- 提交失败发布已提交前缀的完成依赖；若完成登记失败则 drain 后才释放资源。
  失败图禁止重试，避免把同一个前缀执行两次。

新入口 `native_replay_compiled_regions(graph, enabled)` 默认关闭。
`prepared_regions`/`prepared_region_replays` 证明真实覆盖，不能只检查开关。
事件合并和区段提交两个实验开关互斥。

已通过 `compiled-collectives-j` 双卡 FP32/BF16、8/32 段通信链、普通通信
前后衔接、跨流读写、混合 peer 回退、另一 rank 尚未提交时 reset 图及 Storage
存活检查。显式失败注入确认错误发生在 prepared region 内，随后重试没有新增
SDK 提交，reset 和 drain 成功，无设备 reset。

`compiled2-j` 两层 B1/B2 生成与六个完整 logits 检查点通过，性能基本持平。
70 层 `compiled70-j` 已通过 B1/B2 全序列及完整 logits 检查，但没有性能收益：

| batch | control ms | candidate ms | 吞吐 |
|---|---:|---:|---:|
| 1 | 11.836 | 11.853 | 84.49 → 84.37 TPS |
| 2 | 15.463 | 15.525 | 129.34 → 128.83 tokens/s |

同进程 ABBA 只切换区段提交开关。不能用热链测试或 CPU 提交耗时下降
宣布模型达到 90 TPS。约 84 TPS 是当前结果，90 TPS 尚未达成。

## 下一轮：缓存地址解析并改善事件集合的算法复杂度

`3e515e36a` 已编译并完成以下验收；正确性通过，仍未证明性能收益：

- allocator 持有地址翻译缓存，每次使用仍取得短期防搬移租约；
  碎片整理世代和每个逻辑句柄的分配世代共同校验地址。
  句柄释放再复用不能伪装成原分配，无关分配活动不使整个图失效。
- 大事件的地址移除由逐次线性扫描改为索引 swap-pop，小事件保留简单向量。
  事件管理器仍负责并发互斥，地址遍历顺序不构成执行次序。
- 没有待提交 future 时避免按地址重复取得 future-map 锁。
- inference 模式下同一内建输入容器不再递归执行自拷贝；保留自定义对象语义。

验收包含 CPU 集合/句柄复用测试、真实显存碎片整理后的缓存失效、
双卡跨流和失败前缀测试，再做两层与完整 70 层 ABBA。
初版 `0a36693a3` 虽然编译成功，但发现句柄复用缺口后未部署测试。


64 线程构建耗时 375.98 秒；21 项 C++ 测试、5 项输入语义测试通过。
`cached-collectives-k` 双卡实际分配约 91.4GB，释放交错的 3.81GB 块后
申请约 18.29GB，触发 allocator 整理；两卡均从一次缓存解析增加到两次，
重新 replay 的数值正确。跨流读写、peer 阻塞后 reset、提交前缀失败与
零新增提交的禁止重试检查通过。这是设备压力测试，不是手动增加世代的模拟。

`cached70-k` 完整 B1/B2、160 输出、4K、同进程 ABBA：

| batch | control ms | candidate ms | 吞吐 |
|---|---:|---:|---:|
| 1 | 11.7402 | 11.7348 | 85.1777 → 85.2166 TPS |
| 2 | 15.4575 | 15.4177 | 129.3870 → 129.7207 tokens/s |

输出及六组完整 logits 检查均通过。8 rank 各 B1 主图的 1004 次候选提交
均进入 prepared region，地址缓存 1003 次命中、1 次解析，不是未生效或回退。
这次对照的两臂都包含公共输入/事件集合改动；跨实验从约 84 到约 85 的变化
不能据此归因。prepared 区段本身仍无明确收益，保持默认关闭。

90 TPS 需要 11.1111ms，当前候选还差约 0.6237ms/步。下一诊断在独立的
4K 检索问答 fixture 上复测并检查生成答案，CPU trace 放在计时完成之后。
不以设备或 CPU 局部时间的倒数冒充真实模型 TPS。


## 覆盖审计纠正：此前只证明进入了一个小区段

`retrieval70-l` 的稳态 CPU trace 揭示前述 region 计数不足以证明模型覆盖：
每步 286 个命令，control 为 286 次普通 Launch；candidate 仍为 283 次
普通 Launch，只有 3 个命令进入 PreparedRegionSubmit。候选区段约 91us
（带 profiler，仅用于归因），不是整模型提交主体。

因此“1004 次 prepared replay、1003 次缓存命中”只证明小区段运行，
**不能用此前的无收益结果否定大区段预准备或缓存，也不能宣称主路径已迁入。**
这是验收统计的缺陷。新增按命令/compute/collective 计数，以及带命令索引、
绑定类型和 Tensor ID 的区段拒绝原因，先定位再修复；不再以 region_count>0
作为模型覆盖门。


两层 `coverage2-m` 在 `c7a2a1fea` 上复现了精确拒绝原因：
`commands 7..11: command 8: non-data binding type=1 id=18446744073709551615 name=14`。
该 Tensor ID 是 `INVALID_SYN_TENSOR_ID`，绑定已被编译器删除，却仍残留于
原 launch 参数向量。依赖审计早已跳过这一哨兵；区段 lowering 没有应用同一规则。

修复 `ea40817c6` 仅在实例化时排除这些编译器删除的项。有效绑定的类型、
地址、生命周期和外部事件规则不变。并非放宽 live shape Tensor 的准入。

`lowered-collectives-n` 双卡安全门通过；`lowered2-n` B1/B2、页边界完整 logits
通过。两层从 prepared 3/13 命令增至 8/13，可合并 collective 5/5，拒绝项为零。
新测试选项 `--require-prepared-coverage` 要求 8 rank 真正执行 prepared 命令，
且全部已准入 serial collective 均有 prepared adapter。


## 完整覆盖后的实测与真正获得收益的输入准备

`lowered70-n` 已通过全部质量门。每张卡主图 286 个命令中 281 个进入
prepared region，141/141 个可合并 collective 均覆盖，拒绝项为零。
B1 为 84.432 → 84.615 TPS，B2 为 129.051 → 128.876 tokens/s，基本持平。
CPU graph.replay 从 6.818 降至 4.519ms，但输出等待从 3.501 增至 5.808ms；
二者重叠于设备运行，不能把 CPU 提交节省直接计入模型收益。

`layout70-r` 保持相同设备计算，复用 vLLM decode 的静态 CPU/HPU 输入布局，
按实际 scheduler 的 token、位置和页表更新动态字段。请求、页表或布局改变则
回到原 builder；没有预测下一个 token，没有绕过 PyTorch copy 或 allocator。
完整 70 层、TP8、4K、160 输出、B1/B2 同进程 ABBA：

| batch | control ms | candidate ms | 吞吐 |
|---|---:|---:|---:|
| 1 | 11.7785 | 11.4938 | 84.9002 → 87.0033 TPS |
| 2 | 15.4555 | 15.2255 | 129.4034 → 131.3586 tokens/s |

两次候选和两次对照的输出一致，六组完整 logits 检查通过，各 rank 输入布局
缓存实际命中。CPU 输入准备约 0.660 → 0.345ms，这是位于提交前的真实收益。
距离 90 TPS 的 11.1111ms 门槛仍差约 0.3827ms，不能宣称达标。

## GIL／事件锁／提交队列的死锁与修复

随后 `static2-s` 移除未使用的 KV 输入树、加快标量和元数据哈希，但在质量阶段
挂起，因此其计时全部作为未验收数据保存，不纳入性能结论。真实调用栈显示：

1. Python 主线程持有 GIL，在 graph.replay 中等待 HCCL 提交 future。
2. HCCL 提交线程在 RecipeLauncher 中等待事件 mutex。
3. 资源清理线程持有该事件 mutex，释放带 Python owner 的 Tensor 时等待 GIL。

这是一条确定的锁依赖环，不是模型数值问题。旧的 caller-thread launch 有
内部 AutoNoGIL；将执行排入独立提交队列后，调用方不能继续持有 GIL 等待。
`7022d55e2` 在 graph Python/C++ API 边界、取得 graph/plan 锁之前释放 GIL，
覆盖 replay、捕获、reset、输入标记等可能阻塞的接口。统计先在无 GIL 的作用域
复制纯 C++ 数据，再取得 GIL 构造 Python 容器，避免逆向的 GIL/graph 锁次序。

测试 interposer 暂停一次 SDK launch，必须由 Python 辅助线程释放，并在释放后
并发读取状态和 reset。超时上限 2 秒，不依赖随机触发垃圾回收：

- 旧 wheel `e0ea1872f` 的 `gil-old-u` 在该点超时失败，符合负对照预期。
- 新 wheel `7022d55e2` 的 `gil-new-u` 双卡普通和 prepared 两条路径均通过；
  Python 在提交返回前取得执行机会，并发 snapshot/reset 与输出均正确。
- 同轮 FP32/BF16 通信链、跨流依赖、提交前缀失败和真实显存整理后的地址
  缓存失效测试通过。6 项 hash 语义测试及 4 项 CPU 输入布局对照测试通过。

修复后的整模型验证仍需收账；本节不把修复死锁当作 TPS 收益。
原始结果、旧版本失败、调用栈、源码 bundle 和 wheel 均归档于本地
`bridge-event-batch-20261005` 实验资产目录。


## GIL 修复后的完整结果与小输入批量上传

`gil70-u`（bridge `7022d55e2`、serving `847487d`）通过完整 70 层、TP8、4K、
无 MTP、160 输出、B1/B2 ABBA。使用独立检索 fixture，所有答案含正确的 7391，
六组页边界完整 logits 一致，没有外部执行器。连续输出墙钟结果：

| batch | control ms | candidate ms | 吞吐 |
|---|---:|---:|---:|
| 1 | 11.6049 | 11.2904 | 86.1705 → 88.5712 TPS |
| 2 | 15.2311 | 14.8948 | 131.3104 → 134.2753 tokens/s |

候选距 90 TPS 仍差 0.1792ms/步。去掉未使用的 KV 树、原生标量/元数据哈希、
GIL 修复处在该版本的共同底座里，不能用此 ABBA 单独归因其中任何一项。
`rows70-v` 加入 unit-query CPU 行准备及直接更新已选缓冲后，质量仍通过，
B1 88.5216、B2 134.1970，与前轮基本持平；尚未证明这两处带来整模型收益。

继续复用现有 `device::copy_data_to_device(transfer_manifest)`，为标准
`aten::_foreach_copy_` 补小型、独立 H2D 输入批量提交。保留普通 Lazy Tensor
准备，整批只做一次 DMA/producer 登记。视图、重叠目标、转换等非准入情况
委托原 PyTorch slow kernel；非 inference 调用不改变 accumulation-thread 行为。
大传输保留原路，避免为权重上传额外复制大块 host staging。

- `copy-x` 混合 INT32/BF16/FP32 的五输入微基准：含 graph 和完成等待，约
  117–118us → 79–82us。32 次变输入并立即毒化 CPU 源、待完成 D2H 及别名、
  视图、类型转换回退均通过。这是微基准，不是模型 TPS。
- 首次 `copy-w` 把 profiler 的 B/E 两个事件误数为两次调用，导致覆盖门假失败；
  修为只计 B/X，失败记录保留。
- 审计真实模型发现 slot 为逻辑 INT64，故复用原 `preProcessIfLongorDouble`
  转换后再批量提交。不能将逻辑八字节直接送入物理四字节 Tensor。
- `copy-aa` 测试误以为 `_foreach_copy_` 支持反向传播；旧 wheel 的纯 CPU 路径
  独立复现相同的 derivative-not-implemented 错误。验收改为与上游 CPU 行为
  对照，不把上游没有实现的功能误判为回归。
- `copy-ab` 真实混合类型、异步依赖和上述回退检查通过；`copy2-ab` 两层 B1/B2
  及六组 logits 通过。计时后 CPU trace 确认八 rank 各 62 次批量 H2D 提交，
  没有只在微基准生效。

进一步把 Lazy 边界按 Data 身份去重，并批量验证 Tensor 绑定。Lazy 层拥有该表，
backend plan 仍仅处理 Tensor lease；所有 retained Tensor 的完整元数据校验保留。
同时补齐 GIL 释放后 stats/诊断/clear_inputs 与 reset 的 graph 互斥。
该改动尚在构建验证，不能宣称已达到 90 TPS。
