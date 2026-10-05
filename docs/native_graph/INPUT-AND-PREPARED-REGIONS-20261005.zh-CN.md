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
