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

## 正在验收的区段提交

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
70 层 `compiled70-j` 仍待结果，不能用热链测试宣布模型达到 90 TPS。
