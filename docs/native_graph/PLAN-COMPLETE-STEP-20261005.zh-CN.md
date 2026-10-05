# 完整设备计算范围迁移

后续结果见 [RESULTS-COMPLETE-STEP-20261005.zh-CN.md](RESULTS-COMPLETE-STEP-20261005.zh-CN.md)：
70 层实现和完整 logits 检查完成；收益约 0.24–0.32ms，未达到 90TPS。下文保留
第一检查点原始预算与阶段说明，以后续结果为当前裁决。

## 问题与边界

同驻留模型的旧 SDK / bridge 对照确认 70 层存在 2.8–2.9ms 差距；缩到两层
仍有约 2.3ms。先补完整执行范围，不能把差值直接归因到通信事件或硬件吞吐。
详细证据见 `GAP-ISOLATION-20261005.zh-CN.md`。

唯一执行核心仍是 bridge 的 HPUGraph / NativeExecutionPlan。vLLM 前端负责
声明 forward、选行、logits、sampler 的计算范围及请求语义；bridge 负责捕获、
依赖、内存所有权和提交。不能往 bridge 加 token、KV、专家路由或模型层数。
外部 SDK executor 仅作历史对照，不作为候选的执行依赖。

## 实施顺序

1. 前端完整图：无约束同步 greedy decode 将 forward 至 sampler 放进一张图；
   内层 forward 包装器 bypass，避免嵌套捕获。其他请求继续原路径。
2. 复用输入 Tensor：按角色、形状、dtype、device 复用；仍经普通 copy_，由
   bridge 保留 dtype 转换、WAR 依赖和在途所有权，不直接缓存裸地址。
3. 同进程 ABBA：先两层 B1/B2，再 70 层；所有候选 rank 必须实际执行 ready
   的 NativeExecutionPlan，禁止把未命中候选的结果算作收益。
4. 补全数值/生命周期验收：输出序列、完整 logits、请求更替、形状更替与页边界；
   随机采样、约束处理、异步、MTP 在没有相应状态合同前不纳入。
5. 剩余成本按新证据定位。仅在关键路径实测成立时继续将输入绑定、资源准备
   或通用提交工作前移到 capture。收益按完整步 wall，不能相加嵌套 host 时长。

## 第一检查点

前端代码在 gaudi-kernels `cca9e535`，使用已编译 bridge `77d8490270`。
没有修改模型算术，没有加入新执行器；本检查点也不宣称已经改进 bridge C++。

两层、TP8、4K、64 输出 token，同进程 forward / step / step / forward：

| batch | forward 平均步 | step 平均步 | 减少 |
|---|---:|---:|---:|
| 1 | 3.139ms | 1.934ms | 1.205ms |
| 2 | 3.186ms | 2.040ms | 1.146ms |

这是 resident engine.step 耗时口径，不是完整 70 层 TPS，也不是与旧 SDK 同场
配对结果。后续驱动同时保存连续 coordinator wall，避免漏算步间开销。
8 rank 全部通过，两种 batch 的所有 ABBA 输出一致，原生计划 ready、无失败、
有实际 replay。完整 step 每图 8 compute recipes、6 collectives；相较先前
forward 的 7/5 范围增加，同时图内选行/logits/采样编译合并，因此不能以旧 SDK
10/6 的 recipe 数为机械验收标准。

未完成：70 层收益、完整 logits 对比、剩余开销归因。默认仍为 forward。
资产根目录：`gaudi-kernels/artifacts/builds/bridge-step-20261005/`，继续保留本地。
