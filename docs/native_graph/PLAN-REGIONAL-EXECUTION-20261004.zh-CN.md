# Bridge 执行区域优化：实施计划

## 目标与现状

在已有 `HPUGraph` / `NativeExecutionPlan` 内完成通用的编译依赖、提交和资源
生命周期。保留公开 Tensor、stream、ProcessGroup 语义；最终退役两个外置执行器。
不新增模型包装器、allocator、线程池或 serving runtime。

当前 70 层 MiMo、TP8、4K、无 MTP、B1 实测约 76 resident TPS。
90 / 100 TPS 分别要求约 11.11 / 10.00 ms 每步，还需省约 2 / 3 ms。
这些是验收目标，不是已经归因或保证能删除的开销。旧约 90 TPS 路径还捕获了
输入搬运和采样，并改变 execute/sample RPC 组织，不能当作只换 bridge 的等价对照。

用户要求：使用 Git；编译 32 线程；权重、kernel 和数值策略保持对照一致；
FP32 合法舍入可接受；所有源码、构建、失败、原始结果和报告保存在本地。

## 阶段与交付

| 阶段 | 实施 | 完成条件 |
|---|---|---|
| A：编译依赖 | 记录 compute/collective 的真实读写，区分输入使用结束登记和真正输出；编译存储级 RAW/WAR/WAW 与 collective 顺序；标明 partial/external 事件 | C++ 最小图验证别名、输入覆写、独立分支和外部事件；正常模型能够报告计划分类 |
| B：顺序区域 | 只对已证明必须串行的计算→通信→计算区域评估同一物理 stream；区域入口/出口保持原 communicator FIFO | TP2 正常/混合路径、图外跨流读写、变化输入、连续执行和在途销毁通过；对现有 bridge 配对计时，无 overlap 回退 |
| C：区域资源 | 每次执行持有 Storage、地址租约、recipe、communicator、workspace、staging，完成后统一回收；对外可见完成与保活回收分开 | 内存压力、异常、并发执行、reset、图外消费；显存高水位和完成计数有界 |
| D：预编译提交 | 静态绑定索引、别名、workspace需求前移；共享原 SDK helper，减少逐命令通用 launcher 管理 | 输入内容变化与合法布局重绑正确；地址仍经 allocator 解析和保护；关键路径或模型步延迟有收益 |
| E：公共输入和异步 API | 补 V3 / replay_with_inputs / async adapter；通用拷贝计划，完成前持有 host 源 | view/in-place、pinned/non-pinned、新 Storage、跨流、异步销毁和多 bucket 覆盖 |
| F：捕获与前端统一 | Lazy 和 compile 产物进入同一执行计划；通过正规上层图范围覆盖 logits/采样等设备计算 | 第二模型复用执行核心；B1、batch、prefill、verify 共用机制；不向 bridge 加 token/KV页规则 |

先完成 A，再分别评估 B 和 C，避免一轮同时改变 stream 和回收造成不可归因。
每个候选使用小提交；支持不足的区域在提交前保留原路径。提交过一部分的计划
失败必须终止，禁止重复执行整个 collective 链。

## 设计约束

- 捕获时以 Storage / 别名建立保守访问关系；未知副作用不推定为纯读。
- 相同 stream 的顺序边不需要重复 SDK wait；不同 stream 的真实依赖保留。
- 独立分支和 recipe 的 partial readiness 可能有有效 overlap，不能全部串行化。
- 通信顺序还包括不同调用之间的 communicator FIFO，不能全局改 ProcessGroup stream。
- Storage 保活不保证物理地址稳定；地址租约不保证 Storage 保活，两者都需要。
- 区域回收可以晚于某 Tensor 对外可用时间；不能强制所有输出都等终点才发布。
- 顺序 recipe 可以复用 workspace；并发执行不能共享未经保护的 workspace。
- 不要求一次硬件 launch，也不承诺任意图 O(1) 提交。真实 SDK 命令仍需执行。
- bridge 不理解 token、position、slot、SWA 页或模型层名。

## 验收和退役

先验证公开 API 和异步生命周期，再做 operator A/B；有收益或明确的质量诊断目的
才运行完整模型。对整模型记录 EngineCore resident wall、worker wall、提交与设备
完成，计时关闭 profiler；嵌套区间和重叠引擎不相加。硬件 trace 仅用于诊断。

保留 d8f717434 的 wheel / 源码作为最近已验收对照。B1 90–100 TPS、B2/B8 和
prefill 不显著回退是性能目标；单次或小于噪声的变化不能宣称稳定收益。
图外 CPU 逻辑只能由上层正规改造消除，不能用模型特例伪装成仅替换 bridge 成功。

外置执行器只有在公开入口覆盖、正确性、生命周期、真实性能与可部署性均通过后
才停止使用。冻结源码和历史失败资产继续保留，不能为简化仓库删除证据。

## 本轮起点

Bridge 起点 `c997a59d8`；kernel/验证起点 `87e0443`。
本轮首先实现 A 的保守依赖描述和测试。A 的元数据本身不记性能收益。
后续实现、远端可用性、设备选择、结果与阻塞写入本计划相邻的结果报告。

## 首轮后的路线修订

见 [实测报告](RESULTS-REGIONAL-EXECUTION-20261004.zh-CN.md)。A 和 B 的首个受限
适配器已经实现；19项 C++、TP2 和完整70层执行等价通过。实际模型140/141通信
合流，B1仍约76 TPS，没有可确认收益。单通信区域被配对测量否定并排除。

下一步将 C/D 作为一项资源设计、分开提交和测量：

1. 明确公开输入、输出、被修改输入与内部 Tensor；编译 entry/exit及必要早期
   readiness 发布点。不能把所有子图边界都当公开边界，也不能猜中间变量不逃逸。
2. 创建每次执行的资源实例，复用 bridge allocator/stream；在提交前预定本区域
   workspace容量、解析地址和保活所有实际使用资源。异常和在途 reset 不提前释放。
3. 替换可证明内部边的 producer-map查询/登记；减少逐命令 SDK 完成事件和回收，
   保留图外读写与早期完成发布。未知语义继续原路径，不一刀切只留一个尾事件。
4. 对已经准备好的命令直接复用底层 launch helper，更新变化绑定；不复制通用
   launcher，不缓存无租约的裸物理地址，不把权重永久钉在新的私有 arena。

验收重点改为**设备完成/完整步确实缩短**。本轮 host forward 提前返回，但等待
移到 cpu()，采样窗口没有变；下一候选不能重复把这个移动当作收益。小图需覆盖
变化输入、alias、跨流图外消费/覆写、peer-blocked销毁、并发图、内存压力和异常。
记录完成事件/回收数量、显存高水位及 workspace增长，再验收70层和batch/prefill。
E/F继续使用同一个执行核心，旧两套执行器仍仅作为冻结对照，不满足退役条件。
