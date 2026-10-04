# 单一 bridge 原生执行核心：实施计划

日期：2026-10-04。状态：**P0构建闭环完成；P2静态计算子集已通过设备检查，P3部分通过，模型目标尚未验收。**
依据：[FEASIBILITY.zh-CN.md](FEASIBILITY.zh-CN.md)。

## 目标和不可偷换的验收

最终安装物是修改后的 PyTorch Gaudi bridge wheel。应用使用原来的 PyTorch/HPU
Graph 入口，不依赖外置 `legacy_executor`、`native_graph` 或 LD_PRELOAD 拦截。
同一套无 MTP MiMo TP8 应用代码、权重/内核/精度，通过更换 bridge 达到单流
**90 TPS 以上**；不能靠更换 runner、减少模型层数、缩短上下文或改变输出计数达标。

基础设施必须独立于 MiMo、vLLM：执行核心不得出现层数70、top8、SWA128、token
位置、专家数、模型路径、采样规则等语义。模型/服务适配不进入 bridge 核心。

原生90TPS是参照而非已证明的 bridge-only 上限。相关历史实测是约40–42/90–91；
“60/90同策略配对”尚未确认。P1重新固定实际基线，不追求把对照人为调到60。

## 唯一实现的边界

| 模块 | 责任 | 不做什么 |
|---|---|---|
| 现有 Python HPU Graph API | 缓存选择与用户入口，保持兼容 | 再包一层 gaudi-kernels Graph |
| HPUGraph 捕获/编译适配 | 收集各子图编译产物、Tensor binding、依赖 | 模型代码截获、全局私有 ABI hook |
| bridge 内 NativeExecutionPlan（拟定内部类型） | ordered commands、binding slots、资源依赖、执行状态 | 第二个高层算子编译器、模型调度器 |
| bridge Tensor/allocator/stream 适配 | storage/offset/物理地址锁、外部 producer/consumer 依赖 | 裸指针永久固定、绕过分配器 |
| 现有 Synapse/HCCL | recipe 编译、设备命令与通信执行 | 宣称一次 replay 变成一次硬件提交 |

建议在 `backend/habana_device/` 内落执行计划实现，由 HPUGraph 持有。
通过正式 C++ builder 调用接收编译产物。显式 producer 后续交同一格式。
模块名可调整；不得新增第三套包、服务启动器或独立内存池管理器。

## 分阶段交付

### P0：构建闭环与源码身份

改动/工作：
- 固定上游 `5176b1b2`；保存现有安装 wheel/build-ID/依赖信息。
- 按 `.devops/build.py` 准备独立构建目录与 venv，先构建未修改的上游。
- 用 `.devops/build.py -cr` 类构建入口准备 release 产物；实际参数以锁定环境为准。
  不在共享环境直接执行 README 的 `-i` 安装步骤；安装到专用验证 venv。
- 先确认 CMake>=4、匹配的 Torch/C++ ABI、Synapse/HCCL 头文件/库和离线依赖。

验收：可复现 wheel、隔离 import、单卡正常张量与原版 HPU Graph 测试；产物
SHA和版本可追溯。无此闭环不进行大范围源码迁移。当前：完成，见
[RESULTS-20261004.zh-CN.md](RESULTS-20261004.zh-CN.md)。

### P1：冻结 A/B/C 基线与区域外成本

复用70-g冻结源码、同策略结果和现有 benchmark，不重新造服务压测器。

| Arm | 应用 | bridge | 外置执行器 |
|---|---|---|---|
| A | 冻结的普通 PyTorch 路径 | 上游未改 wheel | 无 |
| B | 同计算配置的旧 native 参照 | 同一上游 wheel | 仅作对照启用 |
| C | 与A逐文件相同的应用代码/配置 | 本fork wheel | 无 |

A/C 的 runner、RPC组织、内核、配置、输入完全相同；B额外优化必须单列。
B的 single_rpc/reuse_pages 进行独立消融，不把其收益记给 bridge。
对于普通路径原有的自研计算算子，仅保留计算接入；所有 execution hook 必须从
A/C可执行依赖中排除并用进程加载库检查证明。

记录：冷启动/编译与热重放分别计时；HPUGraph、worker、EngineCore、客户端到达；
recipe/collective数、H2D/D2H次数字节、事件/同步数、内存峰值和图外时间。
profiling与干净计时分开。使用ABBA或交错重复；不同wheel用独立进程，不能谎称
同进程替换。返回token数除以相应完整时间，不把中位耗时倒数冒充吞吐。

验收：A、B可重现，输出一致性可核对；可解释哪些差距属于bridge内部。
若图外工作已超过90TPS预算余量，立即报告严格目标的阻塞，不扩大bridge职责。

### P2：集成最小计算计划，不接模型

复用新runtime的命令/访问范围检查思想及旧提交循环；增加 bridge 内计划类型。
- 从编译缓存获取 recipe shared owner、workspace、输入输出/中间Tensor。
- 从两个相连的recipe开始，通过正常HPUGraph入口产生计划。
- 保留正常输出Tensor身份及必要的Lazy状态更新，只在边界执行。
- 首轮只支持明确静态推理子集，检查不支持条件后才提交。
- 区分capture_end与线程工作实际完成；直接集成不能继续依赖全局截获顺序。
- dry-run不能仅监听Launch；在编译完成时导出描述，或首个真实执行后物化。

改动点：`HPUGraph.h/.cpp`、`hlexec.cpp`、`hpu_habana_launch_op_pt.cpp`、
`hpu_habana_cache.h/.cpp`、`backend/CMakeLists.txt`。

验收：正常API执行两recipe链，变输入重复运行；热重放不进入逐子图
`ExecuteCachedGraph`/`HlExec::Launch`；命令、输出、异常正确。不能只测私有新API。

### P3：Tensor、地址与生命周期

- 统一记录 storage身份、offset、dtype/layout/shape、读写与别名。
- 使用bridge分配器的workspace/地址锁；不另向Synapse抢占一份模型显存。
- 初始支持固定shape下输入新Tensor重绑；view/alias不支持时提交前明确回退。
- graph外producer/consumer使用框架stream依赖；输出可被普通Torch算子立即消费。
- 完成通知接回现有资源回收机制，默认不为每个recipe独立强制同步。
- cache淘汰/reset销毁必须处理未完成工作、输入释放与错误失效。

验收矩阵：连续/非连续view、slice、transpose、alias、原地更新、输入地址替换、
相同地址复用但不同owner、allocator churn、允许时的搬移/碎片整理、两个stream、
多个graph并存、异步后释放/reset。BF16/FP32/整数含逻辑Long与物理I32情况。
检查全部输出与真实消费结果；不能把逻辑data_ptr当成物理地址稳定性的证明。

### P4：通信与完整图重放

- 从 `CollectiveKernelInfos` 与具体 collective operator 导出结构化命令，
  覆盖现有目标路径的AG/AR，再补RS。
- 保留communicator owner、rank/count/dtype、buffer区间及跨stream依赖。
- 只对证明内部有序的区域简化事件；SFG/external-event未知语义先不优化。
- 每次replay前验证完整计划；分布式各rank选择一致路径。
- 提交后的错误进入失败状态，禁止重跑fallback造成重复写入或collective死锁。

验收：单卡计算＋拷贝、TP2/TP8 producer→collective→consumer，输入变化、
in-place/out-of-place、空量、图切换、通信错误清理；无外置录制库，正常HPUGraph API。
在profile里证明没有逐collective重复的旧线程/future路径，同时保留必要边界依赖。

### P5：同应用、只换bridge的整模型验收

- 先2层定位，再70层TP8；保持A/C应用哈希与内核产物一致。
- B1/4K无MTP，普通生成与正常EOS、长生成跨128页边界；另测32K。
- 重放算术顺序不变时要求逐值/逐位一致；若不可避免改变合法FP32运算顺序，
  使用正常FP32级误差和teacher-forced/生成检查，不引入FP64精确门。
- 数值错误、漏算、错layout、stale KV、读错路由不能被容差掩盖。
- C无MTP单流达到90TPS以上，并记录均值/尾部、显存、编译和fallback占比。
- A/C仅bridge变化；B只是参照，不能让C调用其native_step或模型专用输入捷径。

若未到90，按P1的分层计时解释剩余成本；功能正确或局部replay变快都不能代替最终目标。

### P6：证明是基础设施

- 同一核心覆盖B2/B3/B8、静态T桶和prefill形状切换；测试2048/4096 chunk执行
  及真实内存占用。算子本身的GEMM收益仍单独验收，不混进bridge差值。
- 至少一个不含MiMo语义的独立Torch图程序，通过正常API运行计算＋通信。
- DFlash之后使用同一核心，draft/verify/KV提交逻辑属于上层；不另造MTP执行器。
- 训练/任意动态图暂不承诺，必须保持未支持路径的正确行为，不静默误用静态计划。

验收：新增形状/模型不修改执行核心；输入地址/生命周期变化不需要模型级补丁。

### P7：退役与交付

- 把已迁移实现的出处、许可证、测试及失败证据归档本地。
- 删除生产依赖中的旧preload/私有launch hook、tensor_hold、native_step/RPC快捷入口。
- 删除外置native_graph的重复执行、缓存和资源管理产品入口；保留必要历史档案。
- gaudi-kernels保留算子库和适配所需算子注册，不再携带第二套运行时。
- 清理旧HPUGraph在已支持静态路径上的重复管理，保留必要普通执行fallback。
- 交付wheel、版本锁、兼容范围、基准命令、API说明、迁移报告与本地资产清单。

验收：干净环境安装，仅普通PyTorch/模型接口即可获得C的结果；进程中没有两套
外置执行器。旧路径只在归档与对照工具存在，不作为永久可选生产backend。

## 当前可执行的下一项

**继续P1、P3与P4。** 当前实现状态见 [IMPLEMENTATION.zh-CN.md](IMPLEMENTATION.zh-CN.md)。
已完成隔离源码构建和两 recipe 同 API 的配对实验；尚未量化整模型同代码 A/C
能消除的全部时间。先拆开计算注册与外置执行 hook，再完成输入重绑和通信计划。
每个阶段用独立Git提交交付，沿用既有最小实验、模型runner与本地资产归档工具。
