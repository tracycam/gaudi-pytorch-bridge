# 原生执行能力并入 PyTorch Gaudi bridge：源码可行性审计

日期：2026-10-04。上游基线：`5176b1b2d6865608514e83a2b1fd66459920676e`
（`v1.24.1`）。旧执行器与新 runtime 的审计基线是 gaudi-kernels `74bdf88`。

本文件保留实施前的源码审计时点。“未构建、未上卡”等描述指该时点；后续
构建与设备结果见 [RESULTS](RESULTS-20261004.zh-CN.md)，复核验收见
[REVIEW](REVIEW-20261004.zh-CN.md)。

## 裁决

**认可在 bridge 内建立唯一原生执行核心，并退役两个外置执行器。源码接口支持这条路线。**
**不认可“把新 graph 原样放进 HPUGraph.replay 就必然达到 90 TPS”。**
当前结论是源码级工程可行性，尚未完成 fork 的构建、设备最小实验或整模型性能证明。

用户的最终验收保持不变：相同应用代码、模型、算子、权重和精度，仅更换 bridge，
通过正常 PyTorch/HPU Graph 入口达到无 MTP 单流 90 TPS 以上，不加载外置执行器。
可复用的硬件软件基础设施是主目标；该性能实验是关键验收实例。

## 1. 找回的历史证据，及必须纠正的数字

已读取原始 `production-native-lanes-70-g/result.json`、运行命令和冻结源码，
并用 `canonical-70-b/result.json` 交叉核对。没有把 native-only ABBA 当成
bridge/native ABBA。原始 70-g 为 70 层、TP8、4K 输入、160-token greedy、
128 个计时步；历史计时名为 coordinator wall，包含所选窗口内边界/捕获成本，
不是 HTTP 完整请求吞吐。

70-g冻结执行器提交为 `827148e068d1a804e7bed6d906047561695146b5`。
本轮核对其native_backend、benchmark、C++录制/step/存储代码、两个共享库，以及
plugin runner共8个文件，全部匹配原存档 `sha256.json`；不是用后来代码推测旧实验。

| 70-g 同算子策略配对 | bridge TPS | 旧 native TPS | 输出 token |
|---|---:|---:|---|
| BF16/FP32 QKV＋FP32 norm 参照 | 31.91563 | 61.49983 | 相同 |
| 已优化策略，含 compact8，不含 full-attention post 融合 | 40.65457 | 90.29651 | 相同 |
| 上一策略加 full-attention post 融合 | 42.01230 | 90.57636 | 相同 |

canonical-70-b 同样优化策略为 40.64777/89.82601，含 full-post 为
39.71770/91.03680。它们是历史测量，不是本轮结果。

**在本轮核对的相关完整模型记录中，没有确认用户所回忆的“同策略 bridge60/native90”
精确配对。61.50 那一行本身就是 native，不能拿来证明只改执行器的 60→90 收益。**
这不否定 90 TPS 目标，也不否定框架开销；它要求重新固定 A/B/C 的真实基线。

90.29651 那一组 rank0 捕获记录包含 147 个 recipe 启动与 142 个 collective。
历史运行还打开了 `single_rpc` 和 `reuse_pages`。因此不能把整步差额全归到
`HPUGraph::replay()` 一个函数。

原始文件的 SHA、逐策略数字和当前源码锚点保存在
[机器可读审计](source-audit-20261004.json)。原始 prompt、输出、环境与二进制留本地，
不进入公开仓库。脚本仅抽取显式允许的标量字段。

## 2. 当前 HPU Graph 实际经过哪些代码

| 层 | 已核对源码 | 当前工作 |
|---|---|---|
| Python 包装 | `python_packages/habana_frameworks/torch/hpu/graphs.py`：`wrapped_hpugraph_forward`、`input_hash`、`copy_to` | 输入结构/shape/view 哈希、缓存、输入复制或 V3 重绑 |
| 捕获容器 | `backend/habana_device/HPUGraph.cpp`：`capture_begin/end`、`mark_step` | StepMarker，保存多个 SingleHPUGraph，持有 IR/Lazy Tensor/recipe 参数信息 |
| 重放入口 | 同文件：`replay`、`replayV3`、`SingleHPUGraph::replayGraph` | 逐子图处理执行状态、任务/stream、输入输出对象，调用缓存执行 |
| 缓存执行 | `habana_lazy/hpu_lazy_tensors.cpp`：`ExecuteCachedGraph` | 构造 IValue 参数栈，处理 seed 和输出 Tensor 状态 |
| launcher | `habana_lazy/hlexec.cpp`：`HlExec::Launch` | 计算符号形状/布局 hash，创建 launcher 并运行 |
| recipe 与资源 | `backend/kernel/hpu_habana_cache.cpp`：`RecipeLauncher::Launch` | recipe、workspace、Tensor owners、事件、资源回收、collective |
| 物理提交 | `backend/synapse_helpers/graph.cpp`：`graph::launch` | workspace、地址锁定/转换、最终 Synapse launch |
| 通信 | `backend/helpers/collective_kernel_info.cpp`、`habana_kernels/hccl_kernels.cpp` | 持有 collective 描述，选择通信 stream，依赖事件，地址锁，任务线程/future，HCCL |

HPU Graph 缓存命中不是重跑 Python forward，也不应被描述为每次重新编译。
可削减的是在已固定计算计划上反复做的框架管理。Tensor/view、RNG、跨 stream 和
生命周期管理本身具有语义，必须转移或预计算，不能直接跳过。

## 3. 为什么能正式集成，不需要继续拦截

`RecipeLauncher` 已持有 `shared_ptr<recipe_handle>`、workspace 大小以及
`CollectiveKernelInfos`；`Launch` 参数直接带输入、中间、输出和 DMA Tensor。
bridge 内能直接建立执行描述及资源引用，不必借旧代码的私有 mangled symbol 截获。

`HPUGraph::mark_step` 已将一次捕获组织成有顺序的子图列表；这是将编译后的描述
交给一个计划 builder 的自然入口。新执行核心应隶属于 HPUGraph 的生命周期，
不能另建 Python 模型包装器和第二份外部 shape 缓存。

旧执行器的顺序提交循环已经在真实模型证明：相同 recipe 和 collective 可以通过
更短的主机路径运行。新 runtime 已具备静态绑定检查、明确读写资源、完成生命周期、
独立 recipe 持有等可复用实现。因此核心技术不需要重新发明。

但导出点不能只有 `RecipeLauncher::Launch`：
`hpu_habana_launch_op_pt.cpp` 在 `dry_run_` 时跳过 Launch，部分绑定 patch 也跳过。
`disable_tensor_cache` 的默认 dry-run 路径必须通过编译完成/物化接口导出计划，
或明确延迟到首次真实执行完成实例化。不能让首次运行重复更新 KV/原地输出。

## 4. 三个不能原样照搬的地方

### 4.1 设备指针不是普通固定裸地址

`graph::launch` 使用 `device.lock_addresses()`，将框架地址转成锁定的物理地址。
通信路径也调用 `deviceCtxt->lock_address()`。`GenericResourceHolder` 保留地址锁
直到异步工作结束。**只保留 at::Tensor 引用，不足以证明物理地址不会改变。**

计划应记录 storage 身份、offset 和 binding slot，通过 bridge 分配器取得受保护的
物理地址；重绑/重定位时更新。禁止将首次 SDK capture 的裸物理指针永久缓存。
长期固定所有内存可能增加碎片，不能作为未经量化的默认策略。

### 4.2 通信必须与计算一并进入计划

只绕过 `ExecuteCachedGraph` 而仍逐次走 HCCL Tensor 搜索、future/线程调度及
producer 登记，无法继承旧执行器全部效果。应从 `CollectiveKernelInfos::Info` 和
具体 `RunCollective` 的参数建立 typed collective 描述，保留 communicator 所有权。

跨 stream、外部事件和分布式顺序必须显式表达。先实现经过依赖检查的单 stream
静态区域，未知依赖在提交前回退；不能复制旧实现“过滤事件”当成通用正确算法。
一旦提交部分 collective，失败必须终止该计划，禁止重新执行整段 fallback。

### 4.3 bridge 管运行资源，不能再叠一套外部资源系统

内存通过框架分配器，stream 使用框架映射，完成状态接回已有事件/资源回收机制。
同进程 recipe 可以保留 bridge 的 shared owner，不需要为每次捕获强制落盘克隆。
复用新 runtime 的资源契约和检查；不原样引入其默认 H2D/D2H INPUT/OUTPUT、
独立 Synapse arena、固定外部裸地址以及每子图一张独立 ticket。

## 5. 90 TPS 的实际不确定性

冻结的 `native_backend.py` 在捕获时调用原来的 `sample_tokens`，之后命中路径改为
`native_step_bound` 并直接组织结果；`single_rpc` 还改变 execute/sample 的 RPC 组织。
原 plugin 的 `sample_tokens` 包含准备输入、model forward、采样及 CPU 返回处理。

因此：

- bridge 计划能消除区域内的重复框架管理；
- 它不能自动消除 vLLM 请求调度、RPC、Python 输入准备或图外采样代码；
- 不能把这些模型/服务语义塞进 bridge，来伪装“仅替换 bridge”成功。

实施时必须分别量度：图 replay、worker 完整步、EngineCore 和用户 token 到达。
主机 enqueue 与设备执行重叠，不把两者相加；各阶段分位数也不相加。
若只换 bridge 仍被区域外工作限制，保持 90 TPS 目标未完成，报告剩余差额与位置；
不要偷偷换 runner，也不要把设备吞吐改名为端到端 TPS。

## 6. 构建与兼容性状态

源码提供 standalone 构建流程；`backend/CMakeLists.txt` 将 HPUGraph 编入
`habana_pytorch_backend`。根 CMake 要求 **CMake >=4.0、C++17**。
应固定 torch、Python、Synapse/HCCL、compiler、C++ ABI 和第三方依赖，先构建
未修改上游 wheel 并在隔离环境导入；README 的“最新”下载不能作为版本锁。

本轮没有构建 bridge，也没有上卡。旧控制连接/非交互 key SSH 检查未通过认证，
因此没有验证当前远端工具链。可行性结论来自本地源码与已归档实验，不能标为
“fork 已编译通过”或“新 bridge 已获得收益”。构建环境验证是计划第一个阶段。

## 7. 范围与退出条件

只建设一个 bridge 内执行核心。PyTorch 捕获与显式 Synapse producer 是输入来源，
不是两个运行时。gaudi-kernels 继续提供优化算子，不作为另一套 serving executor。
框架普遍语义不支持的分支可使用原普通算子执行路径，但不能继续部署外置 recorder。

第一阶段覆盖推理、静态 bucket、正常 Tensor 输入输出、TP collective；不宣称立即
覆盖任意动态图、autograd、任意跨线程 capture 或 MTP 请求生命周期。
后续模型、batch、prefill 和 DFlash 应使用相同核心，不再出现模型专用执行器。

实施顺序和硬验收见 [PLAN.zh-CN.md](PLAN.zh-CN.md)。
