# 旧SDK执行器与bridge的同进程差分

后续实现已完成[完整 step 对照](RESULTS-COMPLETE-STEP-20261005.zh-CN.md)：扩大
范围与输入复用只回收 70 层约 0.24–0.32ms；两层则约 1.1–1.2ms。本文的
差分事实保留，但不能再把“执行范围不同”当作全部 2.8–2.9ms 的已证根因。

## 结论

同一驻留模型中重新切换两条路径，旧SDK执行器仍能超过90TPS；当前bridge仍慢。
问题不是旧结果已经失效，也没有证据指向算术kernel回退。当前迁移覆盖forward
的计划提交，旧路径则固化了输入搬运、forward、logits、采样到结果返回的完整链。
前几轮只围绕图内提交/stream/event做优化，没有覆盖这个执行范围差异。

重要证据：70层的差值约2.8–2.9ms，缩成2层、保持4K上下文后仍约2.3ms；单独
切换RPC合并仅约0.05–0.06ms。因此应优先处理每token稳定执行链仍被拆开的
固定成本，不能继续把主要缺口预记到每层通信换stream、事件数或kernel内环。

这是执行范围与开销层次的定位，不是每一微秒的最终归因。特别是两层与70层的
差值不能相减后精确命名为某个函数成本；CPU提交与设备工作存在重叠。

## 对照条件

- 在同一个LLM实例中进行bridge/native/native/bridge，权重只加载一次。
- 两路径同一Torch、bridge wheel 77d849027、SDK、优化kernel与typed策略。
- retained SDK recorder在两臂都加载，仅native臂启用重放；对照不等于不含任何
  recorder插桩的发布配置，但排除了“只native加载了不同计算库”。
- 70层、TP8、B1、无MTP、4K、greedy、160输出token，预热完整位置范围后每臂
  计时32≤emitted<160，连续coordinator首尾wall，包含窗口内页边界处理。
- 分别使用此前76TPS的重复句子prompt与历史90TPS的检索问答prompt。
- 两种输入各4臂共1280输出token；每种输入的全部序列一致。全部native rank
  命中，32份捕获在hidden、selected、logits和sample边界逐位重放检查通过。
- 检索问答四臂均包含7391，但本轮不代替全面模型质量/精度验收。
- 全部rank实际PCI、module与所有CPU线程本地性通过。性能请求不开profiler，
  轻量host观察另跑请求。

## 完整70层结果

| 输入 | bridge A1 / A2 TPS | SDK native B1 / B2 TPS | 配对平均步延迟：bridge → native | 差值 |
|---|---:|---:|---:|---:|
| 重复句子 | 76.168 / 72.135 | 96.121 / 90.729 | 13.496 → 10.713ms | 2.783ms |
| 检索问答 | 71.357 / 72.158 | 90.993 / 90.340 | 13.936 → 11.030ms | 2.907ms |

平均先在等步数的wall毫秒上计算，不能算两个TPS的算术平均。重复句子的四臂
存在明显运行漂移，不能承诺96TPS稳定；相邻A1/B1与B2/A2的差值约2.726/2.840ms，
仍保留同方向的毫秒级差距。这是旧控制的收益，不是本轮bridge已获收益。

## 完整执行范围究竟缺了什么

从实际捕获/依赖诊断核对：

| 范围 | 两层：recipe / collective | 70层：recipe / collective |
|---|---:|---:|
| 当前缓存forward计划 | 7 / 5 | 144 / 141 |
| 旧完整token SDK计划 | 10 / 6 | 147 / 142 |

forward计划有混合compute/collective命令，不能用命令数减通信数估recipe数。
完整token在forward之外还有3个recipe、1次通信以及输入输出搬运。
这不意味着应该删掉这些计算，而是它们现在仍从图外普通PyTorch/Lazy路径逐步进入。

源码热路径：

1. `HPUModelRunner._prepare_inputs` 每步构造/更新调度与注意力元数据；
   `async_h2d_copy` 创建HPU临时Tensor并非阻塞上传。
2. `graphs.py::wrapped_hpugraph_forward` 做输入结构处理及固定图输入复制，然后
   进入`NativeExecutionPlan::replay`。
3. `NativeExecutionPlan` 仍逐命令调用通用`RecipeLauncher::Launch`，做地址租约、
   producer/readiness、resource holder等管理；还不是完整预准备提交。
4. `_execute_model_generic` 从forward返回后继续view、选行和`compute_logits`。
5. `_run_sampling` 在图外做sampling metadata、采样与mark_step；最后`.cpu()`返回。

旧`native_step_bound`则更新已有pinned输入描述，直接提交事先固定的整段SDK链，
一次末尾同步后读回4字节采样ID；上层只做必要请求状态提交。相同设备算术在
两条路径中通过重放/输出一致性验证，差别在执行组织，不是换了更快的MoE算法。

## 固定成本和RPC的区分实验

两层、TP8、**同样4K**，每臂64输出、相同输入与预热，在一个进程中运行：
bridge / native分RPC / native合RPC / native合RPC / native分RPC / bridge。
两个prompt均逐位通过。下表为各模式两臂平均步wall：

| 输入 | bridge | native，保留两次RPC | native，合成一次RPC |
|---|---:|---:|---:|
| 重复句子 | 3.268ms | 1.014ms | 0.958ms |
| 检索问答 | 3.304ms | 1.009ms | 0.955ms |

RPC本身仅约54–56µs。保留原execute/sample RPC分割，SDK路径仍回收约2.25–2.30ms。
所以不能把缺口主要归咎于请求调度RPC，也不能将两层的千TPS当完整模型性能。

两层另跑轻量host观察，rank0重复句子稳态55步：输入准备0.808ms（内含copy0.421），
forward0.905ms，forward连同logits/选行的parent1.299ms，采样parent0.548ms
（内含元数据0.348），返回cpu0.121ms，sample parent2.987ms。
native sample parent为0.725ms。不同层级嵌套，不能逐行相加。

70层同样观察：输入准备约0.80–0.82ms，采样parent约0.55–0.57ms，compute_logits
host约0.25ms；这些固定阶段没有随层数消失。native稳态路径不再经过这些普通
Python/Tensor调用，而是重放包含相同设备计算的计划。
70层cpu()约3ms包含尚未完成的设备工作，不是3ms纯拷贝成本。
native整token SDK enqueue约3.6–3.7ms，而bridge forward host约8.3–8.5ms；
两者覆盖不同范围且与设备重叠，不能把差值当作可直接回收的时间。

## 应当迁入什么

最终仍只保留bridge执行核心，旧控制仅用于诊断：

1. **明确完整的稳定Tensor程序边界**：输入更新/搬运、forward、logits与采样设备
   计算、输出完成。前端按公开接口声明，bridge不识别token或KV页业务字段。
2. **实例化可重复执行的绑定和资源**：固定/可变输入布局、staging、workspace、
   Storage与地址租约、collective顺序及完成发布；每步只更新变化值。
   复用已有HPUGraph与NativeExecutionPlan，不再建立外部执行器或私有allocator。
3. **原生提交覆盖整段**：稳定区域避免重复回到普通Lazy/Tensor准备路径；内部
   命令通过预准备提交适配器执行。图外合法可见性、并发与异常仍必须保留。
4. **上层保留调度语义，但减少重复准备**：页表/采样规则属于vLLM适配层，按
   bucket重用布局与buffer。模型代码可以不改；bridge不能自动跳过调用它之前
   任意Python逻辑，不能把模型专用state machine偷塞进bridge来凑指标。

前述0.28ms输入小图收益可以作为其中一个部件，但不是问题的全部答案。
同样，事件合并只是实例化资源设计的一部分，不能当作完整执行链迁移。
下一实现必须以完整步的单因素对照验收，并保留旧SDK控制作为90TPS能力见证。
尚未完成同一计算范围下的图内/图外逐项开关消融，不声称每个子项收益已定量。

## 失败与资产

preflight-a在设备加载前被runtime admission拒绝：从B1/B2/B3清单切换旧B1执行器
却未同步声明只跑B1。preflight-b修正声明后通过；没有绕过admission。
两层预检、70层、RPC隔离均由已有有界supervisor运行。退出依旧有Device引用
警告、worker SIGTERM与EngineCore强制收尾，不能宣称优雅退出问题已经解决。

源码主实验d03de23，RPC隔离deb78a7；分析脚本复用同一时钟和计时窗口，明确排除
prefill与前8个decode观察，避免把native的冷捕获child阶段误算成稳态调用。

本地资产：`gaudi-kernels/artifacts/builds/bridge-gap-20261005/`。包含源码、Git bundle、
manifest及artifact SHA、原始结果、8rank捕获命令与数值门、placement、失败与日志。
当前发布bridge未更新；本轮是找出差距的对照实验，没有把旧native的90+标为bridge成果。
