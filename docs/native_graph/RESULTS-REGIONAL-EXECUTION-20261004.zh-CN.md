# Bridge 区域依赖与通信 stream：首轮实现和实验

本地日期 2026-10-04，远端日志为 UTC+8 的 10-05。
这是 [区域执行计划](PLAN-REGIONAL-EXECUTION-20261004.zh-CN.md) 的 A/B 检查点。
完整70层已验证执行等价，**没有取得可确认整模收益，90 TPS 未达到**。
不是 C–F 全部完成，也不代表外置执行器可以退役。

## 实现

沿用 `HPUGraph` 拥有的 `NativeExecutionPlan`，没有新 serving executor、allocator
或线程池。捕获时从 recipe 的实际 launch Tensor 元数据取得 compute 读写；
all-reduce 是原地读写，all-gather 区分输入与输出。以保守 allocation key 建立
RAW/WAR/WAW 和 HCCL 队列顺序边；输入的使用结束登记不再被误当成计算写输出。

只有相邻 compute→collective→compute 两边均存在真实 RAW、访问完整、没有
partial/external event 的 collective 才是合流候选。未知访问、混合 compute/HCCL
命令、未入选通信和 communicator 变化切断区域。区域至少含两次通信，以摊薄
入口/出口保护。独立分支和单通信节点保留原 stream 策略。

区域内 replay snapshot 临时借用 bridge compute stream，入口等待 communicator
原 stream 已有工作，出口将完成顺序接回原 FIFO。没有修改 ProcessGroup 全局
stream；保留地址锁、Storage/recipe/communicator 保活和逐命令回收。区域内仍逐条
SDK launch，因此还没有完成区域资源 frame 或统一底层提交。

`native_dependency_diagnostics` 是捕获后的只读诊断，列出命令种类、访问完整性和
缺项，不读取 Tensor 数值，不进入计时 replay。已有 serving 诊断 RPC 可选读取它。

## 找到并修正的问题

1. 新依赖分析初版将 `UINT64_MAX` Tensor ID 视为缺失的真实访问。两层模型发现
   B1 有4个、B2有5个不完整命令。诊断逐项确认全为该 sentinel；它们不在
   `synTensorRetrieveLaunchIds` 返回的实际 launch 集合里。修正只忽略无效 sentinel
   的依赖，不删除 SDK 提交参数；仍核对每个实际 recipe Tensor 有非零绑定、
   有效 ID 有方向及唯一 retained allocation。未知有效 ID 继续阻止优化。
2. 初版整图要求所有通信均合流，混合 compute/HCCL 节点使整个模型无法入选。
   改为在明确边界之间生成最大可证明区域，不绕过这些边界。
3. 单通信节点合流在 A/B/B/A 中延迟增加15–17%，增加的两次 FIFO event record
   无法摊薄。已从默认合流中排除，不能把该初版作为发布候选。
4. 补强硬件验收发现 TP2 探针 CPU/PCI 正确、内存策略仍是默认0。复用现有 PCI
   placement 计划，以 numactl 在后端 autoload 前绑定 CPU和内存；重新跑四臂。
   旧失败 `dependency-a2` 保留，未计入合格计时。完整模型 worker 原绑定不受影响。

## 验证

- A/B/C/D 使用32线程构建，耗时分别533.57/537.68/460.34/52.35秒，全部成功。
  Torch、SDK、权重和计算 kernel 不变；使用独立安装环境。
- 最新 D 的19项 C++ 测试通过，包括独立读、RAW/WAR/WAW、原地写、重复别名、
  partial/未知访问、混合命令、communicator 分区和单节点不合流。
- 独立程序检查2000张随机访问图、10万次合法拓扑执行，读到的写入版本与最终
  写入均保持一致。这验证依赖算法，不代替真实 metadata/设备验收。
- TP2 FP32/BF16、变化输入、正常/混合 fallback、纯通信、8/32段链、独立消费者、
  两个 communicator 全部通过。四臂共1024次数值比较。
- 原 communicator 上的普通通信包围 replay、图外 stream 消费输出通过。
- 对端不提交通信时 reset graph 并删除最后输出 Tensor，弱 Storage 仍存活；
  完成后观察到释放。覆盖纯通信和实际合流的两通信区域、FP32/BF16。
  这不是任意并发、异常或内存压力的全面生命周期证明。
- 合格 TP2 实际 PCI 为 module5/accel6/PCIcd 和 module4/accel7/PCIcc，所有线程
  本地 CPU，内存策略2。未只相信环境变量或 hl-smi index。

## 两卡计时

`dependency-a3 / regions-b1 / regions-b2 / dependency-a4`，同验证源码11577e5，
分别独立进程，每点128次热 replay，计时不启用 profiler。使用两 rank 的较大
完成时间，分别平均两个 A 和两个 B；不是同进程切换 ABBA。

| 图 | FP32 延迟下降 | BF16 延迟下降 |
|---|---:|---:|
| 8段依赖链 | 0.44% | 0.93% |
| 32段依赖链 | 3.00% | 3.64% |
| 单 all-reduce | −15.09% | −16.02% |
| 单 all-gather | −16.58% | −16.63% |

单节点回退的最新 D 功能复测通过；一次 D 的32段链完成时间1432.7/1439.4µs，
说明小收益存在运行波动，不能承诺稳定比例，更不能乘到模型 TPS。
这组是热小图的提交/完成成本，不是 HBM、MoE带宽或完整 LLM 推理吞吐。

## 真实模型预检

相同 MiMo 两层、TP8、4K、B1/B2，每点两轮128 token。A/B/C/D 的768个 token
及六条输出序列一致；每臂8 rank、两份 bucket 计划均正常。属于执行等价检查，
不证明完整70层语言质量，也不使用短模型 TPS 代替目标。

最新 D 访问缺项为0，11命令/5通信中4次被分为两个合流区域；混合命令保留
原路径。每份计划的 coalesced_replays 等于 replays，证明并非只创建未执行的描述。

## 完整70层验收

同应用源码9ef501c，A=2aa4c1a02，D=77d849027，70层、TP8、4K、无MTP、B1/B2、
每点两轮128 token，fresh-process A/D 对。计时不启用 profiler；诊断另跑请求。

| 指标 | A | D | TPS相对变化 |
|---|---:|---:|---:|
| B1 round0 | 76.427 | 76.042 | −0.50% |
| B1 round1 | 76.601 | 75.972 | −0.82% |
| B2 round0，聚合 | 118.200 | 118.922 | +0.61% |
| B2 round1，聚合 | 118.561 | 118.618 | +0.05% |

768 token、六条完整输出序列完全相同。8 rank 的两份 bucket 计划均 ready，
284命令/141通信，访问缺项0，**140次通信确实合流**；coalesced_replays 等于 replays。
一个混合 compute/HCCL 命令仍保留原路径。全部 worker 实际 PCI、模块、线程
CPU本地性验收通过，未加载外置执行器。覆盖通过不等于独立语言质量证明。

整请求吞吐 B1 为21.95 TPS，B2聚合23.74–23.78 tokens/s，包含 prefill；不能用
resident76替代完整请求吞吐。此配对不能证明稳定的亚百分比变化，没有90 TPS收益。

轻量 host 诊断另跑55个稳态步、没有任何 profiler。rank0 的 model.forward
8.571→8.305ms，而输出 cpu() 2.189→2.569ms；sample_tokens 12.817→12.832ms。
rank4 更明显：forward 9.976→8.483ms，cpu() 0.808→2.423ms，完整采样窗口基本不变。
这是**主机更早返回、等待移到了后面的同步点**的证据；嵌套中位数不能相加，
cpu() 包含尚未完成的设备工作，不是2.6ms纯拷贝或CPU税。

因此单独合流没有复现旧90。不能因为我们删掉了跨 stream wait，就给这些 wait
赋予未经测量的毫秒收益。当前候选仍逐命令管理 producer、完成 event、资源与
地址/workspace；下一阶段应减少真实 SDK 完成事件/重复管理并验证关键路径，
不能只优化 enqueue 的返回时间。正常共享 Torch 安装未更换，候选仍在隔离环境。

## 尚欠与资产

下一核心仍是 C/D：以执行区域管理地址/workspace/Storage及回收，减少内部逐命令
事件管理和通用 launcher 重复工作。当前合流仍保留这些成本，不能当作完整移植。
保活回收和对外 readiness 必须分开；输入最后使用、early output、跨流访问都要
保留必要的发布点。预定 workspace 再提交，避免末尾才释放的资源和中途增长互等。
再补公共重绑/异步 API，最后扩大正规捕获范围。bridge 不接入 token/KV页业务。

两臂 supervisor exit0、remaining_owned_processes=[]，后续核查8卡768MiB/0%。
退出依然有 HPUDevice 引用警告、worker SIGTERM 和 EngineCore 强制收尾。
设备释放与优雅退出分别验收，不能以进程 exit0 宣称生命周期债务已消失。

用户“所有资产和成果留本地”的要求持续执行：
`gaudi-kernels/artifacts/builds/bridge-regional-20261004/`。
包含全部 Git bundle、验证源码、wheel及SHA、构建日志、失败、设备/NUMA快照、
随机检查、原始请求结果和分析。代码在本地 Git 提交，未据此宣称已推送。
