# 捕获计划与调用资源分离

本次整理属于 bridge 执行核心。它不识别模型、token、专家数或 KV，不引入另一个执行器。

## 资源与状态

`NativeExecutionPlan` 的生命周期显式区分 Capturing、Ready、Rejected、Failed。执行策略独立保存，诊断接口将状态与策略投影到兼容的统计字段。普通 replay 的准入检查不再复制包含字符串/向量的完整统计对象。

临时异步调用、replayV2/V3 或显式输入调用选择 Lazy adapter，不直接销毁捕获计划。下一次尝试固定地址执行时仍检查当前边界、Storage、offset、shape、stride、dtype 和全部保留的绑定。如果 adapter 实际更换了绑定，则退役快路径，普通 replay 继续走 adapter。部分提交失败进入 Failed，后续禁止 replay；调用 reset 才能开始新的捕获。任何拒绝接口都不能把 Failed 变回可执行状态。

`execution_info()` 新增 `adapter_replays`、`last_replay_native` 和 `last_replay_reason`。`ready=True` 表示捕获计划仍可尝试，不表示上一次调用一定使用了它；实际覆盖应看执行计数和 last 字段。

## 不可变程序与调用实例

Prepared 程序保存 recipe、collective 描述、tensor 元数据、抽象地址映射与依赖。物理地址填在调用实例 `Frame` 的 tensor 描述中，不再改写程序模板。

同一次 `replay_many(N)` 为每个使用到的 prepared region 创建一个 Frame，重复 N 次共用地址锁、物理绑定和资源所有权。workspace 在创建第一个 Frame 前按整个计划的最大需求保留，避免后续普通 launcher 扩容。Frame 保留独立 Storage 和 active recipe 生命周期，提交结束不等于释放设备资源。

外部依赖等待仍在每次 region 入口执行，通信流 FIFO 入口/出口不变。完全 prepared 的连续区间可以延后一次发布；遇到普通 launcher 必须先发布此前的写入，因为普通 launcher 可能向其他通信流提交。每次发布的完成回调持有 Frame；失败路径先发布或 drain 已提交工作，然后传播错误并 poison 计划。因此不宣称任意混合图都能把 publication 缩成一次。

新增计数 `prepared_frames`、`prepared_publications`，用于分别验证实例复用和安全边界处的发布次数。仅用于捕获准入的输入快照在 seal 后释放，运行期仍保留独立的 canonical leases。

## 设备退出顺序

回归复现了一个旧问题：Python 全局 graph 持有 recipe，进程退出时设备先关闭垃圾回收线程，graph 随后的 recipe 析构再向已关闭线程投递，造成 abort。旧 fb703 基线和第一次整理候选均在同一组五个 graph 测试通过后出现此错误。

设备上下文现在通过弱引用登记 `DeviceResource`。设备 Finish 在清空 recipe cache、关闭垃圾回收线程前通知仍然存活的资源释放 SDK 所有权。Native plan 的调用实例继续独立持有已经提交的资源；这里只移除捕获计划本身的长期持有。注册表不延长 graph 的寿命，释放回调在注册表锁外运行，设备关闭后禁止新注册。Failed 状态不会因退出释放而恢复可执行。

这项修复必须通过不调用 graph.reset 的进程退出回归，以及多卡和完整模型验收；不能仅以测试断言通过作为进程退出成功。

## 重复执行范围

`replay_many` 接受可转换为 C++ `size_t` 的正整数。旧的 16 次是实验范围，不是设备能力上限，已从实现中去除。调用者负责限制单次工作量、输出历史容量以及服务取消延迟。SDK 仍逐条提交且可能因队列容量阻塞；此接口不承诺设备硬件循环，也不自动实现 KV、EOS 或输出反馈。

## 验收

代码提交不是资格认证。需要独立安装候选 wheel，验证静态/变绑定、临时 adapter、重复执行、EOS、输出历史，以及多卡 collective/资源释放，再做完整模型回归。不能把基础设施整理的收益当成已经交付的 DFlash 服务收益。
