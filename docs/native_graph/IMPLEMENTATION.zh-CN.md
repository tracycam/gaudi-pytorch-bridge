# 首个 bridge 内部执行计划检查点

目标仍是应用不变、只更换 bridge。此检查点是静态计算子集原型，尚未完成
源码构建、设备验收或模型性能验收，不能据此宣称达到90TPS。

## 实现位置

- `HPUGraph` 持有 `NativeExecutionPlan`，没有新的 Python 执行器包。
- 捕获期间，`RecipeLauncher::Launch` 通过正式内部方法传递 compiled recipe、
  输入/输出/中间 Tensor 和尚未转换成物理地址的 launch bindings。
- 普通同步 `HPUGraph.replay()` 在检查当前 Lazy Tensor 的绑定后直接遍历计划，
  跳过逐子图 `ExecuteCachedGraph` 和 `HlExec::Launch`。
- 每条命令暂时仍走 bridge 的 `RecipeLauncher::Launch`，保留 workspace、
  地址锁、producer/consumer 事件及异步资源持有。它不是最终精简提交循环。

Tensor 租约保存 storage owner、分配器抽象地址、offset、dtype、shape、stride。
每次重放前检查全部租约，提交时重新通过原分配器锁定物理地址，不复用捕获时的
物理指针。以捕获 stream 执行，并恢复调用者的当前 stream。

目前只接纳静态、连续、无梯度的 HPU Tensor，以及冻结的标量参数。多 stream、
通信、外部事件、DMA 输入、动态 shape、RNG、显式输入重绑、dry-run 和输出释放
暂不接纳。检查失败发生在提交前，沿用原有行为。提交开始后异常会使计划失效，
后续任何 replay 入口必须拒绝重跑，避免重复写入。

诊断接口 `_hpu_C.native_replay_stats()` 只提供计划命令数、重放数和拒绝原因，
不提供另一个执行入口。`scripts/validate_native_graph.py --require-native` 通过
普通 API 验证两个相连 recipe、不断改变输入和普通 Torch 下游消费；它不测模型TPS。

## 构建和验收边界

先构建未修改的上游，再在独立环境中构建候选。远端 GitHub 不稳定时，在本机按
`build-dependencies.lock.json` 下载并验证，传输后再次校验，交给上游已有的
`--offline-dependencies-directory`。不要修改依赖查找逻辑或共享 Python 环境。

已安装的官方 wheel 已通过两 recipe FP32/BF16 功能探针；这不代表源码构建通过。
候选尚未经过功能探针。通信、输入生命周期和模型 A/C 仍需按 PLAN 的 P3–P5 完成。

## 应用基线发现

现有 gaudi-kernels serving launcher 无条件 preload 旧 `libe1`，bootstrap 也
无条件调用其启动接口，即使选择普通 bridge。因此 A/C 不能直接复用这个启动
路径并宣称“没有外置执行器”。必须保留同一应用和计算配置，同时先拆开计算算子
注册与执行 hook；不能用部署时文本补丁掩盖这个依赖。
