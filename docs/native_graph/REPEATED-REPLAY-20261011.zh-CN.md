# 同一份图的有界重复提交

`HPUGraph.replay_many(n)` 在同一个 PyTorch bridge 内提交一份捕获程序的 1–16 次执行，不复制 N 份程序，也不引入其他执行器。它复用 `NativeExecutionPlan`、RecipeLauncher、HPU allocator/event map 和既有 HCCL 队列。

```python
import torch
import habana_frameworks.torch as ht

with torch.inference_mode():
    state = torch.tensor([0], dtype=torch.int64, device="hpu")
    ht.core.mark_step()
    graph = ht.hpu.HPUGraph()
    graph.capture_begin()
    state.copy_(state + 1)
    ht.core.mark_step()
    graph.capture_end()
    ht.hpu.synchronize()

    # Capture executes the program, so restore its runtime state afterwards.
    state.copy_(torch.tensor([0], dtype=torch.int64, device="hpu"))
    ht.core.mark_step()
    graph.replay_many(16)
    assert state.cpu().item() == 16
    graph.reset()
```

## 执行合同

- 输入/输出/中间存储必须满足原生计划的固定地址合同。值可以在设备上更新；程序必须自己把反馈写回持久输入缓冲。
- 多次执行需要保留的输出历史由捕获程序写入。API 不会自动复制每次输出、连接模型的采样输出与输入，或预留 KV 页。
- 入口一次检查捕获边界和存储租约；包含 collective 的 N 次执行只入队一个 HCCL job。程序内部每次执行仍沿用既有的依赖、地址锁、完成回调和存储保活。
- CPU 等待这个 job 完成 SDK 提交，保证引用、错误及队列顺序的原合同；设备完成仍是异步的。SDK 队列满时 CPU 会受背压，并继续供给后续命令。这不是一条 SDK 命令，也不是硬件循环或完全设备自治。
- 不支持的绑定在原生提交前抛错，不静默改用另一条测量路径。普通 `replay()` 继续通过既有 Lazy adapter 处理需要重绑定的图。
- 任意一段提交失败后，已有的计划 poisoning 禁止重试整个块，避免已提交的前缀执行两次。重置前须遵守现有完成/存储生命周期合同。

`execution_info()['replays']` 计 SDK 提交成功的程序执行次数；`queued_replays` 计入 HCCL job 次数。因此一次 `replay_many(16)` 的增量应分别为 16 和 1（有 collective 时）。这些计数不是设备完成事件。`prepared_region_replays` 和 `prepared_replayed_commands` 保留逐次执行统计。

## 捕获期存储切换的正确性保护

旧检查只在 capture_end 保存最终 Lazy 边界，再检查它在 replay 前是否改变。它漏掉了“捕获期间已经改变”的情况：INT64 `add_` 会产生不同的物理输入/输出存储，原生计划仍反复读初始输入，得到 127、127、127；普通 replay 则重新绑定，得到 127、128、129。

现在每条 recipe 记录捕获时输入的 Storage owner、地址、offset、dtype、shape、stride；capture_end 与同一 recipe 的最终 Lazy 输入逐项比较。不能满足固定地址合同的捕获被拒绝，普通 replay 使用原有正确路径。保护基于存储合同，不按模型、dtype 或算子名称分支。INT64 只读输入和明确 `copy_` 写回仍可原生执行。

此修复消除了原生路径静默算错，但没有让换存储的原地更新获得原生加速；后者需要单独的重绑定适配与 alias 验证，不能将安全回退包装成已经完成该优化。

## 验证与模型接入

回归位于 `tests/pytest_working/lazy/test_native_graph_mutation.py`，覆盖重复状态更新、只读输入、显式写回、连续块、EOS 掩码、调用计数和提交前拒绝。多卡验证及证据在 gaudi-kernels 的 `tools/validation/executor/bounded_feedback_probe.py --repeat-plan`。

真实 LLM 还必须由上层提供一段有界的调度/KV lease，连接采样 token、position、attention usage、slot mapping 与下一步输入，并返回真实有效长度。EOS 掩码能维持状态正确，不能使固定图尾部自动停止执行。重复一个仍由 CPU 上传下一 token 的现有图，不会自动变成正确的自回归生成。
