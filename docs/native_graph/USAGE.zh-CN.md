# 使用 bridge 内部编译计划

这是原 `HPUGraph` 同步重放的内部快路径，应用仍使用 Habana/PyTorch 的公开图 API。
无需导入 gaudi-kernels 的 Graph runtime 或加载旧 `libe1` 执行 hook。

当前验证环境是 Gaudi2、SDK 1.24.1 系列、Python 3.12、Torch
`2.11.0a0+git009b5f6`。候选 wheel 是 `1.24.1+git413e535ce`。
这只是已经验证的环境组合；没有宣称跨 SDK/Torch ABI 通用。

## 普通应用

在应用原来的独立环境中替换 bridge 安装，保持其他配置不变。已有
`wrap_in_hpu_graph`、`HPUGraph.capture_begin/capture_end/replay` 代码继续调用原 API。
匹配支持条件的同步图自动使用编译计划；不匹配时在提交前回退原 bridge 路径。

最小示例：

```python
import torch
import habana_frameworks.torch as ht

x = torch.ones((8, 8), device="hpu")
ht.core.mark_step()
graph = ht.hpu.HPUGraph()
with torch.inference_mode():
    graph.capture_begin()
    intermediate = x * 2 + 1
    ht.core.mark_step()
    output = intermediate * 3 + 5
    graph.capture_end()

    # 内容变化不改变分配；普通 API，不是额外的原生执行入口。
    x.copy_(torch.full_like(x, 4))
    ht.core.mark_step()
    graph.replay()
    assert torch.equal(output.cpu(), torch.full((8, 8), 32.0))

ht.hpu.synchronize()
graph.reset()
```

按已有应用需要设置 `PT_HPU_LAZY_MODE=1`。分布式图使用 bridge 已有的 HCCL 与
Lazy collective 配置，当前快路径支持 FP32/BF16 all-reduce、all-gather-out。
输入值、路由、位置等可作为固定分配中的运行时数据更新；shape 或分配改变不属于
该静态快路径的支持范围。应用的 shape bucket 与图生命周期策略仍由原框架管理。

## 显式新输入

需要用新的 Tensor 分配替换输入时，捕获期间调用 `mark_user_inputs([x])`，重放时
调用 `replay_with_inputs([new_x])` 或已有 V3 API。本轮修复了 HPU 输入不执行的 bug。
这个绑定 adapter 仍走原 bridge 执行路径，不能把它的成功当成原生快路径命中。

## 验证实际覆盖

测试用的 `_hpu_C.native_replay_stats(graph.hpu_graph)` 返回 `commands`、`collectives`、
`replays`、`queued_replays`、`ready`、`failed` 和 `reason`。这是私有诊断接口，
不要把它作为应用的执行 API。读取统计放在计时区间以外。

确认实际 loaded library 来自所选 wheel，同时检查没有旧 executor preload。
模型吞吐必须测真实 token 返回，不能把小图倍数、部分层 TPS 或 enqueue 时间
当成完整模型效果。运行结果详见 [IMPLEMENTATION.zh-CN.md](IMPLEMENTATION.zh-CN.md)。

## 构建与复现

沿用 bridge 的 `.devops/build.py`；本轮统一使用 `-j 32`。上游/候选各自使用独立
build root 和验证 venv，复用固定 Torch，不修改共享 SDK。依赖下载按
`build-dependencies.lock.json` 固定 commit/SHA，离线依赖走上游已有入口。

计算/绑定探针在 `scripts/validate_native_graph*.py`。分布式与整模型对照在
gaudi-kernels 的 `tools/validation/executor/bridge_collective_test.py` 和
`bridge_model_test.py`；这些是验证工具，不是必须安装的第二套执行器。

目前完整 70 层首轮对照是 B1 约 49 → 75 TPS，B2 聚合约 95 → 119 tokens/s。
90 TPS 尚未达成；训练/RNG、dynamic shape、dry-run materialization、async replay
等仍属于明确回退范围。外置执行器的弃用应等待完整性能与生命周期验收。
