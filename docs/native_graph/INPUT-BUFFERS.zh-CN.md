# 通用输入存储与覆盖诊断

`habana_frameworks.torch.hpu.InputBufferCache` 是推理输入存储工具，不拥有 graph、
stream、allocator 或 SDK 提交。应用通过原 HPUGraph/wrap API 使用同一个 bridge
执行核心，通过普通 Tensor copy/foreach 表达输入变化。

缓存按名字、shape、dtype 区分桶，目标 device 在构造时确定。默认最多保留
128 项，按 LRU 淘汰；图可继续持有自己的 Tensor 引用。不同名字有独立分配。
源可以非连续，返回目的存储连续。只接纳普通 strided Tensor，调用必须处于
no_grad/inference_mode。copy 错误直接传播，新桶在 copy 被接受后才发布。

```python
with torch.inference_mode():
    inputs = ht.hpu.InputBufferCache("hpu", max_entries=8)
    run = ht.hpu.wrap_in_hpu_graph_func(tensor_function)
    x = inputs.copy("x", host_x)
    ht.core.mark_step()
    output = run(x)
```

不要 resize、transpose_、修改返回目的 Tensor 的 dtype/device 或 set_ 重绑
它的存储。检测到布局改变会拒绝更新；该检查不替代 graph 的绑定/地址校验。
需要重绑时 clear 输入缓存并通过原 wrapper/输入重绑接口更新相应图。手动固定
绑定的 HPUGraph 不会因为输入缓存重新分配而自动指向新 Tensor。

源 host Tensor 必须保持有效直到非阻塞传输完成；clear 不同步设备，仅释放
本工具持有的引用。在途所有权、WAR/RAW 和完成回收仍由原 Torch/bridge 负责。
并发独立输入生产者分别持有缓存，不在本工具中另造 flight pool。

`HPUGraph.execution_info()` 返回当前计划、prepared/replayed 命令、通信覆盖及
拒绝原因的快照。它不读取 Tensor 值，不是执行入口，放在计时外使用。ready
不等于全部命令进入 prepared 路径；必须读取实际覆盖计数。

该代码及通用 vLLM 集成的最终设备验收受远端 SSH 中断阻塞。初版 API 构建、
CPU 语义和第一轮 MiMo 回归已运行；新增布局防护通过 5 项本地 CPU 测试。
当前完整源码和上卡待办保存于本地交付资产，不将未跑的第二模型/生产门记为通过。
