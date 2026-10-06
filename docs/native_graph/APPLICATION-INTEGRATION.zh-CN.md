# 应用接入与跨模型迁移

目标：优化封装在 bridge，应用可以通过少量明确声明使用它们。模型迁移不复制
执行器，不依赖 MiMo 类名、层数、KV 页长或专家数量。本文描述已有入口及待收敛
契约，不宣称任意新模型或任意模式已经获得性能收益。

## 优先复用已有 API

1. 用 `wrap_in_hpu_graph_func` / `wrap_in_hpu_graph` 表达可捕获的计算；确需手动
   生命周期控制时用 `HPUGraph`。不要再创建模型专用的 graph runtime。
2. 用普通 Tensor 表达可变输入。稳定输入可以更新内容，不能将变化值隐藏在
   未进入绑定/缓存契约的 Python 全局状态里。
3. 用普通 copy 或 `_foreach_copy_` 表达传输。后者由 bridge 根据实际类型、
   布局、别名和大小准入；应用不自己调用 DMA 或补造完成事件。
4. cache/reset、重绑与 stream 依赖交给框架。模型权重/KV 缓冲被替换时，上层
   必须通知相应图生命周期；不能仅因数据指针碰巧相同就复用旧计划。

## 一个与模型名称无关的接入形状

以下示例说明已有 API 的组合方式，不是本轮新增设备验收或性能结果。
它使用两个独立的连续 BF16 输入，便于展示批量上传；不涵盖动态形状和并发。

```python
import torch
import habana_frameworks.torch as ht

# model 是已有、已放到 HPU 的纯 Tensor 推理模块；例如输出同 shape 的投影。
# h_x 和 h_residual 是本次调用提供的 CPU Tensor，dtype/shape 与目标一致。
with torch.inference_mode():
    x = h_x.to("hpu")
    residual = h_residual.to("hpu")
    ht.core.mark_step()
    ht.hpu.synchronize()

    def tensor_program(x, residual):
        return model(x) + residual

    run = ht.hpu.wrap_in_hpu_graph_func(
        tensor_program, disable_tensor_cache=False, asynchronous=False
    )
    run(x, residual)  # 首次捕获；shape、dtype 和计算范围确定。
    ht.hpu.synchronize()

    # 新输入来自应用，不由 bridge 猜测。源在上传完成前不能被覆写。
    torch._foreach_copy_((x, residual), (next_x, next_residual), non_blocking=True)
    ht.core.mark_step()
    result = run(x, residual)
    # 用普通 Torch 操作消费 result；转 CPU 会等待所需设备结果。
    result_cpu = result.cpu()
```

`asynchronous=False` 不是“所有设备计算已经完成”的同义词。完成计时需要显式
同步或真实输出消费。实际应用的 host staging 生命周期也必须覆盖异步传输。
示例不承诺任意 `model` 都可捕获；随机数、数据依赖控制流和状态修改仍需遵守
原 API 的能力范围。适用的快路径自动执行，应用不选择内部 executor。

## vLLM 应怎样迁移到其他模型

当前 `serving/decode_graph.py` / `decode_inputs.py` 仍有 runner 私有字段和受限
的 greedy/unit-query 契约，是需要收敛的集成实现，不能称为已完成的通用接口。

后续应在同一个 vLLM 适配位置完成三件事：

- 将 scheduler 结果写入原输入 Tensor；请求、页表或布局改变时触发原 builder。
  该逻辑理解 KV，但 bridge 不理解 KV。
- 声明本次可以捕获的计算范围，例如 forward、logits 和设备可执行的 sampler。
  sampling 是否有 CPU 状态依赖由 vLLM 决定，bridge 不能猜测。
- 将状态重新分配、请求布局切换等生命周期变化接到已有图缓存失效机制。

新模型复用同一适配。如果换模型仍需复制每步执行/上传/提交循环，说明抽象
尚未完成。若模型引入新的算子或状态语义，应扩展相应算子或上层契约，不往
执行核心添加模型名字判断。

prefill、decode、verify 共用底层绑定/计划/提交能力；它们上层的 shape、状态
推进和采样事务不同，不强行套成同一个 unit-query 快路径。尚未适配的模式
继续原语义，单独记录覆盖和开销。

## 验收与归因

- 框架兼容：相同应用、相同配置、只换 bridge，输出与生命周期正确。
- 接入收益：启用公共能力前后配对，集成改动及收益单独记录。
- 迁移能力：至少第二个不同结构模型沿用同一接口，不改执行核心、不复制
  runner 热路径；测实际吞吐与回退比例，不以接口可调用代替收益。
- 维护性：唯一资源/提交实现，无新私有执行包、模型专属开关或裸地址缓存。

91.02 TPS 是 MiMo 当前组合方案的成绩，不是上述跨模型验收已经完成的证据。
