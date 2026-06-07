# Qwen3 量化详解

> 以 [quantize_qwen3.py](quantize_qwen3.py) (Qwen3-0.6B, W4A16, GEMM) 为例，从开发者视角逐行追踪 AWQ 量化完整调用链。

---

## 0. 全局调用链

量化一个模型的完整过程只有三行核心调用，但内部链路非常深：

```
AutoAWQForCausalLM.from_pretrained(model_path)    # → 加载 FP16 模型
model.quantize(tokenizer, quant_config, calib_data) # → 逐层搜索 scale → clip → INT4 打包
model.save_quantized(save_path)                     # → 保存量化权重
```

展开后分 6 个阶段：

| 阶段 | 入口方法 | 核心文件 | 工作 |
|---|---|---|---|
| 0 | `from_pretrained()` | [auto.py](awq/models/auto.py) → [base.py](awq/models/base.py) | 加载 HuggingFace FP16 模型并包装 |
| 0.5 | `AwqQuantizer.__init__()` | [quantizer.py](awq/quantize/quantizer.py) | 校准数据准备 + Catcher 捕获首层输入 |
| 1 | `_get_input_feat()` | [quantizer.py](awq/quantize/quantizer.py) | Forward hook 捕获每个 Linear 的输入激活 |
| 2 | `_search_best_scale()` | [quantizer.py](awq/quantize/quantizer.py) + [scale.py](awq/quantize/scale.py) | 网格搜索最优 per-channel scale 并注入 |
| 3 | `_search_best_clip()` | [quantizer.py](awq/quantize/quantizer.py) + [scale.py](awq/quantize/scale.py) | 搜索最优 weight clipping 阈值 |
| 4 | `_apply_quant()` | [quantizer.py](awq/quantize/quantizer.py) + [gemm.py](awq/modules/linear/gemm.py) | 真实 INT4 打包，替换 nn.Linear |
| 5 | `save_quantized()` | [base.py](awq/models/base.py) | 保存 safetensors 量化权重 |

---

## 1. 阶段0：加载 FP16 模型

### 1.1 入口

```python
# quantize_qwen3.py 第 18-24 行
model = AutoAWQForCausalLM.from_pretrained(
    model_path,          # "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    trust_remote_code=True,
    safetensors=True,
    device_map="auto",
)
```

### 1.2 调用链追踪

**第一步**：`AutoAWQForCausalLM.from_pretrained()` — [auto.py:68-94](awq/models/auto.py#L68-L94)

```python
@classmethod
def from_pretrained(self, model_path, ...):
    # 1. 读取 config.json，获取 model_type
    model_type = check_and_get_model_type(model_path, trust_remote_code)
    # → Qwen3-0.6B 的 config 中 model_type = "qwen3"

    # 2. 查表 dispatch 到具体模型类
    return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(...)
    #   AWQ_CAUSAL_LM_MODEL_MAP["qwen3"] = Qwen3AWQForCausalLM
    #   即调用 Qwen3AWQForCausalLM.from_pretrained()
    #   由于 Qwen3AWQForCausalLM 没有重写这个方法，
    #   实际执行的是 BaseAWQForCausalLM.from_pretrained()
```

**第二步**：`BaseAWQForCausalLM.from_pretrained()` — [base.py:321-407](awq/models/base.py#L321-L407)

```python
@classmethod
def from_pretrained(self, model_path, model_type, ...):
    # 1. 定位模型权重文件路径 + 加载 config
    model_weights_path, config, quant_config = self._load_config(
        self, model_path, "", safetensors, trust_remote_code
    )
    # model_weights_path = 本地路径 (已经是本地，跳过下载)
    # config = 标准 HF Qwen3Config
    # quant_config = AwqConfig (从 config.json 中读 quantization_config，此处为空)

    # 2. 确定用哪个 HuggingFace AutoModel
    target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
    # → "qwen3" → "AutoModelForCausalLM"
    target_cls = getattr(transformers, target_cls_name)
    # → transformers.AutoModelForCausalLM

    # 3. 用标准 HF 方法加载 FP16 模型
    model = target_cls.from_pretrained(
        model_weights_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,    # 半精度
        use_safetensors=True,
        device_map="auto",            # 自动多 GPU 分配
    )
    model.eval()

    # 4. 包装为 AutoAWQ 的模型类
    return self(
        model,                       # 内嵌的 HF Qwen3ForCausalLM
        model_type="qwen3",
        is_quantized=False,          # 标记：尚未量化
        config=config,
        quant_config=quant_config,   # 空 AwqConfig
        processor=None,
    )
```

### 1.3 结果

```
Qwen3AWQForCausalLM
├── .model        → Qwen3ForCausalLM (完整 FP16 权重)
│   ├── .model.embed_tokens
│   ├── .model.rotary_emb
│   ├── .model.layers[0..N]   ← 每个是 Qwen3DecoderLayer
│   │   ├── input_layernorm        (Qwen3RMSNorm)
│   │   ├── self_attn
│   │   │   ├── q_proj  (nn.Linear: 896→896)
│   │   │   ├── k_proj  (nn.Linear: 896→128)
│   │   │   ├── v_proj  (nn.Linear: 896→128)
│   │   │   ├── o_proj  (nn.Linear: 896→896)
│   │   │   ├── q_norm  (Qwen3RMSNorm)
│   │   │   └── k_norm  (Qwen3RMSNorm)
│   │   ├── post_attention_layernorm (Qwen3RMSNorm)
│   │   └── mlp
│   │       ├── gate_proj  (nn.Linear: 896→4864)
│   │       ├── up_proj    (nn.Linear: 896→4864)
│   │       └── down_proj  (nn.Linear: 4864→896)
│   └── .model.norm
├── .model_type  = "qwen3"
├── .is_quantized = False
└── .quant_config = AwqConfig()
```

---

## 2. 阶段0.5：构造量化器（校准环境准备）

### 2.1 入口

```python
# quantize_qwen3.py 第 69-75 行
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    },
    calib_data=calib_texts,          # 30 条本地文本
    max_calib_samples=128,
    max_calib_seq_len=512,
)
```

### 2.2 `BaseAWQForCausalLM.quantize()` — [base.py:137-247](awq/models/base.py#L137-L247)

```python
@torch.no_grad()
def quantize(self, tokenizer, quant_config, calib_data, ...):
    # 1. dict → AwqConfig 对象
    self.quant_config: AwqConfig = AwqConfig.from_dict(quant_config)
    # AwqConfig(zero_point=True, q_group_size=128, w_bit=4, version="gemm")

    # 2. 如果模型定义了黑名单，同步到 config
    if hasattr(self, "modules_to_not_convert"):
        self.quant_config.modules_to_not_convert = self.modules_to_not_convert
    # Qwen3 没有 modules_to_not_convert，所以所有 Linear 都会被量化

    # 3. 创建量化器
    self.quantizer = AwqQuantizer(
        self, self.model, tokenizer,
        w_bit=4, group_size=128, zero_point=True, version="gemm",
        calib_data=calib_texts, split="train", text_column="text",
        duo_scaling=True, export_compatible=False, apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128, max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,  # 1 GB
    )

    # 4. 执行量化
    self.quantizer.quantize()

    self.is_quantized = True
```

### 2.3 `AwqQuantizer.__init__()` — [quantizer.py:29-72](awq/quantize/quantizer.py#L29-L72)

```python
class AwqQuantizer:
    def __init__(self, awq_model, model, tokenizer, w_bit, group_size,
                 zero_point, version, calib_data, ...):
        self.awq_model = awq_model      # Qwen3AWQForCausalLM 实例
        self.model = model              # 内嵌的 Qwen3ForCausalLM
        self.tokenizer = tokenizer
        self.w_bit = 4
        self.group_size = 128
        self.zero_point = True
        self.version = "gemm"
        self.duo_scaling = True
        self.export_compatible = False
        self.apply_clip = True
        self.max_chunk_memory = 1024 * 1024 * 1024

        # ⚡ 关键步骤：准备校准环境
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=128, max_seq_len=512
        )
```

### 2.4 `init_quant()` — 最巧妙的设计 — [quantizer.py:556-625](awq/quantize/quantizer.py#L556-L625)

整个量化流程的基础设施都在这里建立，分四步：

#### Step A: 获取 transformer 层列表 + 加载校准数据

```python
def init_quant(self, n_samples=128, max_seq_len=512):
    # 获取所有 transformer decoder 层
    modules = self.awq_model.get_model_layers(self.model)
    # → Qwen3AWQForCausalLM.get_model_layers()
    # → return model.model.layers
    # modules = [Qwen3DecoderLayer_0, Qwen3DecoderLayer_1, ..., Qwen3DecoderLayer_27]
    # Qwen3-0.6B 有 28 层

    # 加载并 tokenize 校准数据
    samples = get_calib_dataset(
        data=self.calib_data,     # calib_texts: 30 条文本的 list
        tokenizer=self.tokenizer,
        n_samples=n_samples,      # 128 (最多取这么多)
        max_seq_len=max_seq_len,  # 512 (超出的截断)
        split="train",
        text_column="text",
    )
    samples = torch.cat(samples, dim=0)
    # samples.shape = [total_tokens,]  整数 token IDs
    # 30 条文本 × 平均 ~200 token/条 ≈ 6000 个 token
```

`get_calib_dataset()` 的内部逻辑（[awq/utils/calib_data.py](awq/utils/calib_data.py)）：

- 如果 `calib_data` 是字符串 → 当作 HuggingFace 数据集名，用 `datasets.load_dataset()` 加载
- 如果 `calib_data` 是 list → 逐条 `tokenizer.encode()` 并截断到 `max_seq_len`

#### Step B: "Catcher" 技巧捕获首层输入

这是整个流程最精巧的地方：需要拿到进入第一层 transformer block 的 **hidden states** 和 **attention_mask / position_ids 等 kwargs**，但不想让模型跑完所有 28 层（浪费时间和显存）。

```python
    class Catcher(nn.Module):
        """临时包装器：捕获输入后立即抛异常终止"""
        def __init__(self, module):
            super().__init__()
            self.module = module      # 保留原始 layer 0

        def forward(self, *args, **kwargs):
            # 捕获 hidden states (第一个位置参数)
            if len(args) > 0:
                hidden_states = args[0]
                del args
            else:
                first_key = list(kwargs.keys())[0]
                hidden_states = kwargs.pop(first_key)

            inps.append(hidden_states)    # ⚡ 保存 hidden states
            layer_kwargs.update(kwargs)   # ⚡ 保存 attention_mask, position_ids 等
            raise ValueError              # ⚡ 抛异常终止后续计算

    # 用 Catcher 替换 layer 0
    modules[0] = Catcher(modules[0])
    try:
        self.model(samples.to(next(self.model.parameters()).device))
    except ValueError:                    # Catcher 抛出，被这里捕获
        pass
    modules[0] = modules[0].module        # 恢复原始 layer 0
```

**为什么用抛异常而不是 hook？** 旧版 PyTorch 的 `register_forward_hook` 只能捕获 `(module, input, output)`，无法获取 `kwargs`（如 `attention_mask`、`position_ids`）。抛异常是最简洁的方式。

#### Step C: 补全 kwargs

```python
    layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
    layer_kwargs.pop("input_ids")
    # 现在 layer_kwargs 包含:
    #   attention_mask:     [batch, seq_len]
    #   position_ids:       [batch, seq_len]
    #   position_embeddings: (cos, sin)  ← transformers >= 4.48.0
```

#### Step D: 清理并返回

```python
    del samples
    inps = inps[0]
    # inps.shape = [total_calib_tokens, hidden_size]
    # 例如: [~6000, 896]  (6000个token × 896维)

    modules[0] = modules[0].cpu()
    self.awq_model.move_embed(self.model, "cpu")

    clear_memory()
    return modules, layer_kwargs, inps
    #     28个层     forward的kwargs   第1层的输入
```

### 2.5 结果总结

初始化完成后，量化器持有：

| 成员 | 值 | 用途 |
|---|---|---|
| `self.modules` | `[Layer_0, ..., Layer_27]` | 待量化的 28 个 Qwen3DecoderLayer |
| `self.module_kwargs` | `{attention_mask, position_ids, ...}` | 前向传播时传给每层的额外参数 |
| `self.inps` | `tensor[~6000, 896]` | 第一层的输入 hidden states |

---

## 3. 主循环：`AwqQuantizer.quantize()` — [quantizer.py:127-216](awq/quantize/quantizer.py#L127-L216)

逐层循环，对每层执行 **Step 1→2→3→4**：

```python
def quantize(self):
    for i in tqdm(range(len(self.modules)), desc="AWQ"):  # i = 0, 1, ..., 27
        # ==================== 设备调度 ====================
        # 把当前层搬到 GPU，前一层处理完的 inps 也同步
        common_device = next(self.modules[i].parameters()).device
        if common_device is None or str(common_device) == "cpu":
            if torch.cuda.is_available():
                best_device = "cuda:" + str(i % torch.cuda.device_count())
            else:
                best_device = get_best_device()
            self.modules[i] = self.modules[i].to(best_device)
            common_device = next(self.modules[i].parameters()).device

        self.inps = self.inps.to(common_device)

        # rotary_emb 需要跟随当前层所在设备
        self.awq_model.move_embed(self.model, common_device)

        # transformers >= 4.48.0: 需要预先计算 position_embeddings
        if (transformers.__version__ >= "4.48.0"
            and self.module_kwargs.get("position_embeddings") is None):
            self.module_kwargs["position_embeddings"] = (
                self.model.model.rotary_emb(
                    self.inps, self.module_kwargs["position_ids"]
                )
            )

        # ==================== STEP 1 ====================
        named_linears = get_named_linears(self.modules[i])
        named_linears = exclude_layers_to_not_quantize(
            named_linears, self.modules_to_not_convert
        )
        input_feat = self._get_input_feat(self.modules[i], named_linears)
        clear_memory()

        # ==================== STEP 2 ====================
        module_config = self.awq_model.get_layers_for_scaling(
            self.modules[i], input_feat, self.module_kwargs
        )
        scales_list = [
            self._search_best_scale(self.modules[i], **layer)
            for layer in module_config
        ]
        apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)

        # ==================== STEP 3 ====================
        if self.apply_clip:
            clip_list = self._search_best_clip(
                self.modules[i], named_linears, input_feat
            )
            apply_clip(self.modules[i], clip_list)

        # ==================== STEP 4 ====================
        if not self.export_compatible:
            self._apply_quant(self.modules[i], named_linears)

        clear_memory()
```

---

## 4. Step 1：捕获输入特征 — `_get_input_feat()`

[quantizer.py:627-685](awq/quantize/quantizer.py#L627-L685)

### 4.1 目的

获取校准数据经过当前层时，每个 `nn.Linear` **输入端的激活值（activation）**，用于后续 scale 和 clip 搜索。

### 4.2 代码

```python
def _get_input_feat(self, layer, named_linears):
    def cache_input_hook(m, x, y, name, feat_dict):
        x = x[0]                         # hook 传入的 inputs[0]
        x = x.detach().cpu()             # 搬到 CPU 释放 GPU 显存
        feat_dict[name].append(x)        # 按层名累积

    input_feat = defaultdict(list)
    handles = []

    # 为每个待量化的 Linear 注册 forward hook
    for name in named_linears:
        handles.append(
            named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
            )
        )

    # ⚡ 前向传播一次，hook 自动收集每个 Linear 的输入
    self.inps = self.inps.to(next(layer.parameters()).device)
    module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)
    self.inps = self._module_forward(self.inps, layer, module_kwargs)
    # ⚡ self.inps 被更新为当前层的输出，下一层的输入！

    for h in handles:
        h.remove()

    # 合并同一个层多次前向收集到的数据
    def cat_and_assert(k, v):
        x = torch.cat(v, dim=0)
        assert x.shape[0] != 0, (
            f"{k} has zero dimension. "
            "Try increasing max_calib_samples."
        )
        return x

    input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}
    return input_feat
```

### 4.3 中间件：`_module_forward()` — [quantizer.py:268-292](awq/quantize/quantizer.py#L268-L292)

处理 `n_parallel_calib_samples` 分批执行的情况：

```python
@torch.no_grad()
def _module_forward(self, x, module, module_kwargs):
    if self.n_parallel_calib_samples is None:
        # 一次性跑完所有校准样本
        module_output = module(x, **module_kwargs)
        if isinstance(module_output, tuple):
            module_output = module_output[0]  # 有些模型返回 (hidden_states, ...)
    else:
        # 分批执行，结果在 CPU 上拼回，防止 OOM
        module_output = []
        partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
        for x_partial in partitioned_inputs:
            partial_output = module(x_partial, **module_kwargs)
            if isinstance(partial_output, tuple):
                partial_output = partial_output[0]
            module_output.append(partial_output.cpu())
        module_output = torch.cat(module_output, dim=0)
    return module_output
```

### 4.4 `_sanitize_kwargs()` — [quantizer.py:687-704](awq/quantize/quantizer.py#L687-L704)

关键细节：不同版本的 transformers 接受的 forward 参数不同，必须用 `inspect.signature` 过滤：

```python
def _sanitize_kwargs(self, inputs_kwargs, module):
    module_signature = inspect.signature(module.forward).parameters
    sanitized_kwargs = {}
    for k, v in inputs_kwargs.items():
        if k in module_signature:        # 只传模块明确接受的参数
            sanitized_kwargs[k] = v
    return sanitized_kwargs
```

### 4.5 结果

```python
input_feat = {
    "self_attn.q_proj":  tensor[~6000, 896],
    "self_attn.k_proj":  tensor[~6000, 896],
    "self_attn.v_proj":  tensor[~6000, 896],
    "self_attn.o_proj":  tensor[~6000, 896],
    "mlp.gate_proj":     tensor[~6000, 896],
    "mlp.up_proj":       tensor[~6000, 896],
    "mlp.down_proj":     tensor[~6000, 4864],
}
# 每个 tensor 的第一维 = 校准数据的总 token 数
# 第二维 = 对应 Linear 的 in_features
```

---

## 5. Step 2：Scale 搜索与注入（AWQ 核心）

### 5.1 `get_layers_for_scaling()` — Qwen3 层分组 — [qwen3.py:37-85](awq/models/qwen3.py#L37-L85)

```python
@staticmethod
def get_layers_for_scaling(module, input_feat, module_kwargs):
    layers = []

    # ===== 组1: Attention 输入 =====
    # prev_op = input_layernorm (RMSNorm)
    # layers  = [q_proj, k_proj, v_proj]  三个共享同一 norm，必须一起 scale
    layers.append(dict(
        prev_op=module.input_layernorm,
        layers=[module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj],
        inp=input_feat["self_attn.q_proj"],
        module2inspect=module.self_attn,   # 用完整 attention 模块评估 loss
        kwargs=module_kwargs,              # attention_mask, position_ids 等
    ))

    # ===== 组2: Attention 输出 =====
    # prev_op = v_proj  (特别: v_proj → attention_op → o_proj)
    # layers  = [o_proj]
    # 仅在 v_proj 和 o_proj 同为方阵时执行
    # Qwen3-0.6B: v_proj 是 896×128, o_proj 是 896×896 → 不同形状 → 跳过!
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(dict(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.o_proj],
            inp=input_feat["self_attn.o_proj"],
        ))

    # ===== 组3: MLP 输入 =====
    # prev_op = post_attention_layernorm (RMSNorm)
    # layers  = [gate_proj, up_proj]  共享同一 norm，必须一起 scale
    layers.append(dict(
        prev_op=module.post_attention_layernorm,
        layers=[module.mlp.gate_proj, module.mlp.up_proj],
        inp=input_feat["mlp.gate_proj"],
        module2inspect=module.mlp,          # 用完整 MLP 模块评估 loss
    ))

    # ===== 组4: MLP 输出 =====
    # prev_op = up_proj  (up_proj → SiLU+gate → down_proj)
    # layers  = [down_proj]
    layers.append(dict(
        prev_op=module.mlp.up_proj,
        layers=[module.mlp.down_proj],
        inp=input_feat["mlp.down_proj"],
    ))

    return layers
```

**关键设计**：每组中 `prev_op` 唯一决定了 scale 注入的位置。当多个 layer 共享同一个 `prev_op`（如 input_layernorm → q/k/v），它们的 scale 必须相同。当只有一个 layer 时（如 up_proj → down_proj），scale 可以独立。

### 5.2 `_search_best_scale()` — [quantizer.py:294-366](awq/quantize/quantizer.py#L294-L366)

对 layer 配置中的每一组执行：

```python
@torch.no_grad()
def _search_best_scale(self, module, prev_op, layers, inp,
                       module2inspect=None, kwargs={}):
    if module2inspect is None:
        assert len(layers) == 1
        module2inspect = layers[0]   # 没有评估模块 → 用 layer 本身

    inp = inp.to(next(module2inspect.parameters()).device)

    # -------- A: 计算 per-channel weight mean (w_mean) --------
    # 拼接同组所有 layer 的权重
    weight = torch.cat([_m.weight for _m in layers], dim=0)
    # 组1: [1152, 896]  (896+128+128)
    # 组4: [896, 4864]  只有 down_proj

    org_shape = weight.shape
    # 按 group_size 分组，组内归一化，消除不同 group 的幅度差异
    weight = weight.view(-1, self.group_size)             # [*, 128]
    w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
    # 每行归一化到 [0, 1]，衡量 channel 内部各位置的相对大小
    w_scale = w_scale.view(org_shape)
    w_mean = w_scale.mean(0)   # 每个 input channel 的平均相对幅度
    clear_memory(weight)

    # -------- B: 计算 per-channel activation mean (x_mean) --------
    # 带 chunk 的内存保护
    inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])  # [tokens, in_features]
    num_elements = inp_flat.size(0)
    num_channels = inp_flat.size(1)
    element_size_bytes = inp_flat.element_size() * 2    # ×2 是 FP32 的预算

    chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
    chunk_size = min(chunk_size, num_elements)

    x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
    for i in range(0, num_elements, chunk_size):
        end = min(i + chunk_size, num_elements)
        chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
        x_sum += chunk_sum.to(inp.device)

    x_mean = (x_sum / num_elements).to(inp.dtype)
    clear_memory(x_sum)

    # -------- C: 计算 FP16 标准输出 (ground truth) --------
    with torch.no_grad():
        module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
        fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

    # -------- D: 网格搜索最优 scale --------
    best_scales = self._compute_best_scale(
        inp, w_mean, x_mean, module2inspect, layers,
        fp16_output, module_kwargs
    )

    return (
        get_op_name(module, prev_op),                     # "input_layernorm"
        tuple([get_op_name(module, m) for m in layers]),  # ("self_attn.q_proj", ...)
        best_scales,                                      # tensor[896]
    )
```

### 5.3 `_compute_best_scale()` — 网格搜索 — [quantizer.py:368-442](awq/quantize/quantizer.py#L368-L442)

**AWQ 论文公式**：

```
L(s) = || Q(W · s) · (s⁻¹ · X) - W · X ||
        \_________/   \________/   \______/
         量化权重    缩放后输入    FP16基准
```

```python
def _compute_best_scale(self, x, w_mean, x_mean, module2inspect,
                         linears2scale, fp16_output, kwargs):
    n_grid = 20              # 搜索 20 个 α 值
    history = []
    best_ratio = -1
    best_scales = None
    best_error = float("inf")

    # 保存原始 state_dict，每轮恢复
    org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

    device = x.device
    x_mean = x_mean.view(-1).to(device)
    w_mean = w_mean.view(-1).to(device)

    for ratio in range(n_grid):
        ratio = ratio / n_grid  # α ∈ {0, 0.05, 0.10, ..., 0.95}

        # ⚡ AWQ 核心公式: s = x_mean^α / w_mean^(1-α)
        if self.duo_scaling:
            scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
        else:
            scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

        # 归一化 (几何平均 = 1)，保持整体幅度不变
        scales = scales / (scales.max() * scales.min()).sqrt()
        scales_view = scales.view(1, -1).to(device)

        # 处理溢出
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1

        # ⚡ Q(W · s) / s: 量化后反量化
        for fc in linears2scale:
            fc.weight.mul_(scales_view)                          # W ← W * s
            fc.weight.data = (
                self.pseudo_quantize_tensor(fc.weight.data)[0]   # Q(W*s)
                / scales_view                                     # / s
            )

        # 前向传播，评估量化效果
        int_w_output = self._module_forward(x, module2inspect, kwargs)
        loss = self._compute_loss(fp16_output, int_w_output, device)

        history.append(loss)
        if loss < best_error:
            best_error = loss
            best_ratio = ratio
            best_scales = scales.clone()

        # 恢复权重到原始状态
        module2inspect.load_state_dict(org_sd)

    assert torch.isnan(best_scales).sum() == 0
    return best_scales.detach().cpu()
```

### 5.4 伪量化：`pseudo_quantize_tensor()` — [quantizer.py:74-109](awq/quantize/quantizer.py#L74-L109)

模拟量化→反量化过程（不产生真实的 INT4 打包）：

```python
def pseudo_quantize_tensor(self, w: torch.Tensor):
    org_w_shape = w.shape
    if self.group_size > 0:
        w = w.reshape(-1, self.group_size)   # [*, 128]

    if self.zero_point:                       # 非对称量化 (本例: True)
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**self.w_bit - 1           # 15  (4-bit 无符号)
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        # 量化: round(w/scale + zero) → clamp → (减去zero)*scale = 反量化
        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
             - zeros) * scales
        zeros = zeros.view(org_w_shape[0], -1)
    else:                                     # 对称量化
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (self.w_bit - 1) - 1   # 7
        min_int = -(2 ** (self.w_bit - 1))    # -8
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)
    return w, scales, zeros
```

**非对称量化图解**（以 `zero_point=True` 为例）：

```
原始 FP16 范围: [min_val = -X, max_val = +Y]
量化范围:        [0, 15]  (4-bit)

scale  = (max_val - min_val) / 15       # 每个量化步长代表多少 FP16 值
zero   = round(-min_val / scale)        # FP16 的 0 对应哪个整数

q(w)   = round(w / scale) + zero        # 量化
w'     = (q(w) - zero) * scale          # 反量化
```

### 5.5 `_compute_loss()` — 内存安全的 MSE — [quantizer.py:444-474](awq/quantize/quantizer.py#L444-L474)

```python
@torch.no_grad()
def _compute_loss(self, fp16_output, int_w_output, device):
    loss = 0.0
    fp16_output_flat = fp16_output.view(-1)
    int_w_output_flat = int_w_output.view(-1)
    num_elements = fp16_output_flat.size(0)
    element_size_bytes = fp16_output.element_size()

    # 动态 chunk: 1GB / (2 * 2字节) = 2.5亿个元素
    chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
    chunk_size = min(chunk_size, num_elements)

    fp16_chunks = torch.split(fp16_output_flat, chunk_size)
    int_w_chunks = torch.split(int_w_output_flat, chunk_size)

    for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
        chunk_loss = (
            (fp16_chunk.to(device) - int_w_chunk.to(device))
            .float().pow(2).sum().item()
        )
        loss += chunk_loss

    loss /= num_elements   # MSE = Σ(xi - yi)² / N
    return loss
```

### 5.6 应用 Scale — [scale.py](awq/quantize/scale.py)

`apply_scale()` 根据 `prev_op` 的类型选择注入策略：

#### 情况一：prev_op 是 RMSNorm/LayerNorm → `scale_ln_fcs()` — [scale.py:88-113](awq/quantize/scale.py#L88-L113)

```python
# 对应组1 (input_layernorm → [q_proj, k_proj, v_proj])
# 对应组3 (post_attention_layernorm → [gate_proj, up_proj])

def scale_ln_fcs(ln, fcs, scales):
    scales = scales.to(ln.weight.device)

    # RMSNorm: 输出 = x * weight / sqrt(...)
    # 等价变换: ln.weight /= scales  →  输出缩小
    if isinstance(ln, GemmaRMSNorm) or isinstance(ln, Gemma2RMSNorm):
        ln.weight += 1
        ln.weight.div_(scales)
        ln.weight -= 1
    else:
        ln.weight.div_(scales)           # ⚡ Qwen3RMSNorm.weight /= s

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    # 补偿: 每个 Linear 的权重放大
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))  # ⚡ Q/K/V/Up/Gate 的权重 *= s
```

#### 情况二：prev_op 是 nn.Linear (1→1) → `scale_fc_fc()` — [scale.py:117-133](awq/quantize/scale.py#L117-L133)

```python
# 对应组2 (v_proj → o_proj) — 但 Qwen3-0.6B 因形状不匹配跳过
# 对应组4 (up_proj → down_proj)

def scale_fc_fc(fc1, fc2, scales):
    scales = scales.to(fc1.weight.device)

    # fc1 的输出侧缩小
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # fc2 的输入侧放大
    fc2.weight.mul_(scales.view(1, -1))
```

#### 情况三：prev_op 是激活函数 → `scale_gelu_fc()` — [scale.py:157-164](awq/quantize/scale.py#L157-L164)

```python
def scale_gelu_fc(gelu, fc, scales):
    # 替换为 ScaledActivation (activation / scales)
    # 同时 fc 权重 *= scales 做补偿
    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
```

#### Scale 同步更新 input_feat — [scale.py:73-79](awq/quantize/scale.py#L73-L79)

```python
# 因为 norm 被 scale 了，经过 norm 后的激活值也会等比例变化
# input_feat 必须同步更新，否则下一步 clip 搜索的误差计算不对
if input_feat_dict is not None:
    for layer_name in layer_names:
        if layer_name in input_feat_dict:
            inp = input_feat_dict[layer_name]
            inp.div_(scales.view(1, -1).to(inp.device))
```

### 5.7 Scale 核心原理总结

```
等价变换:
  输入 X → RMSNorm(w_norm) →  X'  → Linear(w_fc) →  输出 Y
  输入 X → RMSNorm(w_norm/s) → X'/s → Linear(w_fc*s) → 输出 Y (相同!)

但权重 w_fc*s 量化时:
  - 变得更大 → 量化相对误差 1/(w_fc*s) < 1/w_fc
  - outlier channel 被暴露出来，量化时更容易保护

AWQ 通过网格搜索找到每个 channel 的最优缩放因子 s，
在"让权重变大以减小量化误差"和"不让权重过大导致溢出"之间取得平衡。
```

---

## 6. Step 3：Weight Clipping — [quantizer.py:476-554](awq/quantize/quantizer.py#L476-L554)

### 6.1 目的

对每个 `nn.Linear` 的每个量化组，搜索最优的 weight clipping 阈值。将 outlier 权重 clip 掉后再量化，可以减少量化误差。

### 6.2 哪几层 Skip

Q/K 相关层跳过（QK 矩阵乘法对权重精度非常敏感）：

```python
def _search_best_clip(self, layer, named_linears, input_feat):
    clip_list = []
    avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

    for name in named_linears:
        if any([_ in name for _ in avoid_clipping]):
            continue    # ⚡ 跳过 q_proj, k_proj, q_norm, k_norm

        named_linears[name].to(get_best_device())
        max_val = self._compute_best_clip(
            named_linears[name].weight, input_feat[name]
        )
        clip_list.append((name, max_val))
        named_linears[name].cpu()

    return clip_list
```

### 6.3 `_compute_best_clip()` — 网格搜索 clip 阈值

```python
@torch.no_grad()
def _compute_best_clip(self, w, input_feat, n_grid=20,
                       max_shrink=0.5, n_sample_token=512):
    # w:          [out_features, in_features]
    # input_feat: [tokens, in_features]

    group_size = self.group_size  # 128

    # Reshape: 按 group 维度拆分
    input_feat = input_feat.reshape(1, tokens, n_groups, 128)
    w          = w.reshape(out_features, 1, n_groups, 128)

    org_max_val = w.abs().amax(dim=-1, keepdim=True)  # 原始最大绝对值
    org_out     = (input_feat * w).sum(dim=-1)         # 原始输出 (FP16)

    best_max_val = org_max_val.clone()
    min_errs     = torch.ones_like(org_max_val) * 1e9

    # 每次缩小 max_val 一点，测 clip 后的量化误差
    for i_s in range(int(max_shrink * n_grid)):   # 10 步: 1.0 → 0.5
        max_val = org_max_val * (1 - i_s / n_grid)
        min_val = -max_val
        cur_w   = torch.clamp(w, min_val, max_val)
        q_w     = self.pseudo_quantize_tensor(cur_w)[0]  # clip 后伪量化
        cur_out = (input_feat * q_w).sum(dim=-1)

        err = (cur_out - org_out).pow(2).mean(dim=1)  # 逐组 MSE
        cur_best_idx = err < min_errs
        min_errs[cur_best_idx]     = err[cur_best_idx]
        best_max_val[cur_best_idx] = max_val[cur_best_idx]

    return best_max_val.squeeze(1)
```

### 6.4 应用 Clip — [scale.py:25-34](awq/quantize/scale.py#L25-L34)

```python
@torch.no_grad()
def apply_clip(module, clip_list):
    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        layer.to(get_best_device())
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        # clamp 到 [-max_val, +max_val]
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()
```

---

## 7. Step 4：真实 INT4 量化 — `_apply_quant()`

[quantizer.py:227-265](awq/quantize/quantizer.py#L227-L265)

### 7.1 代码

```python
def _apply_quant(self, module, named_linears):
    for name, linear_layer in named_linears.items():
        # 1) 伪量化得到 scales + zeros
        linear_layer = linear_layer.to(get_best_device()).half()
        linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
            linear_layer.weight.data
        )

        # 2) 选择量化 Linear 后端
        if self.version == "gemm":
            scales = scales.t().contiguous()
            if zeros is not None:
                zeros = zeros.t().contiguous()
            q_linear_module = WQLinear_GEMM
        elif self.version == "gemv":
            q_linear_module = WQLinear_GEMV
        elif self.version == "marlin":
            q_linear_module = WQLinear_Marlin
        elif self.version == "gemv_fast":
            q_linear_module = WQLinear_GEMVFast

        # 3) 工厂方法: FP16 权重 → INT4 打包
        q_linear = q_linear_module.from_linear(
            linear=linear_layer,
            w_bit=self.w_bit,
            group_size=self.group_size,
            init_only=False,        # False = 真正做量化
            scales=scales,
            zeros=zeros,
        )

        # 4) 替换原模型中的 nn.Linear
        linear_layer.cpu()
        q_linear.to(next(module.parameters()).device)
        set_op_by_name(module, name, q_linear)
        clear_memory()
```

### 7.2 INT4 打包详解：`WQLinear_GEMM.from_linear()` — [gemm.py:171-251](awq/modules/linear/gemm.py#L171-L251)

```python
@classmethod
def from_linear(cls, linear, w_bit, group_size, init_only=False,
                scales=None, zeros=None):
    awq_linear = cls(w_bit, group_size, linear.in_features,
                     linear.out_features,
                     linear.bias is not None, linear.weight.device)
    if init_only:
        return awq_linear   # 只创建结构，不做打包（用于 from_quantized 加载）

    # 量化: q = round((w + zero * scale) / scale)
    scale_zeros = zeros * scales

    intweight = []
    for idx in range(awq_linear.in_features):
        intweight.append(
            torch.round(
                (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                / awq_linear.scales[idx // group_size]
            ).to(torch.int)[:, None]
        )
    intweight = torch.cat(intweight, dim=1).t().contiguous()
    # intweight.shape = [in_features, out_features], dtype=int32

    # ⚡ 打包: 每 8 个 4-bit 值打包进 1 个 int32
    # 位序: [0, 2, 4, 6, 1, 3, 5, 7] (AWQ 标准位序)
    pack_num = 32 // w_bit   # = 8
    qweight = torch.zeros(
        (intweight.shape[0], intweight.shape[1] // 32 * w_bit),
        dtype=torch.int32, device=intweight.device
    )

    for col in range(intweight.shape[1] // pack_num):
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        for i in range(pack_num):
            qweight_col = intweight[:, col * pack_num + order_map[i]]
            qweight[:, col] |= qweight_col << (i * w_bit)
    awq_linear.qweight = qweight

    # 同理打包 zeros
    qzeros = torch.zeros(...)
    for col in range(zeros.shape[1] // pack_num):
        for i in range(pack_num):
            qzero_col = zeros[:, col * pack_num + order_map[i]]
            qzeros[:, col] |= qzero_col << (i * w_bit)
    awq_linear.qzeros = qzeros

    awq_linear.scales = scales.clone().half()
    if linear.bias is not None:
        awq_linear.bias = linear.bias.clone().half()

    return awq_linear
```

### 7.3 打包结果结构

以 Qwen3-0.6B 的 `q_proj` (896×896) 为例：

| buffer | shape | dtype | 压缩率 |
|---|---|---|---|
| `qweight` | `[896, 112]` | int32 | 896×896×16bit → 896×112×32bit = 50% |
| `qzeros` | `[7, 112]` | int32 | 7 = 896/128 |
| `scales` | `[7, 896]` | float16 | 7×896×2字节 = 12.25KB |

**总压缩率**（4-bit 量化、group_size=128）：
- 原始: `in_features × out_features × 16 bits`
- 量化: `in_features × out_features × 4 bits` (qweight) + `(in_features/group_size × out_features) × (16 + 4) bits` (scales + zeros)
- 典型值: **~3.5-3.8 倍压缩**

### 7.4 前向推理时 `WQLinearMMFunction.forward()` — [gemm.py:24-86](awq/modules/linear/gemm.py#L24-L86)

```python
@staticmethod
def forward(ctx, x, qweight, qzeros, scales, w_bit=4, group_size=128,
            bias=None, out_features=0):
    # 3 级回退策略:
    if awq_ext is not None:
        # 1) CUDA kernel (最快) — autoawq_kernels 包
        FP16_MATMUL_HEURISTIC = x.shape[0] * x.shape[1] >= 1024
        if FP16_MATMUL_HEURISTIC:
            out = awq_ext.dequantize_weights_cuda(
                qweight, scales, qzeros, 0, 0, 0, False
            )
            out = torch.matmul(x, out)
        else:
            out = awq_ext.gemm_forward_cuda(
                x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, 8
            )
    elif TRITON_AVAILABLE:
        # 2) Triton kernel (GPU 通用) — awq/modules/triton/gemm.py
        FP16_MATMUL_HEURISTIC = x.shape[0] * x.shape[1] >= 1024
        if FP16_MATMUL_HEURISTIC:
            out = awq_dequantize_triton(qweight, scales, qzeros)
            out = torch.matmul(x, out.to(x.dtype))
        else:
            out = awq_gemm_triton(x.reshape(-1, x.shape[-1]),
                                  qweight, scales, qzeros, split_k_iters=8)
    else:
        # 3) 纯 Python 回退 (最慢)
        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)

    return out + (bias if bias is not None else 0)
```

**启发式规则**：当矩阵足够大（≥1024）时，先完整 dequantize 为 FP16 再 matmul 更快；矩阵较小时，fused dequantize+gemm kernel 更快。

---

## 8. 阶段 5：保存量化模型 — `save_quantized()`

[base.py:274-319](awq/models/base.py#L274-L319)

### 8.1 代码

```python
def save_quantized(self, save_dir, safetensors=True, shard_size="5GB"):
    save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

    # 1) AwqConfig → HF 兼容格式写入 config.json
    self.model.config.quantization_config = (
        self.quant_config.to_transformers_dict()
    )
    # 产生:
    # {
    #   "quant_method": "awq",
    #   "zero_point": true,
    #   "bits": 4,
    #   "group_size": 128,
    #   "version": "gemm",
    #   "modules_to_not_convert": null
    # }

    # 2) 先保存 config + generation_config（空壳权重，稍后删除）
    self.model.generation_config.do_sample = True
    self.model.save_pretrained(
        save_dir, state_dict=EmptyModule().state_dict()
    )

    # 3) 如果有 processor（如 vision models），也保存
    if self.processor is not None:
        self.processor.save_pretrained(save_dir)

    # 4) 删除空壳权重文件
    for path in [f"{save_dir}/model.safetensors",
                 f"{save_dir}/pytorch_model.bin"]:
        if os.path.exists(path):
            os.remove(path)

    # 5) 用 huggingface_hub 的工具函数保存真正的量化权重
    save_torch_state_dict(
        state_dict=self.model.state_dict(),
        # state_dict 中每个 Linear 已被替换为 WQLinear_GEMM
        # 包含 qweight (int32), qzeros (int32), scales (float16)
        save_directory=save_dir,
        max_shard_size=shard_size,
        safe_serialization=safetensors,
        force_contiguous=True,
        shared_tensors_to_discard=self.model._tied_weights_keys,
    )
```

### 8.2 保存后的目录结构

```
Qwen3-0.6B-AWQ/
├── config.json                       # 含 quantization_config 字段
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── model-00001-of-00001.safetensors  # 所有量化权重
    # 内部结构 (safetensors 是 key-value 格式):
    # model.layers.0.self_attn.q_proj.qweight  (int32)
    # model.layers.0.self_attn.q_proj.qzeros  (int32)
    # model.layers.0.self_attn.q_proj.scales  (float16)
    # model.layers.0.self_attn.q_proj.bias    (float16, 如果有)
    # ...
    # model.layers.0.input_layernorm.weight   (float16, 已被 scale 修改)
    # model.model.embed_tokens.weight         (float16, 不量化)
    # model.model.norm.weight                 (float16, 不量化)
    # model.lm_head.weight                    (float16, 不量化)
```

---

## 9. 回顾：完整调用栈

```
quantize_qwen3.py
│
├─ AutoAWQForCausalLM.from_pretrained(path)        [auto.py:68]
│  ├─ check_and_get_model_type()                   [auto.py:50] → "qwen3"
│  └─ Qwen3AWQForCausalLM.from_pretrained()        [base.py:321]
│     ├─ _load_config()                            [base.py:572] → 定位权重 + 加载config
│     ├─ AutoModelForCausalLM.from_pretrained()    HF 标准加载 FP16 模型
│     └─ return Qwen3AWQForCausalLM(model, ...)    [base.py:400]
│
├─ model.quantize(tokenizer, config, calib_data)   [base.py:137]
│  ├─ AwqConfig.from_dict(config)                  [_config.py]
│  ├─ AwqQuantizer.__init__(...)                   [quantizer.py:29]
│  │  └─ init_quant()                              [quantizer.py:556]
│  │     ├─ get_model_layers()                     [qwen3.py:24] → model.model.layers
│  │     ├─ get_calib_dataset()                    [calib_data.py]
│  │     ├─ Catcher 捕获首层输入 + 抛异常终止        [quantizer.py:578-601]
│  │     └─ prepare_inputs_for_generation()        补全 kwargs
│  │
│  └─ quantizer.quantize()                         [quantizer.py:127]
│     └─ for each transformer layer:
│        ├─ get_named_linears()                    [module.py]
│        ├─ _get_input_feat()                      [quantizer.py:627]
│        │  └─ register_forward_hook() × N         hook 收集激活值
│        │
│        ├─ get_layers_for_scaling()               [qwen3.py:37]
│        │  └─ 返回 4 组 layer scale 配置
│        │
│        ├─ _search_best_scale() × 4 组            [quantizer.py:294]
│        │  ├─ 计算 w_mean                          per-channel weight 均值
│        │  ├─ 计算 x_mean (带 chunk)              per-channel activation 均值
│        │  ├─ _module_forward() → fp16_output      FP16 基线
│        │  └─ _compute_best_scale()               [quantizer.py:368]
│        │     ├─ for ratio in 0..19:              网格搜索 α
│        │     │  ├─ scales = x_mean^α / w_mean^(1-α)
│        │     │  ├─ pseudo_quantize_tensor()       模拟量化
│        │     │  ├─ _module_forward()              评估
│        │     │  ├─ _compute_loss() (带 chunk)     MSE
│        │     │  └─ load_state_dict(org_sd)        恢复权重
│        │     └─ return best_scales
│        │
│        ├─ apply_scale()                           [scale.py:37]
│        │  ├─ scale_ln_fcs()    (RMSNorm → QKV, RMSNorm → GateUp)
│        │  ├─ scale_fc_fc()     (Up → Down)
│        │  └─ 同步更新 input_feat_dict
│        │
│        ├─ _search_best_clip()                     [quantizer.py:476]
│        │  ├─ 跳过 Q/K 层
│        │  └─ _compute_best_clip()                 网格搜索 clip 阈值
│        ├─ apply_clip()                            [scale.py:25]
│        │
│        └─ _apply_quant()                          [quantizer.py:227]
│           ├─ pseudo_quantize_tensor()             最终 scale/zero 确定
│           └─ WQLinear_GEMM.from_linear()          [gemm.py:171]
│              └─ INT4 打包 + 替换 nn.Linear
│
└─ model.save_quantized(path)                       [base.py:274]
   ├─ quant_config.to_transformers_dict()           写入 config.json
   ├─ model.save_pretrained() + 删空壳               保存 config
   └─ save_torch_state_dict()                       保存量化权重 safetensors
```

---

## 10. 关键设计要点

### 10.1 为什么 Q/K/V 必须一起 scale？

Q/K/V 共享 `input_layernorm` 作为 prev_op。scale 只能在 `input_layernorm.weight` 处统一注入（`ln.weight /= scales`），无法分别对 Q/K/V 做不同的 scale。因此它们共享同一个 scale vector。

### 10.2 为什么 gate_proj 和 up_proj 要一起 scale？

同理，共享 `post_attention_layernorm` 作为 prev_op。

### 10.3 为什么 o_proj 的 prev_op 是 v_proj？

Attention 输出路径的特殊性：`v_proj → attention_op(QK^T·V) → o_proj`。通过将 scale 注入 v_proj 的输出侧同时等比例缩放 o_proj 输入侧（`scale_fc_fc`），实现等价变换。注意 Qwen3-0.6B 的 v_proj (896×128) 和 o_proj (896×896) shape 不同，所以这一步被跳过。

### 10.4 `max_chunk_memory` 的两次使用

1. **per-channel mean 计算**（`_search_best_scale` 的 step B）：`inp_flat` 求和时避免 chunk 过大 OOM
2. **loss 计算**（`_compute_loss`）：`fp16_output` 与 `int_w_output` 的 MSE 计算避免 OOM

两个地方都会因为校准 token 数大或 hidden_size 大而显存爆炸。`max_chunk_memory=1GB` 是默认值，该改大就改大。

### 10.5 `export_compatible` 两阶段导出

```
阶段一: quantize(export_compatible=True)
        → 搜索 scale + clip + 注入到 FP16 权重
        → 不替换 nn.Linear
        → 保存 FP16 模型（权重已经被 scale 调整）
        → 可用于 GGUF 等其他格式转换

阶段二: model.pack()
        → 执行 _apply_quant()
        → nn.Linear → WQLinear_GEMM (INT4 打包)
        → 保存 → CUDA 推理可用
```

### 10.6 校准数据的关键约束

- `max_calib_samples=128`：最多用 128 条校准文本
- `max_calib_seq_len=512`：每条文本截断到最多 512 token
- `n_parallel_calib_samples=None`：全部一次跑（显存不够时建议设为 1-4）
- 校准数据不需要标签，只需要纯文本 → tokenize 后前向传播收集激活值
- 校准数据的质量会显著影响量化效果，建议用任务相关的数据
