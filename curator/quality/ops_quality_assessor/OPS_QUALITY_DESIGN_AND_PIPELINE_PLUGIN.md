# Ops Quality Assessor 设计、功能与 Pipeline 插件接入说明

本文档用于说明当前 `ops-quality` 的整体设计和能力，并以类似 Git 新增文件展示的方式，整理为了让 `ops_quality` 在 pipeline 中更清晰接入时可新增或重点暴露的函数/插件入口。

## 1. 当前 ops-quality 目录结构

```text
curator/quality/ops_quality_assessor/
├── __init__.py
├── assessor.py
├── base.py
├── config.py
├── context.py
├── results.py
└── metrics/
    ├── chair_object_hallucination.py
    ├── coherence_score.py
    ├── common.py
    ├── image_text_alignment.py
    ├── qae_grounding_alignment.py
    ├── text_noise_contamination.py
    ├── text_normalization_validity.py
    ├── text_semantic_preservation.py
    ├── video_image_alignment.py
    ├── visual_robustness.py
    └── visual_transform_consistency.py
```

## 2. 当前核心设计

### 2.1 总入口：`OpsQualityAssessor`

`assessor.py` 中的 `OpsQualityAssessor` 是 ops-quality 的统一评测入口，主要职责是：

- 持有默认指标集合 `DEFAULT_METRIC_ASSESSORS`。
- 将输入数据转换为评测上下文。
- 调用每个 metric assessor 执行评测。
- 汇总每条样本、每个指标的评测结果。

当前主要调用方式包括：

```python
assessor = OpsQualityAssessor()
results = assessor.evaluate(dataset)
```

或在 pipeline 插件中针对抽样行调用：

```python
results = assessor.evaluate_rows(dataset, rows)
```

### 2.2 上下文构建：`EvaluationContextBuilder`

`context.py` 负责把原始数据、变换前后字段、图片/视频/文本内容等统一成 `EvaluationContext`。

metric assessor 不直接依赖 pipeline，而是依赖标准化后的上下文，因此可以复用于：

- 单元测试/烟测。
- 直接脚本运行。
- pipeline after-operation hook。
- 后续批处理质量评估任务。

### 2.3 指标基类：`BaseMetricAssessor`

`base.py` 中的基类统一处理：

- metric 名称。
- 必需字段检查。
- 前置条件检查。
- 单个上下文的 `assess(context)` 调用封装。
- 失败时返回结构化失败结果，而不是让整个评测流程崩溃。

### 2.4 结果对象：`MetricResult`

`results.py` 提供统一的 metric 输出结构，便于 pipeline、测试和后续报告消费。

典型输出包含：

- metric 名称。
- 是否通过/是否可用。
- 分数值。
- 失败原因或诊断信息。
- 附加 metadata。

## 3. 当前默认评测功能

当前 ops-quality 默认覆盖多模态数据处理中的文本、图像、视频、QAE 与一致性场景，默认指标集合约包括以下类别：

```text
DEFAULT_METRIC_ASSESSORS
├── CHAIR object hallucination
│   └── 检查 caption/文本中对象提及与视觉对象之间的幻觉风险。
├── Image-text alignment
│   └── 评估图像与文本描述的一致性。
├── Video-image alignment
│   └── 评估视频帧/图像与文本或参考图像之间的一致性。
├── Visual robustness
│   └── 检查视觉增强/扰动后的质量稳定性。
├── Visual transform consistency
│   └── 检查视觉变换前后语义是否保持一致。
├── Text noise contamination
│   └── 检测文本中噪声、异常字符或污染模式。
├── Text normalization validity
│   └── 检查文本规范化操作是否符合预期。
├── Text semantic preservation
│   └── 检查文本处理前后语义是否保持。
├── QAE grounding alignment
│   └── 检查问答/证据/视觉 grounding 是否对齐。
└── Coherence score
    └── 评估文本或多模态描述的连贯性。
```

## 4. 当前 pipeline 接入设计

pipeline 侧已有插件机制，核心文件位于：

```text
curator/core/
├── pipeline.py
└── pipeline_plugin.py
```

### 4.1 插件基类：`PipelinePlugin`

`pipeline_plugin.py` 中的 `PipelinePlugin` 定义 pipeline 生命周期 hook，例如：

```text
PipelinePlugin
├── on_pipeline_init(...)
├── before_operation(...)
├── after_operation(...)
└── on_pipeline_finish(...)
```

ops-quality 最适合接入 `after_operation`，因为质量评估通常需要比较 operation 处理前后的数据快照。

### 4.2 已有插件：`OpsQualityPlugin`

`pipeline_plugin.py` 中已有 `OpsQualityPlugin`，其职责是：

- 在 pipeline 初始化时创建 `OpsQualityAssessor`。
- 在每个 operation 执行后抽样 before/after 数据。
- 调用 `OpsQualityAssessor.evaluate_rows(...)` 执行质量评估。
- 将评估结果保存到插件内部，供 pipeline 结束后查询或报告。

当前典型接入方式：

```python
from curator.core.pipeline import MultimodalPipeline
from curator.core.pipeline_plugin import OpsQualityPlugin

ops_quality = OpsQualityPlugin(sample_size=8)
pipeline = MultimodalPipeline(
    enable_tracking=False,
    plugins=[ops_quality],
)
```

### 4.3 当前推荐报告格式

为了让评测结果能直接看出“哪个算子对应什么分数”，`OpsQualityPlugin.get_reports()` 建议按指标项输出 `report_items`，每一项至少包含：

- `test_metric`
- `related_operations`
- `score`
- `analysis`

示例结构：

```json
{
  "stage": "after_operation",
  "operation_name": "image_captioning",
  "op_index": 2,
  "duration_sec": 0.31,
  "report_items": [
    {
      "test_metric": "image_text_alignment",
      "related_operations": ["image_captioning"],
      "score": {
        "clip_score_mean": 0.82,
        "low_alignment_rate": 0.04
      },
      "analysis": "得分细项: clip_score_mean=0.8200, low_alignment_rate=0.0400",
      "status": "warning"
    }
  ]
}
```

如果是 Markdown 或控制台展示，建议也保持同样顺序，用无序列表写成：

```markdown
- 测试指标: `image_text_alignment`
  涉及算子: `image_captioning`
  得分: `clip_score_mean=0.8200`, `low_alignment_rate=0.0400`
  解析: caption 与图像主体整体对齐，但仍有少量低对齐样本。
```

## 5. 建议新增的 pipeline 函数入口

为了让用户在 pipeline 中更直观地启用 ops-quality，可以新增一个轻量 factory 函数，例如 `ops_quality_plugin(...)`。

该函数不替代已有 `OpsQualityPlugin`，只作为更易读、更易导入的创建入口。

建议放置位置：

```text
curator/core/pipeline_plugin.py
```

建议新增函数：

```python
def ops_quality_plugin(
    *,
    sample_size: int = 8,
    metrics: Optional[Sequence[str]] = None,
    enabled: bool = True,
) -> OpsQualityPlugin:
    return OpsQualityPlugin(
        sample_size=sample_size,
        metrics=metrics,
        enabled=enabled,
    )
```

使用方式：

```python
from curator.core.pipeline import MultimodalPipeline
from curator.core.pipeline_plugin import ops_quality_plugin

pipeline = MultimodalPipeline(
    enable_tracking=False,
    plugins=[
        ops_quality_plugin(sample_size=8),
    ],
)
```

## 6. Git 新增文件视角的建议展示

如果将 pipeline 接入能力作为一次 Git 变更展示，可以组织成如下文件树：

```text
DataForge/
├── curator/
│   ├── core/
│   │   ├── pipeline.py
│   │   │   └── 可选：在默认插件初始化中接入 ops-quality 开关
│   │   └── pipeline_plugin.py
│   │       ├── 已有：PipelinePlugin
│   │       ├── 已有：OpsQualityPlugin
│   │       └── 新增：ops_quality_plugin(...) factory 函数
│   └── quality/
│       └── ops_quality_assessor/
│           ├── assessor.py
│           │   ├── 已有：DEFAULT_METRIC_ASSESSORS
│           │   └── 已有：OpsQualityAssessor
│           ├── context.py
│           │   └── 已有：EvaluationContext / EvaluationContextBuilder
│           ├── base.py
│           │   └── 已有：BaseMetricAssessor
│           ├── results.py
│           │   └── 已有：MetricResult
│           ├── metrics/
│           │   └── 已有：各具体质量指标 assessor
│           └── OPS_QUALITY_DESIGN_AND_PIPELINE_PLUGIN.md
│               └── 新增：当前 ops-quality 设计、功能与 pipeline 接入说明
└── tests/
    └── test_ops_quality_assessor.py
        └── 可选：新增/扩展 ops_quality_plugin pipeline 接入测试
```

## 7. 推荐最小新增内容

若只希望最小化改动，建议新增内容如下：

```text
新增文件/函数
├── curator/quality/ops_quality_assessor/OPS_QUALITY_DESIGN_AND_PIPELINE_PLUGIN.md
│   └── 文档：说明当前设计、功能和 pipeline 插件接入方式
└── curator/core/pipeline_plugin.py
    └── 函数：ops_quality_plugin(...)，返回 OpsQualityPlugin 实例
```

最小新增函数示例：

```python
def ops_quality_plugin(**kwargs):
    return OpsQualityPlugin(**kwargs)
```

如果希望类型更明确，可以使用显式参数版本：

```python
def ops_quality_plugin(
    *,
    sample_size: int = 8,
    metrics=None,
    enabled: bool = True,
):
    return OpsQualityPlugin(
        sample_size=sample_size,
        metrics=metrics,
        enabled=enabled,
    )
```

## 8. 后续可选增强

后续如果需要把 ops-quality 作为 pipeline 默认能力，可以进一步考虑：

```text
可选增强
├── pipeline.py
│   ├── 新增 enable_ops_quality 参数
│   ├── 新增 ops_quality_sample_size 参数
│   └── 在 _init_default_plugins() 中按开关注入 OpsQualityPlugin
├── pipeline_plugin.py
│   └── 暴露 ops_quality_plugin(...) factory
└── tests/
    └── 增加 pipeline 默认启用/手动启用两类测试
```

示例 API：

```python
pipeline = MultimodalPipeline(
    enable_tracking=False,
    enable_ops_quality=True,
    ops_quality_sample_size=8,
)
```

该方案会改变 `MultimodalPipeline` 构造参数，属于更显式的 pipeline 级功能接入；若只是希望新增可用入口，优先选择 `ops_quality_plugin(...)` factory 即可。
