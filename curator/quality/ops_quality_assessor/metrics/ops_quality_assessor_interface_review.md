# Ops Quality Assessor Input Interface Review

审阅日期：2026-04-24

审阅范围：

- `curator/quality/ops_quality_assessor`
- `curator/operations`

目标：检查 `ops_quality_assessor` 下各评测指标读取的字段，是否能和现有 `operations` 的默认输出字段自然衔接。

## 总体结论

当前接口只有一部分链路是完整的。视频分段、采样、描述、时间结构、脚手架、QAE 物化这条主链路的字段命名大体统一，例如 `segments`、`samples`、`description_units`、`temporal_structures`、`scaffold_units`、`qae_triplets`。但图像链路和 QAE grounding 链路存在明显断点：

- 图像相关 metrics 默认读 `image_path`，但 `ImageLoadingOperation` 直接返回 `DataLoaderFactory.load_dataset()` 的结果，并未补 `image_path`，实际 Ray image loader 更常见的是 `image` 数据列。
- `image_text_alignment` 当前有实现 bug：构造的是三元组 `(row, image_value, text_value)`，调用侧按二元组解包，会在存在图文对时直接失败。
- `qae_grounding_alignment` 的视觉证据索引只从 `evidence_units` 建，但当前注册算子几乎不会产出 `evidence_units`；`VideoQaeMaterializationOperation` 的 `evidence_refs` 通常指向 `description_units`、`text_signals`、`temporal_structures` 或 `scaffold_units`，导致视觉过滤默认拦截。
- `OpsQualityPlugin` 只自动构造 `before_text = before_row["text"]` 和 `after_text = after_row["text"]`。对于输出到 `output_text` 的文本增强算子，文本保持性/规范化指标会评估错字段。

<del>

## P0 Findings

### P0-1 `image_text_alignment` 有三元组/二元组不匹配，实际有图文对时会失败

位置：

- `metrics/image_text_alignment.py:61`
- `metrics/image_text_alignment.py:110`
- `metrics/image_text_alignment.py:132`

当前 `_build_pairs()` 声明并返回 `list[tuple[dict[str, Any], Any, str]]`，并 append `(row, image_value, text_value)`。但 `_assess()` 中使用：

```python
for image_value, text_value in pairs:
```

这会在 `pairs` 非空时触发 `ValueError: too many values to unpack`，被 `BaseMetricAssessor.assess()` 捕获后返回 `failed`。这不是字段缺失导致的不可用，而是 metric 自身实现和接口形状不一致。

影响：

- `ImageCaptioningOperation` 产出 `caption` 后，理论上正好满足 `image_text_alignment` 的评估场景，但指标会失败。
- `QaeGroundingAlignmentAssessor` 也复用了 `ImageTextAlignmentAssessor._score_from_embeddings()`，虽然不直接触发该解包问题，但图文 alignment metric 本身不可用。

修改建议：

- 将 `_build_pairs()` 改为返回 `(image_value, text_value)`。

### P0-2 `qae_grounding_alignment` 使用 `self.llm_client`，但 assessor 没有初始化这个成员

位置：

- `metrics/qae_grounding_alignment.py:226`
- `metrics/qae_grounding_alignment.py:227`

`_call_llm_judge()` 调用：

```python
response = self.llm_client.chat(...)
```

但 `QaeGroundingAlignmentAssessor` 没有 `__init__` 注入，也没有从 config 中创建 client。当前异常会被 `except Exception` 静默吞掉，表现为 `judgment is None`、`llm_judged_count=0`、最终支持率为 0 或缺失，而不是显式报错。

影响：

- 即使关闭视觉过滤，只要需要 LLM judge，当前指标也不会真实执行 LLM 判断。
- 报告会看起来像模型判断未通过，实际是 client 未配置。

建议：

- 在 config 中显式配置 LLM client/provider，或者把 LLM judge 设计成可选依赖并在缺失时返回 `failed_precondition`。
- 不建议静默吞掉所有异常；至少 metadata/notices 应暴露 `llm_client missing`。

</del>

<del>

## P1 Findings

### P1-1 图像 metrics 默认 `image_path`，但 image loading 默认没有补这个字段

位置：
- `operations/loading/image_loading_operator.py:41`
- `operations/loading/image_loading_operator.py:42`
- `utils/io/data_loader_factory.py:160`
- `utils/io/data_loader_factory.py:165`
- `metrics/image_text_alignment.py:23`
- `metrics/visual_robustness.py:21`
- `metrics/visual_transform_consistency.py:22`
- `metrics/chair_object_hallucination.py:22`

图像相关 metrics 默认读取 `image_path`：

- `image_text_alignment.image_field = "image_path"`
- `visual_robustness.image_field = "image_path"`
- `visual_transform_consistency.image_field = "image_path"`
- `chair_object_hallucination.image_field = "image_path"`

但 `ImageLoadingOperation` 只是调用 `DataLoaderFactory.load_dataset(data_paths)`，未像 `VideoLoadingOperation` 那样添加路径列。`DataLoaderFactory` 对 image 直接调用 Ray reader。现有示例中也经常使用 `display_image_results(..., image_field_name='image')`，说明实际数据列更可能是 `image`。

影响：

- `image_loading -> image_captioning -> image_text_alignment` 默认不连通，因为 `image_captioning` 默认也读 `image_path`。
- `image_random_crop/image_random_flip/image_canny -> visual_robustness/visual_transform_consistency` 默认要求原图字段为 `image_path`，但加载后若只有 `image`，metric 会 `failed_precondition: missing original image field`。

建议：

- 统一图像主字段约定：
  - 把图像相关 ops/metrics 默认 `image_key/image_field` 改为 `image`，不考虑`image_path`。


### P1-2 `OpsQualityPlugin` 对 before/after 文本的自动桥接只支持 `text`，不支持 `output_text`

位置：

- `core/pipeline_plugin.py:331`
- `core/pipeline_plugin.py:332`
- `core/pipeline_plugin.py:333`
- `operations/augmentation/remove_stopwords_operation.py:74`
- `operations/augmentation/random_noun_operation.py:96`
- `operations/augmentation/text_normalization_operation.py:91`

`OpsQualityPlugin` 每个 operation 后构造临时评估 row：

```python
row.setdefault("before_text", before_row.get("text"))
row.setdefault("after_text", after_row.get("text"))
```

这只适配会原地写回 `text` 的操作，例如：

- `TextNormalizationOperation`
- `RemoveRepetitionsPunctuationOperation`
- `SpellingCorrectionOperation`
- `StemmingLemmatizationOperation`

但这些操作输出到 `output_text`：

- `RemoveStopwordsOperation`
- `RandomNounAugmentationOperation`

对于这类操作，`after_row["text"]` 仍是原文，`after_row["output_text"]` 才是变换结果。当前 `text_semantic_preservation` 和 `text_normalization_validity` 会评估原文 vs 原文，给出虚假的高保持率或无变化结果。

建议：
- 或者要求所有文本增强算子统一原地更新同一个 `text_key`，避免 `text`/`output_text` 混用。

### P1-3 `video_image_alignment` 与 `video_resize_aspect_ratio` 的视频路径字段不衔接

位置：
- `operations/augmentation/video_resize_aspect_ratio_operation.py:60`
- `operations/augmentation/video_resize_aspect_ratio_operation.py:61`
- `operations/augmentation/video_resize_aspect_ratio_operation.py:97`
- `metrics/video_image_alignment.py:178`
- `metrics/video_image_alignment.py:190`
- `metrics/video_image_alignment.py:192`

`VideoResizeAspectRatioOperation` 默认输出新视频路径到 `output_video_path`，但 `VideoImageAlignmentAssessor` 固定从 row 的 `video_path` 抽取 segment midpoint 和 sample frame。它没有 `video_key`/`video_path_key` 配置。

影响：
- 如果 resize 后想评估 resized video 与 samples/segments 的一致性，metric 仍会读取旧 `video_path`。
- 如果 pipeline 中只有 `output_video_path` 而没有 `video_path`，且 `samples` 没有 `frame_path`，embedding 分支无法计算。

建议：
- 给 `VideoImageAlignmentConfig` 增加 `video_field: str = "video_path"`。
- 或要求 `VideoResizeAspectRatioOperation` 可选覆盖 `video_path`，并在文档中说明后续视频 metrics 依赖 `video_path`。

### P1-4 `chair_object_hallucination` 的 `text_field` 不支持 nested path 的字段存在性判断

位置：
- `metrics/chair_object_hallucination.py:147`
- `metrics/chair_object_hallucination.py:151`
- `metrics/chair_object_hallucination.py:152`

`_row_texts()` 先执行：

```python
if not context.has_field(metric_config.text_field):
    raise InvalidMetricInputError(...)
```

随后才根据 `.` 判断是否用 `get_path_values()`。因此如果配置为 `description_units[].description_text`，`context.has_field()` 会查找顶层字段名的完整字符串，必然失败。虽然当前默认是 `caption`，但这会阻断视频描述链路复用该 metric。

建议：

- 如果 `text_field` 包含 `.` 或 `[]`，应检查顶层字段是否存在，例如 `description_units`，而不是检查完整 path。
- 或明确该 metric 只支持顶层 `caption`，删除 nested path 支持代码，避免接口误导。

</del>

## P2 Findings

### P2-1 `text_noise_contamination` 报错文案提到 `caption`，实际默认候选不包含 `caption`

位置：

- `metrics/text_noise_contamination.py:95`
- `metrics/text_noise_contamination.py:100`

默认候选只有：

- `text`
- `input_text`

但缺失提示写的是 `expected one of: text, input_text, caption`。这会误导使用者以为 image caption 数据可以直接被该 metric 评估。

建议：

- 要么把 `caption` 加入候选字段；
- 要么把错误文案改成 `text, input_text`。

### P2-2 `video_image_alignment` 在 pipeline 中过早运行会产生预期内失败

位置：

- `metrics/video_image_alignment.py:52`
- `operations/augmentation/video_partition_operation.py:166`
- `operations/augmentation/video_sampling_operation.py:181`

`video_image_alignment` 需要同时存在 `segments` 和 `samples`。但 `VideoPartitionOperation` 只输出 `segments`，`VideoSamplingOperation` 才输出 `samples`。当前 `OpsQualityPlugin` 默认每个 operation 后都执行 enabled metrics，因此如果在完整视频 pipeline 中全程开启该 metric，它在 partition 后会先返回 `failed_precondition`。

建议：

- `OpsQualityPlugin` 支持按 operation name/stage 触发 metric。
- 或 metric 对“上游尚未完成”的状态返回 `not_applicable`，直到 `segments` 和 `samples` 同时出现。

### P2-3 `VideoSamplingOperation` 默认不保存 `samples[].frame_path`，部分图像/主体类 metric 的视频复用路径依赖这个字段

位置：

- `operations/augmentation/video_sampling_operation.py:223`
- `metrics/image_text_alignment.py:115`
- `metrics/chair_object_hallucination.py:24`
- `metrics/chair_object_hallucination.py:137`

`VideoSamplingOperation` 只有在 `save_sample_frames=True` 时才写 `frame_path`。`image_text_alignment` 和 `chair_object_hallucination` 都会尝试从 `samples[].frame_path` 取图像。对于没有 `image_path` 的视频数据，如果不保存采样帧，这两个 metric 无法直接复用视频 samples。

不过 `video_image_alignment` 可以从 `video_path + timestamp/frame_index` 重新抽帧，所以它本身不强依赖 `frame_path`。

建议：

- 文档中明确：图文/主体类 metric 若用于视频 samples，需要 `save_sample_frames=True`。
- 或这些 metric 也支持从 `video_path + samples[].timestamp_sec/frame_index` 动态抽帧。

## Metric 与 Operation 字段对照表

| Metric | 默认读取字段 | 可对应的 operation 输出 | 当前状态 |
| --- | --- | --- | --- |
| `text_noise_contamination` | `text` / `input_text`，可配置 `text_field` | `TextLoadingOperation`、文本过滤/清洗类保留字段 | 基本可用；默认不读 `caption/output_text` |
| `text_normalization_validity` | `before_text` / `after_text` 或若干 fallback | `OpsQualityPlugin` 临时注入；`TextNormalizationOperation` 原地写 `text` | 对原地写 `text` 的算子可用；对 `output_text` 算子不准 |
| `text_semantic_preservation` | `before_text` / `after_text` | `OpsQualityPlugin` 临时注入 | 同上；依赖 plugin 正确桥接 before/after |
| `image_text_alignment` | `image_path` + `caption`，也读 `samples[].frame_path` | `ImageCaptioningOperation` 输出 `caption`；`VideoSamplingOperation` 可输出 `samples[].frame_path` | 有实现 bug；图像主字段默认也可能不匹配 |
| `chair_object_hallucination` | `image_path` 或 `samples[].frame_path` + `caption` | `ImageCaptioningOperation`、`VideoSamplingOperation(save_sample_frames=True)` | 图像字段默认可能不匹配；nested text path 判断不通 |
| `visual_robustness` | `image_path` + `crop_image_data_list/flip_image_data_list/canny_image_data` | `ImageRandomCropOperation`、`ImageRandomFlipOperation`、`ImageCannyAugmentationOperation` | transform 字段对得上；原图字段默认可能不匹配 |
| `visual_transform_consistency` | 同上 | 同上 | transform 字段对得上；原图字段默认可能不匹配 |
| `video_image_alignment` | `video_path` + `segments` + `samples` | `VideoLoadingOperation`、`VideoPartitionOperation`、`VideoSamplingOperation` | 主链路可用；不支持 `output_video_path`；过早评估会失败 |
| `coherence_score` | `segments` + `description_units` | `VideoPartitionOperation`、`VideoMultigranularityDescriptionOperation` | 字段对得上 |
| `qae_grounding_alignment` | `qae_triplets` + evidence text/visual index | `VideoQaeMaterializationOperation` 输出 `qae_triplets` | QAE 字段对得上；evidence visual index 与上游 refs 不匹配；LLM client 缺失 |

## 建议的修复优先级

1. 修复 `image_text_alignment` tuple shape bug。
2. 统一图像主字段，决定默认使用 `image` 还是 `image_path`。
3. 重写 `qae_grounding_alignment._build_visual_index()`，让它覆盖 `samples`、`description_units`、`text_signals`、`temporal_structures`、`scaffold_units`。
4. 为 `qae_grounding_alignment` 显式接入 LLM client，缺失时返回 `failed_precondition`。
5. 让 `OpsQualityPlugin` 的 before/after 文本桥接支持 `output_text` 或 operation-aware 配置。
6. 给 `VideoImageAlignmentConfig` 增加 `video_field`，支持 `output_video_path`。
7. 为 metric 增加 stage/operation 触发控制，避免 pipeline 中过早评估产生噪声。

## 推荐的接口契约

建议后续明确以下最小契约：

- 图像样本：顶层主字段统一为 `image`；如果两者共存，metric 按 `image_path -> image` fallback。
- 文本变换：所有文本增强算子写法不变，由 plugin 自动映射到 `after_text`
- 视频采样：如果后续图像类 metric 要消费 samples，应开启 `save_sample_frames=True` 并保证 `samples[].frame_path` 可读。
- QAE evidence refs：`evidence_refs` 必须能在统一 evidence index 中解析到文本证据；若开启视觉评估，还必须能解析到 frame/image 证据。
- Metric 触发：每个 metric 声明最早可运行的 upstream operation，例如 `video_image_alignment` 至少在 `video_sampling` 之后运行。
