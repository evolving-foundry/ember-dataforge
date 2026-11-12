# curator-quality — Curator 指标面接入 Evaluator

对应设计：`docs/Evaluator_接入设计_v0.1.md`（Q1 指标面契约 §4）。本目录交付
TS 薄插件 + 配置样例；一切指标计算都在 Python 侧
（`python -m curator.trusted_worker.op`，op 模式），插件只做装配与转发。

## 组成

| 文件 | 说明 |
| --- | --- |
| `curator-quality.mjs` | metrics 插件（编译后 JS，Evaluator 经 cordis.patch 直接加载） |
| `cordis.patch.example.yaml` | patch 样例 |

## 接入步骤

1. 准备 Curator Python 环境（`curator` 可导入；op 模式算子依赖 `ray`）。
2. 拷贝 `curator-quality.mjs` 到实验目录（如 `./plugins/`），并放置
   `cordis.patch.yaml`（参照 `cordis.patch.example.yaml`）。
3. 实验 YAML 的 `metrics` 段加入 `curator-quality`（可与内置
   `success-rate` / `duration` / `usage` 并存）：

```yaml
metrics:
  - name: curator-quality
    config:
      pythonExecutable: <venv>/bin/python   # 缺省 python；含路径分隔符时经 resolvePath 展开
      op: CandidateLogDeduplicate           # 或 CandidateBatchQualityAssess
      # includeStatuses: [completed]        # 参与计算的 attempt 状态；缺省仅 completed
      # params: {}                          # 算子参数透传（如 quality_config 覆盖项）
      # env: { PYTHONPATH: <curator 仓根或 venv> }   # 传给 Python 子进程（显式给出，无 venv 探测）
      # timeoutMs: 120000                   # 子进程超时；缺省 120s，超时 SIGTERM
```

validate 拒绝未知键；路径类配置经插件 factory 的 `runtime.resolvePath` 展开。

## 指标键名（即契约，改名属破坏性变更）

| op | metrics |
| --- | --- |
| `CandidateLogDeduplicate` | `{input_count, kept, removed, dedup_ratio}`；`dedup_ratio = removed / input_count`（对 run 内全部 finalText 去重，量化 mode collapse） |
| `CandidateBatchQualityAssess` | `{quality_score, completeness, consistency, uniqueness, text_quality, issues_count}`；维度被 `params.quality_config.enabled_dimensions` 关闭时对应键为 `null`（不伪造 0 分） |

`params` 透传说明：`CandidateBatchQualityAssess` 接受
`quality_config`（`sample_size` / `null_threshold` / `enabled_dimensions` /
`dimension_weights` / `text_fields`；缺省值即公共 API，改动需双宿主回归）；
`CandidateLogDeduplicate` 无参数。

## 失败语义（红线，§4.5）

| 场景 | 行为 |
| --- | --- |
| 空输入（0 条可参与记录） | 插件短路返回 `{"eligible": 0}`，**不 spawn** |
| 请求不合法 / 数据行缺字段（worker 退出码 2） | 抛错 → `metrics.compute` 失败 → run 退出码 1，无 metrics.json / report.md |
| op 异常 / 数据不可读 / 非有限分数（worker 退出码 1，stderr 结构化诊断） | 同上；**不产伪造指标** |
| 子进程超时 | SIGTERM → 同上 |

指标面 fail-closed 由引擎错误路径决定（指标失败 = 报告失败）；不提供
`onError: skip` 降级（静默缺指标比失败更难排查）。

## 确定性

同 run 目录 `agent-evaluator report` 离线重算，Q1 指标逐字节一致
（Python 侧同快照 + 同参数重算确定；Ray 去重输出顺序由算子冻结）。

## 历史端到端验证

原 `examples/evaluator-e2e/` fixture 已随退役示例清理移除，不再提供可执行
入口。插件契约与 Python worker 仍由
`tests/curator/test_trusted_worker_op_mode.py` 覆盖；历史实测结论仅保留在
本文档中，不作为当前环境验收承诺。

RSI 循环内直调（evaluation 模式）的端到端见 RSI 仓
`tests/native/curator-e2e.test.ts`：真实 hill-climb 两轮经 command evaluator
spawn 本生产包打分，RSI 仓零 src 改动。注意：RSI `assertPlan` 必带
`plan.targetConfigId`，worker 端已放行该字段（不消费）。

## 语义约束备忘

- `timeout/error/cancelled` 的 attempt 缺省不参与计算（uncollected work
  不进指标，与引擎红线一致）；如需纳入，显式配置 `includeStatuses`。
- 数据经请求 JSON 内 `dataPath` 文件引用（插件写入系统临时目录，run 结束
  后清理），不内联；Q1 按单文件实现（>100MB 量级再议分片，C6）。
- op 产物（deduped 明细等）落在临时 `outputDir`，不纳入 Evaluator store
  制品清单（C7：指标是派生数据，原始 finalText 已在 attempt 落盘）。
- Q2 软评分算子 `CandidateTextOverlapScore` 已在 op 模式注册表
  （`bleu`/`cider`/`ngram`），供 DataForge E1 桥或 benchmark 插件的
  `verify` 消费；不经本插件（本插件只承载 run 级指标面）。
