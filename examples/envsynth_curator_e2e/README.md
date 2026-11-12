# Curator -> DataForge -> Gym 环境合成 E2E 示例

这是一个可直接运行的正式示例，覆盖 Curator → DataForge → Gym 三组件环境合成链路：

```text
真实 DOCX/PDF/PNG
    │
    ▼
Curator parsed-doc-v2（文档解析 + 图片 OCR）
    │
    ▼
data_designer.pipelines.envsynth.cli.source_import
    │
    ▼
data_designer.pipelines.envsynth.cli.synthesis_worker
    -> ArtifactCatalog(env_release + taskset)
    │  + release 文件镜像
    │  + 一条 gym-env-package-v1.zip
    ▼
scripts/gym_publish_env.py -> Gym overlay 发布（只放置/登记，不改包）
    │
    ▼
Gym load_package_zip() -> office profile seed/step/verify
```

示例会现场生成三个真实输入文件，不依赖测试 fixture：

- `seed.docx`：表头为“工单编号 / 标题 / 状态”的电力工单样例；
- `regulation.pdf`：包含审批和驳回规则的规程文本；
- `scan.png`：包含工单字段的扫描件，由 Curator `ImageOcrOperation` 在 CPU 上完成 OCR。

DOCX/PDF 经过 Curator 的 `parse_inputs`，PNG 经过 Curator 的图片 OCR
算子；两者统一产出 `parsed-doc-v2` 行。DataForge 只产出一条
Gym-native 包；包根就是 Gym schema，DD 富任务与 anchors 在 `audit/` 中。

## 前置条件

- Python 3.10+
- `data-forge` 从当前仓库安装；Curator 已随仓提供，无需额外 checkout：

  ```bash
  pip install -e .[docs,ocr-paddleocr]
  ```
- 若要验证 Gym 运行期，需要一个包含
  `overlay/resources_servers/synth_env` 的 Gym checkout（PR #7 或之后），
  并可通过 `--gym-root` / `DATA_FORGE_GYM` / sibling checkout 发现；
  Gym 依赖需在运行示例的 Python 环境中可用。
- 文档与图片依赖可用：`python-docx`、`pdfplumber`、`jsonschema`、`Pillow`。
- OCR 后端可用：默认使用 `paddleocr`，也支持 `mindocr`；CPU 即可运行，
  不需要 NPU。

## 运行

从仓库根目录执行：

```bash
python examples/envsynth_curator_e2e/run_example.py \
  --work-dir /tmp/envsynth-curator-e2e \
  --gym-root /path/to/AscendAgent-Gym \
  --ocr-backend paddleocr \
  --ocr-device cpu \
  --ocr-lang en
```

`--work-dir` 必须尚不存在；省略时使用唯一临时目录。`--gym-root` 省略时
先查 `DATA_FORGE_GYM` 与 sibling checkout；都找不到时仍会完成
DataForge 构建与结构化发布，只是 `gym.runtimeVerified=false`。
`--ocr-backend` 可选 `paddleocr`、`mindocr` 或 `auto`，默认
`paddleocr`；`--ocr-device` 默认 `cpu`。

成功时退出码为 0，并打印 `example_summary.json` 内容。摘要中的：

- `curator.schema` 应为 `parsed-doc-v2`；
- `curator.mediaTypes` 应为 `docx`、`pdf` 和 `png`；
- `curator.ocr.backend` / `device` / `modelId` 记录实际 OCR provenance；
- `dataDesigner.release` / `taskset` 为 `ArtifactCatalog` 冻结结果；
- `gym.status` 应为 `published`，`gym.runnable` 应为 `true`；
- 指定 Gym checkout 时，`gym.runtimeVerified` 应为 `true`，且
  `gym.runtime.tasks[*].verified` 全部为 `true`。

## 输出目录

```text
<work-dir>/
├── inputs/
│   ├── seed.docx
│   ├── regulation.pdf
│   └── scan.png
├── source-input.json
├── seed-import/
│   ├── seed_manifest.json
│   ├── domain_dataset.json
│   └── parsed/documents.jsonl
├── compile-input.json
├── artifact-store/                    # env_release + taskset 不可变制品
├── snapshot.json                      # C-WK-1 快照
├── assets/<refId>/v<version>/
│   ├── package/                       # Gym-native 包根
│   ├── gym-env-package-v1.zip         # 唯一被 Gym 加载的产物
│   ├── env_release.json
│   ├── release.json
│   └── tasks.json
├── gym/                               # 未发现 Gym checkout 时的结构化发布落点
│   └── overlay/benchmarks/synth-.../
│       ├── assets/gym-env-package-v<version>-<fingerprint>.zip
│       ├── prepare.py
│       ├── config.yaml
│       ├── manifest.yaml
│       └── publish.json
└── example_summary.json
```

指定真实 Gym checkout 时，发布落点是该 checkout 的 `overlay/`，而不是
`<work-dir>/gym`。

## 不覆盖的内容

- Gym eval profile 与全量 rollout；宿主 HTTP 路由层未在示例中启动；
- 真实模型 agent 模式（本示例使用确定性的 `template` 模式）；
- FSM timeout、自然语言 assertions、DD event_types 的 Gym 投影；
- OCR 精度/吞吐基准；示例只验证 CPU OCR 到 Gym 环境包的契约与执行链路；
- bge/MindIE embedding、NPU 加速运行态和第二任务族。

这些边界与 `docs/环境合成接入设计_v0.2.md` 的后续工作清单一致。

示例本身有回归测试：`tests/envsynth/test_curator_example.py`。
