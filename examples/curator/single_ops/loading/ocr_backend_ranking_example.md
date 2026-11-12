# OCR 后端效果排名端到端示例

本示例验证的是**真实 OCR 推理 + 真实指标计算 + 真实 `auto` 排名路由**，
不是用 fake adapter 或合成指标做契约测试。

示例会依次完成：

1. 读取带人工真值和 bbox 的图片 manifest。
2. 调用 MindOCR / PaddleOCR 官方默认模型执行 OCR。
3. 对候选按 `top,left` 排序并把单词框合并为整行，计算 CER、行准确率、
   layout F1 和吞吐，写出 `ocr-ranking-v1` YAML。
4. 使用 `backend=auto` + `backend_ranking_path` 运行 `ImageOcrOperation`，并检查
   `provenance.routing_reason="ranking"`。

## 文件

| 文件 | 说明 |
|---|---|
| `ocr_backend_ranking_example.py` | 基准、排名写出和 `auto` 路由验证脚本 |
| `assets/ocr_ranking/manifest.jsonl` | 3 张图片、18 行文本的人工真值 manifest |
| `assets/ocr_ranking/invoice.png` | 发票类测试图片 |
| `assets/ocr_ranking/receipt.png` | 小票类测试图片 |
| `assets/ocr_ranking/report.png` | 报表类测试图片 |
| `assets/ocr_ranking/ocr_backend_ranking.yaml` | 本次真实推理生成的示例排名表 |
| `assets/ocr_ranking/ocr-auto-route.jsonl` | `backend=auto` 命中排名的完整输出 |

示例 manifest 由 `create_sample_assets()` 可重复生成；图片内容为确定性文档图像，
但 OCR 结果和指标全部来自真实模型执行。

## 运行

MindOCR 与 PaddleOCR 的锁文件对 NumPy 的约束不同，建议分别在隔离环境中执行，
再用 `--merge` 合并排名行。

PaddleOCR CPU：

```bash
python examples/curator/single_ops/loading/ocr_backend_ranking_example.py \
  --backend paddleocr --device cpu --lang en --allow-model-download \
  --ranking-output examples/curator/single_ops/loading/assets/ocr_ranking/ocr_backend_ranking.yaml \
  --output-dir data/output/ocr_backend_ranking_paddle
```

MindOCR CPU：

```bash
python examples/curator/single_ops/loading/ocr_backend_ranking_example.py \
  --backend mindocr --device cpu --lang en --allow-model-download --merge \
  --ranking-output examples/curator/single_ops/loading/assets/ocr_ranking/ocr_backend_ranking.yaml \
  --output-dir data/output/ocr_backend_ranking_mindocr
```

两个后端锁定了不同的 NumPy 主版本，不支持在同一环境同时安装。必须先分别在隔离环境生成排名行，再用 `--merge` 合并；不要把两个 extra 装进同一个环境。

仅验证已有排名表的跨后端选择，不重新跑基准：

```bash
python examples/curator/single_ops/loading/ocr_backend_ranking_example.py \
  --route-only --device cpu --lang en --allow-model-download \
  --ranking-output examples/curator/single_ops/loading/assets/ocr_ranking/ocr_backend_ranking.yaml \
  --output-dir examples/curator/single_ops/loading/assets/ocr_ranking
```

## 验证结果（2026-11-02，CPU）

验证环境：WSL Linux x86_64、Python 3.10.22。

| 后端 | 版本 | 模型包 | CER | 行准确率 | layout F1 | 吞吐（页/秒） |
|---|---|---|---:|---:|---:|---:|
| PaddleOCR | 3.7.0 | `paddleocr-official-ppocr-default-v1` | 0.00348432 | 0.94444444 | 1.0 | 0.20718002 |
| MindOCR | 0.5.0 | `mindocr-official-dbnet_mobilenetv3-crnn_resnet34-crnn_resnet34_ch-ppocr_keys_v1-v1` | 0.05923345 | 0.5 | 0.94444444 | 0.34522724 |

排名按 CER 优先，因此 PaddleOCR 排第一。`--route-only` 在同时安装两个适配器的
环境中执行，实际选中 PaddleOCR，并写出：

```text
backend=paddleocr
device=cpu
routing_reason=ranking
ranking_version=ocr-ranking-demo-v1
ranking_evaluation_set=ocr-ranking-demo-en-v1
```

完整输出见 `assets/ocr_ranking/ocr-auto-route.jsonl`；输出中
`provenance.backend`、`model_id` 与排名第一行一致。

指标在比较前执行 NFKC、casefold，并忽略空白和标点；OCR 原始输出不变。
MindOCR 的英文 CRNN 仅支持数字和小写字母，因此原始文本仍会丢失大小写和标点，
但不会再让格式差异覆盖真正的内容识别差异。单词框会先合并为整行，避免把 56 个
单词框与 18 个整行真值直接比较。

## 边界

- 这是 3 张图片、18 行文本的演示评测集，不能替代生产评测集，也不得直接复制为
  `curator/operations/loading/ocr_backend_ranking.yaml` 包内默认排名。
- 示例证明真实指标计算、排名写出和 `auto` 路由链路可用；不代表生产语料上的
  OCR 精度结论。
- MindOCR 与 PaddleOCR 建议隔离环境运行。组合环境的 MindSpore import 可能受
  oneDNN/MKL 符号冲突影响；`--route-only` 不重新执行 MindOCR，只验证已有真实
  排名行的选择和被选中后端的实际推理。
- NPU 训练/推理环境未在本次示例中验证。
