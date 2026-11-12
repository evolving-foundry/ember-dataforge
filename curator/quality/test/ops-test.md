规则：
1. 若需要进行测试实验，请先激活项目所需的 Python 环境。
2. 确保评测的输入源于真实的算子输出
3. 确保评测输入数据来自仓库根目录下的 `data/`。
4. 不允许添加任何 mock/monkeypatch 来逃避真实的评测
5. 不要用 Read 工具来读代码

补充教训：
1. 运行测试或生成 fixture 时使用项目环境中的 Python，避免误用系统 Python。
2. OpsQualityAssessor 的 fixture 必须保存真实算子产物；segments、samples、description_units、temporal_structures、scaffold_units、qae_triplets 不应在测试里手写拼装。
3. Megamind 相关 fixture 应落在仓库根目录的 `data/` 下，采样帧也应保存在 `data/ops_quality_sample_frames` 这类 data 子路径中。
4. video_scaffold_construction 依赖 temporal_structures；即使只验证 a1/a3/a4/a7/a8 链，也需要先运行 video_temporal_structure 来产生真实时间结构。
5. 不要默认依赖 Hugging Face 在线下载或不完整缓存；生成描述前先确认本地模型缓存可用，必要时使用明确的本地 model_path(从huggingface的cache 和 data/llm_models 两条路径去搜索)
6. qae_grounding_alignment 在没有真实 llm_client 时可能返回 failed_precondition；不能用 mock/monkeypatch 规避这个真实前置条件。
7. 使用本地离线模型时优先从配置的本地模型目录选择同时包含 config、preprocessor/tokenizer 和 safetensors 的完整目录；不要只凭 Hugging Face cache 中存在权重文件就判断模型可用。
8. 旧版 torch 加载 .bin 权重可能触发 torch.load 安全版本限制；测试模型路径优先选择 safetensors 快照或完整 safetensors 本地模型目录。
