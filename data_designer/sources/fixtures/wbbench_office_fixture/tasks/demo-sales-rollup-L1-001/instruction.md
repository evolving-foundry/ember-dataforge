你是销售数据汇总助理。当前工作区提供了合成静态数据包：`input/sales_q1.csv`（列：region、product、units、revenue_cny）和 `input/region_codebook.json`（region 代码 → 标准名映射）。请只使用工作区提供的静态文件，不运行脚本、不联网、不查询外部数据。

请读取上述两个文件，生成一个文件 `output/summary.json`，包含以下字段：

- `total_revenue_cny`：所有 revenue_cny 非空行的收入合计（保留一位小数）。
- `by_region`：按标准 region 名（用 codebook 归一）分组的收入合计字典；缺失值行不计入。
- `top_region`：收入合计最高的标准 region 名。
- `excluded_rows`：revenue_cny 为空的行数（整数）。

关键口径：revenue_cny 为空的行不得计入合计，也不得补猜缺失值；region 必须用 codebook 归一为标准名，不得自由解释 codebook 外的别名。

请读取工作区中的输入文件：`input/sales_q1.csv`、`input/region_codebook.json`。
