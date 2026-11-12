你是会议纪要整理助理。当前工作区提供了合成静态资料包：`input/minutes_2026-02-10.md`、`input/minutes_2026-02-17.md`（两次项目例会纪要 markdown）和 `input/owner_roster.json`（成员花名册）。请只使用工作区提供的静态文件，不运行脚本、不联网。

请从两份纪要中提取所有行动项（action item），生成 `output/actions.json`：

- 顶层为数组，每条行动项包含字段：`owner`（用 roster 归一为标准姓名）、`due`（统一为 `YYYY-MM-DD`；纪要中只有周表述时按该周周五推算）、`action`（一句话描述）、`source_file`（来源文件名）、`status`（open / done，纪要未标注视为 open）。
- 已在第二次纪要中标记完成的行动项 `status` 写 `done`；不得把讨论中的事项当作行动项。

请读取工作区中的输入文件：`input/minutes_2026-02-10.md`、`input/minutes_2026-02-17.md`、`input/owner_roster.json`。
