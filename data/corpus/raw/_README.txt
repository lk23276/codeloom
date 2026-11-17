示例语料（data/corpus/raw/）
================================

本目录下的 demo_* 文件是检索链路的示例语料，随仓库发布，
目的只有一个：克隆之后无需准备材料，就能跑通「切分 -> 建索引 -> 检索」。

    demo_echo.h     代码：类声明
    demo_echo.cpp   代码：实现（accept / 读 / 写 / 关闭）
    demo_notes.md   文档：设计说明，按标题层级切分
    demo_adr.md     文档：一条 ADR，演示决策类文本

用法：

    uv run python scripts/build_corpus.py
    uv run python scripts/index_corpus.py

两点说明：

1. 这些文件是示例，不是生产代码，也不参与任何评测结论。
   eval/baseline.json 里的数字来自另一套完整语料（7 文件 / 2237 行 / 139 块），
   用本目录的示例语料重跑评测不会复现那些数字。

2. 自己的语料直接放进本目录即可，支持 .cpp / .cc / .c / .h / .hpp / .md / .markdown，
   子目录会递归扫描。文件名不要用中文与空格，行号引用会用到它。
   注意：.gitignore 只放行 demo_* 与本说明文件，自有语料不会进仓库。

本文件是 .txt，不会被加载器当成语料。
