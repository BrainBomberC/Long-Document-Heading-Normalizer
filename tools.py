"""
MarkdownHeadingNormalizer - 目录驱动的长书标题标准化工具。

这个脚本处理 PDF/EPUB/OCR 转 Markdown 后常见的标题污染问题：目录条目、
广告、代码、图片 OCR 文本可能被错误标成 `#`，真正正文标题又可能等级错乱。

核心思想是“先确定目录，再让正文服从目录”。目录提供完整标题树，正文只负责
提供标题出现位置和正文内容。没有真实目录的文件不强行转换，直接写入
`no_toc_region` 错误 JSON。

五层网络:
  预处理: 提取原文 `#` 候选，跳过代码块中的伪标题。
  第一层: LLM 定位真实目录区域，tail 扫描继续补足目录尾部。
  第二层: LLM 将目录原文解析成 `{"text", "level"}` 标题列表。
  第三层: 按完整 Part/章节让 LLM 重建 parent tree，再确定性计算标题等级。
  第四层: 在正文候选中匹配目录标题，给正文标题赋予目录等级。
  第五层: 基于目录树补齐可定位的缺失父级标题。
  输出层: 只输出目录和正文；目录项为普通列表文本，正文标题保留 Markdown 标题。

常用命令:
  python tools.py book.md
  python tools.py data_folder --out-dir output --json-dir heading_json --resume
"""

import hashlib
import difflib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path


# ============================================================
# 常量
# ============================================================

BASE_URL = "http://brain-X99:8000/v1/chat/completions"
MODEL = "qwen3-30b"
API_KEY_ENV = "LOCAL_LLM_API_KEY"
API_KEY_OVERRIDE = None
LLM_USE_VLLM_EXTRAS = True
OUTPUT_TOKENS = 2000                     # LLM 单次最大输出 token 数
TIMEOUT = 600                            # API 请求超时（秒）
CACHE_VERSION = "v21_fast_preserve_atomic"  # 缓存版本标识，修改后旧缓存自动失效
TOC_PARSE_SINGLE_MAX_LINES = 220         # 目录清洗后 ≤ 此行数则单批发给 LLM，不分块
TOC_PARSE_CHUNK_LINES = 120              # 分块模式时每块最多发送多少行
TOC_PARSE_CHUNK_MAX_TOKENS = 8000        # 分块模式时每块 LLM 输出 token 上限
LAYER2_LOGICAL_CORE_LINES = 12            # 第二层逻辑标题恢复每批负责的原文行数
LAYER2_LOGICAL_CONTEXT_LINES = 8          # 第二层逻辑标题恢复每批前后重叠上下文
LAYER2_LEVEL_CHUNK_TITLES = 160           # 第二层初始等级分配每批负责的逻辑标题数
LAYER5_POST_MAX_RANGE_TOKENS = 40000     # 第五层后机制单个缺失块允许的最大正文范围
LAYER5_POST_SINGLE_TITLE_MAX_RANGE_TOKENS = 50000  # 单个缺失标题可适度放宽，避免略超阈值误跳过
LAYER5_POST_MIN_BAD_ANCHORS_TO_SKIP = 30  # 错乱真实锚点达到该数量后才触发可信度熔断
LAYER5_POST_MAX_BAD_ANCHOR_RATIO = 0.35   # 错乱真实锚点占比超过该值，认为正文锚点不可信
LAYER5_POST_MAX_LARGE_FILL_TITLES = 100   # 第五层后机制单次最多可硬补的缺失标题规模
LAYER4_PREVIEW_LLM_MAX_CALLS = 24         # 第四层最多让 LLM 判断多少个疑似导读/预告锚点
LAYER4_PREVIEW_SIBLING_LINE_SPAN = 100    # 疑似导读块中兄弟标题的最大行距
LAYER4_PREVIEW_LATER_LINE_SPAN = 3000     # 查找同一目录标题后续真实候选的最大行距
TOC_TAIL_CHUNK_LINES = 200                 # 第一层 tail 每轮检查的新文本行数
TOC_TAIL_OVERLAP_LINES = 30                # tail 每轮携带的前文重叠上下文


class SkipMarkdownFile(RuntimeError):
    """当前文件不可靠，批量处理时应跳过。"""

    def __init__(self, error: str, stage: str, message: str):
        """记录跳过原因、失败阶段和面向日志/JSON 的错误说明。"""
        super().__init__(message)
        self.error = error
        self.stage = stage
        self.message = message


TOC_PAGE_RE = re.compile(r"(?<!\d)(?:[…·.．]\s*)+\d{1,5}\s*$")


def _has_toc_page_marker(text: str) -> bool:
    """判断一行是否像目录页码条目，例如 `1.1 引言 …… 12`。"""
    return bool(TOC_PAGE_RE.search((text or "").strip()))


# ---- 第一层 Prompt: 目录定位 ----
TOC_DETECT_SYSTEM = """你是书籍结构分析师。你的任务不是寻找包含“目录”二字的标题，而是识别**全书正文目录 book_toc**，并给出完整覆盖该目录区域的宽松范围。
后续步骤会精确解析目录，所以 book_toc 的范围宁可稍大，不能偏小；但假目录必须拒绝。

## 你必须先做目录类型分类
只接受 toc_type="book_toc"。其他类型都不能进入后续处理。

可返回的 toc_type:
- book_toc: 全书正文目录/目次/Table of Contents。后面是密集的全书篇、章、节、小节、附录、参考文献等标题序列。
- figure_toc: 图目录、插图目录、List of Figures。
- table_toc: 表目录、List of Tables。
- index: 索引/Index，但不是正文目录本身。
- body_section: 正文中的普通小节，例如“查看目录和文件”“文件目录管理”“目录结构说明”“工作目录”。
- front_matter: 前言、本书内容体系、章节简介、各章内容、读者对象等介绍性文字。
- unknown: 证据不足，不能确认是全书正文目录。

## book_toc 的判断标准
1. 后续是密集标题序列，通常覆盖多个篇/章/节/附录，而不是某一章内部几个小节。
2. 常见起点包括: 目录/目次/Contents/Table of Contents、前言、第1篇、第1章、Chapter 1、Part I、1.1。
3. 条目之间通常只有空行或短续行，少有完整正文段落、代码块、命令讲解。
4. 可带页码，也可无页码；页码不是必要条件。
5. 如果候选在正文很深处，例如已经出现多个正文章节之后，必须极其谨慎；除非它后面明显重新出现完整的全书目录，否则不要判为 book_toc。
6. 如果出现独立的目录标记，例如单独一行“目录/目次/Contents/Table of Contents”，且其后紧跟第1章、1.1、第2章、2.1 等跨多章密集标题序列，应优先判为 book_toc。
7. 标题中“包含目录二字”不等于目录标记。带章节编号的“3.3 目录与文件”“3.3.5 移动目录或文件”“查看目录和文件”通常只是目录条目或正文小节，不是 book_toc 起点。

## 必须排除的假目录
1. “图目录/表目录/插图目录/List of Figures/List of Tables”不是 book_toc。
2. “查看目录和文件/文件目录/目录管理/工作目录/目录结构”通常是正文小节，不是 book_toc。
3. “本书内容体系/章节简介/各章内容/读者对象”通常是 front_matter，不是 book_toc。
4. 如果候选后面紧跟正文段落、代码块、命令说明、示例输出，它不是 book_toc。
5. 如果候选只从 5.5、25.3、图目录等中途位置开始，而前文已有正文结构，通常不是 book_toc。
6. 编号小节中出现“目录/文件目录/工作目录/目录结构”时，只能作为假候选或普通目录条目处理；不能因为这些假候选存在，就否定前面已经成立的独立 book_toc。

## 多个候选的处理
- 不要机械选择最早候选。先比较每个候选覆盖的标题数量、编号连续性和层级深度，选择信息最完整的全书详细目录。
- 书籍可能先出现简略目录 `brief contents / contents at a glance / 简目 / 章节速览`，随后出现详细目录 `contents / table of contents / 目录 / 目次`。两者同时存在时，必须选择后面的详细目录作为唯一 book_toc，toc_start_line 从详细目录开始，不要把两份目录合并。
- 即使后面的详细目录没有再次出现 `contents/目录` 标记，只要它在简略目录后重新从 Part 1 / 第1章 / Chapter 1 / 1 开始，并包含更多 1.1、1.1.1 或无编号子标题，它仍是详细目录，应优先于前面的简略目录。
- 只有当后续重启后很快出现连续正文段落、代码、图表或讲解文本时，才把它判为正文开始；密集短标题序列重启通常是详细目录，不是正文。
- 如果只有一份简略目录、没有更完整详细目录，简略目录也可以作为 book_toc。
- 后面再次出现的目录含义标题也可能是页眉、正文小节或图表目录，必须结合其后结构判断，不能仅凭出现顺序决定。
- 若同时看到“# 目录”这类独立目录标记和“3.3 目录与文件”这类编号小节，必须先评估独立目录标记；只要独立目录标记后面是跨多章的密集标题序列，就返回该 book_toc。

## 范围原则
- toc_start_line: book_toc 的起始行，通常是目录标记行或第一条目录条目行。
- toc_end_line: book_toc 后第一个明确非目录内容之前的最后一行。不确定时可多包含 10-20 行。
- 附录/Appendix/参考文献/References/索引/Index 如果出现在全书正文目录尾部，属于 book_toc 的一部分。
- 不要只根据“候选目录标记附近原文”决定尾部；必须同时参考“全文 # 标题大纲”。如果大纲中在你准备给出的 toc_end_line 之后仍有 `第7章 ... 331`、`7.12 ... 362`、`习题 100` 这类短标题+页码式条目，它们仍属于目录，应继续纳入。
- 对有页码目录，toc_end_line 应覆盖目录中最后一个带页码或页码式数字结尾的目录标题/子条目，直到后面出现没有页码并开始正文段落的第1章/正文标题重启。
- 如果目录尾部是实验列表，形如 `7.12 实验十二 ... 362`、`7.13 实验十三 ... 364`、`7.16 小结 370`，这些也是目录条目，不能在 `7.11` 或其中间子项处提前结束。
- 如果存在简略目录和详细目录，toc_start_line/toc_end_line 只覆盖详细目录；不要从 brief contents 开始，也不要把 brief contents 条目重复放入详细目录范围。

## 输出
只返回 JSON，不要解释过程。
如果找到全书正文目录:
{"found": true, "toc_type": "book_toc", "toc_marker_line": 行号, "toc_start_line": 行号, "toc_end_line": 行号, "confidence": 0.0到1.0, "reason": "一句话证据"}
如果没有找到全书正文目录，或只找到图目录/表目录/正文小节:
{"found": false, "toc_type": "figure_toc|table_toc|index|body_section|front_matter|unknown", "toc_marker_line": 行号或null, "toc_start_line": null, "toc_end_line": null, "confidence": 0.0到1.0, "reason": "一句话原因"}"""


TOC_DETECT_USER = """文件名: {file_name}

## 全文 # 标题大纲 (行号+文本，共 {total_count} 个)

大纲中标注了 `[页码]` 表示该 `#` 标题文本中含有页码（……数字）。
没有标注的可能是正文标题、广告、前言、书名等。
大纲只包含 `#` 行，目录中很多普通文本行不会出现在这里。

{headings_outline}

## 前部标题详细上下文 (用于精确判定边界)

{detailed_headings}

## 候选目录标记附近原文 (用于判断 book_toc / 假目录)

{toc_marker_contexts}

请先判断是否存在“全书正文目录 book_toc”。不要被“目录”两个字误导：
- `图目录/表目录` 不是 book_toc；
- `查看目录和文件/文件目录管理/工作目录` 是正文小节；
- 只有后面接密集全书篇章结构的目录才是 book_toc。
- 如果存在独立的 `# 目录` / `# 目次` / `# Contents`，且后面紧接 `第1章`、`1.1`、`第2章`、`2.1` 等跨多章密集条目，应优先把这个独立标记判为 book_toc。
- `3.3 目录与文件`、`3.3.5 移动目录或文件` 这类带编号标题只是目录条目或正文小节，不能作为全书目录标记，也不能否定前面已经成立的独立 `# 目录`。
- 给 toc_end_line 时必须参考“全文 # 标题大纲”：如果后面还有带页码/页码式数字结尾的目录条目，例如 `7.12 实验十二 ... 362`、`7.16 小结 370`，不要提前截断；应一直包含到正文从第1章重新开始之前。
- 如果同时存在 `brief contents/contents at a glance/简目` 和更完整的 `contents/table of contents/目录`，选择后面的详细目录，不能把两份目录合并。
- 如果简略目录后重新出现 Part 1 / 第1章 / Chapter 1 / `1 Title`，且后面是更密集、更深层的标题序列而不是正文段落，这表示详细目录开始，不是正文开始。

若找到 book_toc，请给出完整宽松范围；若没有找到，请 found=false。"""

# ---- 第一层 B Prompt: 目录尾部扫描 ----
TOC_TAIL_SYSTEM = """判断全书正文目录 book_toc 的尾部边界。只返回JSON，不解释。"""

TOC_TAIL_USER = """文件名: {file_name}

全书正文目录 book_toc 暂止于行{current_end}。检查后续文本，判断 book_toc 是否继续、是否在本窗口内结束，或本窗口开头已不是 book_toc。

判断要点：
- 目录条目通常是密集短行，包含章/节/小节编号、附录、参考文献、索引、习题等。
- 目录尾部可能出现实验/项目类条目，例如 `7.12 实验十二 ... 362` 后面接 `一、实验目的 362`、`二、实验内容 362`、`三、实验步骤 362`；这些短行带页码或像子条目时仍属于 book_toc，不要在其中间截断。
- 无页码目录也可以成立；不要只因为没有页码就停止。
- 如果窗口内大部分仍是目录条目，并且最后几行仍像目录，返回 continue。
- 如果窗口前半仍是目录、后半出现大段正文/版权正文/前言正文，返回 stop，并给出最后一行目录条目。
- 如果窗口第一批非空行已经是大段正文，不是目录延续，返回 not_toc。
- 如果看到 `第1章/Chapter 1/1.1` 等从头重启，且后面紧跟完整正文段落而不是页码式短目录行，才说明 book_toc 已结束；toc_end_line 应该是重启正文之前最后一条目录行。
- 如果当前范围来自 `brief contents/简略目录`，后续从 Part 1 / 第1章 / Chapter 1 / `1 Title` 重启并继续出现密集 1.1、1.1.1、无编号子标题或页码条目，这是详细目录延续，不是正文开始；应继续扫描。
- 判断重启是“详细目录”还是“正文”的关键不是编号重复，而是后续形态：密集短标题列表属于详细目录，标题后接连续正文段落才属于正文。
- 本窗口就是行{current_end}之后的后续文本。如果本窗口开头的非空行仍然是目录条目，last_toc_line 必须推进到本窗口内的某一行，不能返回旧的行{current_end}。
- 只有当本窗口开头已经完全不是目录延续时，才可以返回 not_toc；不要用 stop+last_toc_line={current_end} 表示“没有继续”。
- 如果窗口出现图目录/表目录/List of Figures/List of Tables，不能把它当作 book_toc 延续，除非它明确只是全书目录的一个普通条目。
- 如果窗口出现正文小节、命令说明、代码块、操作步骤，不是 book_toc 延续。
- “重叠上下文”仅用于理解目录结构，已经包含在当前目录范围内；last_toc_line 必须从“待判断后续文本”中选择。
- 即使待判断文本开头是 `二、实验内容/三、实验步骤` 等重复短栏目，只要重叠上下文或后文显示 `7.1 → 7.2 → 7.3` 等连续编号目录仍在延续，就不能返回 not_toc，也不能停在重复栏目中间。
- 如果无法确认，返回 uncertain。

## 当前目录尾部的重叠上下文
{context_text}

## 待判断后续文本
{chunk_text}

返回JSON（仅JSON，不输出其他内容）：
{{"decision": "continue|stop|not_toc|uncertain", "last_toc_line": 行号或null, "may_continue_after_chunk": true/false, "reason": "一句话"}}"""

# ---- 第二层 Prompt: 目录解析 ----
TOC_PARSE_SYSTEM = """你是书籍目录结构解析器。从目录原文中提取所有标题条目并分配等级。

## 核心原则
1. 目录 = 标题的连续列表。先完整提取 → 再统一分配等级。
2. 有页码时按页码提取，没有页码时按标题密度提取（见下）。

---

## 第一步：标题提取

### 情况 A：目录有页码（最常见）

页码形式：省略号/点线+数字（……545、..36、.42、... 23）

- `#` 开头 + 有页码 → 提取（如 "# 第 1 章 概述……3"）
- 无 `#` + 有页码 → 提取（如 "1.1 背景……5"）
- 最高优先级例外：实验/项目条目内部的固定栏目即使带页码也不要提取，例如 `一、实验目的 352`、`二、实验内容 352`、`三、实验步骤 352`、`四、常见问题 353`、`五、拓展思考 354`。这些只是实验条目的重复栏目，不是全书目录树标题。
- 无页码但属于连续编号/层级结构的标题 → 也必须提取。
  例：
    "# 第 1 章 从这里开始，起飞了"
    "1.1 脚本文件的书写格式"
    "1.2 脚本文件的各种执行方式"
    "1.3 如何在脚本文件中实现数据的输入与输出"
    "1.4 输入与输出的重定向"
    "1.5 各种引号的正确使用姿势 .24"
  即使 1.1-1.4 没有页码，也不能从 1.5 才开始；必须完整提取第1章和 1.1-1.4。
- 只有明显不是目录结构的无页码行才跳过，如 "# 目录"、广告、正文段落、说明文字。

### 情况 B：目录没有页码（纯标题列表）

当整个目录区域都找不到页码时，按**标题密度**判断：

**目录条目**：该行之后紧跟的是另一个标题（中间只有空行，没有正文段落）。
  例：
    行31: "# 第1章 历史和标准"
    行32: ""
    行33: "# 第2章 系统编程概念"  ← 紧接下一个标题 → 这是目录条目 ✓

**前辅文/非目录**：该行之后有大段正文文字（一段或多段）。
  例：
    行400: "## 目标读者"
    行402: "本书主要面向以下读者：为Linux..."  ← 正文段落，不是标题 → 不提取 ✗

**口诀：标题后面是标题 → 提取。标题后面是段落 → 跳过。**

### OCR 断行处理（两种情况都适用）

先恢复“逻辑标题”，再输出标题列表。不要把目录原文的物理换行等同于标题边界。

断行的上半截无页码，下半截有页码。合并为一条：

**例1**（纯文本截断）:
  "20.5 通过 Keepalived 搭建 LVS" + "高可用性集群系统 ……545"
  → "20.5 通过 Keepalived 搭建 LVS 高可用性集群系统"

**例2**（# 标题截断）:
  "# 第 18 章 虚拟化云计算平台" + "Proxmox VE 485"（下一行不是编号小节）
  → "第 18 章 虚拟化云计算平台 Proxmox VE"

**例3**（编号标题后半截被换到下一行）:
  "16.1.7 创建GUI的其他方法"
  "平台无关的窗口API"
  "16.2 GTK+简介"
  → "16.1.7 创建GUI的其他方法 平台无关的窗口API"
  → "16.2 GTK+简介"

判断方法：
- 如果一行带明确编号（第X章、16.1.7、A.3 等），后面紧跟一行或多行无编号短文本，
  再后面出现下一个连续编号标题（如 16.2、16.1.8、18.1），
  不能默认把中间无编号短文本合并到上一条标题，必须判断语义和版式。
- 只有上一标题语义明显未完成，并且无编号短文本能直接补全上一标题时，才作为续行合并。
- 如果中间连续出现多个语义完整、句式相似或主题并列的短标题，它们通常是上一编号标题下面的兄弟子标题，应分别保留，不能全部拼接到上一标题。
- 英文目录尤其常把无编号子标题列在编号标题下面。例如：
  `2.3 Tokenization and LLM capabilities`
  `LLMs are bad at word games`
  `LLMs are challenged by mathematics`
  `LLMs and language equity`
  后三项是独立兄弟子标题，不是 `2.3` 的续行。
- 被合并的无编号续行不要单独输出为目录标题。
- 合并后保留上一条编号标题的等级。
- 只有当无编号短文本明显是独立目录标题时，才单独输出。
- 下一行如果是独立编号小节（"18.1 xxx……50"）→ 不合并到上一行。

### 其他
- 一行中可能挤了多个标题，需拆分
- 输入中 `行123:` 只是原文行号，不能写进输出 text
- 实验/项目章节中常见的重复大纲项，例如 `一、实验目的`、`二、实验内容`、`三、实验步骤`、`四、常见问题`、`五、拓展思考`，只是每个实验条目内部的固定栏目，不是全书目录树中的独立正文标题；即使这些行带页码，也必须从输出 headings 中删除。
- 如果目录尾部出现 `第7章 上机实验` 下的 `7.1 实验一 ...`、`7.2 实验二 ...`、`7.3 实验三 ...`，这些 `7.x` 实验条目应互为兄弟，等级都应为 H3，不能因为中间夹着 `一、实验目的/二、实验内容` 等固定栏目而降到 H5/H6。

错误示例（禁止这样输出）:
  H3 7.1 实验一 安装 Ubuntu
  H4 一、实验目的
  H5 二、实验内容
  H6 7.2 实验二 熟悉 Ubuntu 图形界面

正确示例:
  H2 第7章 上机实验
  H3 7.1 实验一 安装 Ubuntu
  H3 7.2 实验二 熟悉 Ubuntu 图形界面
  H3 7.3 实验三 Ubuntu 基本命令(一)

---

## 第二步：等级分配

- H1: 书名（最多1个）
- H2: 第X章 / Chapter X / Part X / `1 Chapter title` / `2 Chapter title` / 附录 / 参考文献 / 词汇表 / 内容提要 / 前言 / 序
- H3: 编号小节（1.1、2.3等）/ 章末通用标题（小结、习题）/ H2 章节下面的无编号一级子标题
- H4: 更深编号（1.1.1、2.3.4等）
- H5: 更深
- 标题等级由全书结构和编号深度决定，不由中文或英文决定：
  - `第1章 标题`、`Chapter 1 Title`、`1 Title`、`2 Title` 若在全书章序列中，都是 H2。
  - `1.1 Title`、`2.3 Title` 是 H3；`1.1.1 Title` 是 H4。
  - `1. Title`、`2. Title` 若只是某节内部短列表，则不是章标题；结合前后是否形成全书章节序列判断。
- 新章出现时必须回到 H2。不能因为上一章末尾存在 H3/H4 无编号子标题，就把后续 `2 Title`、`3 Title`、`Chapter 2` 降为 H3/H4/H5。
- 对实验/项目章节，`7.1/7.2/7.3` 这种一级小节仍然是 H3；固定栏目 `实验目的/实验内容/实验步骤/常见问题/拓展思考` 不输出为标题。

---

## 输出自检
- 有页码：优先检查页码；但局部页码 OCR 丢失时，连续编号标题不能删除。
  如果解析结果从 1.5、2.3 这类中途编号开始，而前文有第1章/1.1/1.2，则说明漏提取，必须补回。
- 没有页码：每条标题下紧跟的是另一个标题（而非正文段落）？跟着段落的删除
- 如果无编号短标题夹在两个相关编号标题之间，并导致后一个编号标题等级异常，
  必须重新判断它是否是上一条编号标题的续行；若是续行，合并后删除该无编号目录节点。
- 检查所有章标题是否同级：`1 Title`、`2 Title`、`3 Title` 或第1章、第2章、第3章必须保持 H2；不能随着目录推进逐渐下沉到 H3/H4/H5。
- 检查无编号短标题是否被错误拼接：连续多个语义完整的并列短标题应分别输出，不能全部合并成一个超长标题。
- 如果出现 `7.1 实验一` 后接固定栏目，再出现 `7.2 实验二`，最终 headings 中应保留 `7.1` 和 `7.2` 为同级 H3；不要保留那些固定栏目导致层级错乱。
- 最终 JSON 中不得出现这些固定栏目标题：`实验目的`、`实验内容`、`实验步骤`、`常见问题`、`拓展思考`。如果已经解析出来，返回前必须删除。

## 输出
{"headings": [{"text": "条目(去页码)", "level": 1-5}, ...]}
按出现顺序完整列出。只返回一个合法 JSON 对象，第一个字符 `{`，最后一个字符 `}`。"""


TOC_PARSE_USER = """文件名: {file_name}

{page_hint}

## 目录区域原文 (行 {toc_start}-{toc_end})

{toc_content}

请解析上述目录，提取所有正文标题及其等级。

通用结构提醒:
- 先理解整本书的父子关系，再分配等级；同一章序列中的 `1 Title/2 Title/3 Title`、`Chapter 1/Chapter 2`、`第1章/第2章` 都必须保持 H2，不能逐章下沉。
- `1.1/2.3` 通常是 H3，`1.1.1/2.3.4` 通常是 H4；编号深度相同的标题通常同级。
- 编号标题下面连续出现多个语义完整、句式并列的无编号短标题时，它们通常是独立子标题，不是上一标题的 OCR 续行。
- 只有上一标题语义明显未完成、下一行能直接补全它时才合并；不要把多个完整标题拼成一个超长标题。
- 一条物理行里若挤入多个逻辑标题，应拆成多条 headings。

特别注意实验/项目章节:
- `7.1 实验一`、`7.2 实验二`、`7.3 实验三` 这类编号实验条目是正式目录标题，通常同级。
- `一、实验目的`、`二、实验内容`、`三、实验步骤`、`四、常见问题`、`五、拓展思考` 是每个实验内部重复栏目，即使带页码也不要输出到 headings。
- 父标题的作用范围到下一个同级或更高级标题结束；`7.2` 出现时，`7.1` 已结束，`7.2` 必须与 `7.1` 同级，不能挂在 `五、拓展思考` 下面。"""

TOC_PARSE_CHUNK_USER = """文件名: {file_name}

{page_hint}

## 当前目录分块 {chunk_index}/{chunk_count} (原文行 {chunk_start}-{chunk_end})

{toc_content}

请只解析这个分块中的目录条目，提取标题及其等级。
不要输出其他分块的内容；不要把 `行123:` 这种原文行号写进 text。
即使当前分块没有显示前一分块的父标题，也要按编号本身恢复稳定层级：全书章序列中的 `1 Title/2 Title`、`Chapter 1/Chapter 2`、`第1章/第2章` 为 H2，`1.1/2.3` 为 H3，`1.1.1/2.3.4` 为 H4；新章必须回到 H2，不能随着分块或目录推进逐渐下沉。
编号标题后连续出现多个语义完整、句式并列的无编号短标题时，应分别保留为独立子标题；只有上一标题语义明显未完成时，才把紧邻短行作为 OCR 续行合并。
一条物理行里若挤入多个逻辑标题，应拆成多条 headings。
如果本分块包含实验/项目目录，`7.1/7.2/7.3` 等编号实验条目是同级正式目录标题；`一、实验目的/二、实验内容/三、实验步骤/四、常见问题/五、拓展思考` 是重复固定栏目，即使带页码也不要输出。
父标题的作用范围到下一个同级或更高级标题结束；后续 `7.2` 不能作为 `五、拓展思考` 的子标题。
只返回 JSON，不要输出分析过程。"""


# ---- JSON 修复 Prompt ----
REPAIR_PROMPT = """上次输出非法。请只返回合法 JSON。必须格式: {expected_format}"""


# ============================================================
# 工具函数
# ============================================================


def count_tokens(text: str) -> int:
    """粗略估算 prompt token 数。

    作用:
        用于控制第一层 prompt 的上下文预算，不依赖 tokenizer，速度快但只是近似值。

    输入:
        text: str，任意待估算文本。

    输出:
        int，估算 token 数。中文按约 1.2 token/字，英文按约 4 字符/token。

    例子:
        count_tokens("第1章 嵌入式系统") -> 约 10
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return int(cjk * 1.2 + (len(text) - cjk) / 4) + 1


def _normalize_chat_completions_url(url: str) -> str:
    """把 API 根地址规范化为 chat/completions 端点。"""
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return cleaned + "/chat/completions"
    return cleaned + "/v1/chat/completions"


def configure_llm(base_url: str | None = None,
                  model: str | None = None,
                  api_key: str | None = None,
                  api_key_env: str | None = None,
                  use_vllm_extras: bool | None = None) -> None:
    """根据命令行参数配置本次运行使用的大模型。"""
    global BASE_URL, MODEL, API_KEY_OVERRIDE, API_KEY_ENV, LLM_USE_VLLM_EXTRAS

    custom_url = bool(base_url)
    if base_url:
        BASE_URL = _normalize_chat_completions_url(base_url)
    if model:
        MODEL = model
    if api_key_env:
        API_KEY_ENV = api_key_env
    if api_key:
        API_KEY_OVERRIDE = api_key

    if use_vllm_extras is not None:
        LLM_USE_VLLM_EXTRAS = use_vllm_extras
    elif custom_url:
        LLM_USE_VLLM_EXTRAS = False


def parse_json(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON 对象。

    作用:
        LLM 有时会返回 ```json 代码块、前后解释文字、或被 max_tokens 截断的 JSON。
        本函数按多种策略尽量恢复出 dict。

    输入:
        text: str，LLM 原始响应文本。

    输出:
        dict，例如:
        {"headings": [{"text": "第1章 概述", "level": 2}]}

    异常:
        ValueError: 所有策略都无法解析 JSON 时抛出。

    例子:
        parse_json('```json\\n{"a": 1}\\n```') -> {"a": 1}
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 策略1: 直接解析
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # 策略2: 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try: return json.loads(m.group(1).strip())
        except json.JSONDecodeError: pass

    # 策略3: 找 { ... } 区域
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[start:i + 1])
                    except json.JSONDecodeError: break
        end = text.rfind("}")
        if end > start:
            try: return json.loads(text[start:end + 1])
            except json.JSONDecodeError: pass

    # 策略5: 截断修复 — JSON被max_tokens截断，尝试补全
    # 从后往前找最后一个完整的 key-value 对，截断到那里并加上 }]}
    if start >= 0:
        # 找最后一个完整的 "key": value 或 "key": { 或 "key": [
        last_complete = text.rfind('"}')
        if last_complete < 0: last_complete = text.rfind('"]')
        if last_complete > 0:
            # 从该位置往后找下一个逗号或换行，在那里截断
            repair_point = text.find(',', last_complete)
            if repair_point < 0:
                repair_point = text.find('\n', last_complete)
            if repair_point > start:
                truncated = text[start:repair_point]
                # 补上结尾
                truncated += '\n  ]\n}'
                try: return json.loads(truncated)
                except json.JSONDecodeError: pass

    raise ValueError(f"无法解析 JSON: {text[:500]}...")


def clean_title(text: str) -> str:
    """清理标题文本，生成更适合比较的标题正文。

    作用:
        去掉目录条目尾部的点线、省略号、页码，并压缩空白。

    输入:
        text: str，原始标题或目录条目。

    输出:
        str，清理后的标题。

    例子:
        clean_title("1.1 概述…… 3") -> "1.1 概述"
        clean_title("小结 29") -> "小结"
    """
    t = text.strip()
    t = TOC_PAGE_RE.sub("", t)
    t = re.sub(r"[.…·]{2,}\s*\d*\s*$", "", t)
    t = re.sub(r"\s+\d{1,4}\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def compact_text(text: str, max_chars: int | None = None) -> str:
    """压缩 prompt 上下文文本。

    作用:
        删除多余空格、换行、制表符，必要时截断长度，降低 prompt token 消耗。

    输入:
        text: str，原始上下文文本。
        max_chars: int | None，最大字符数；None 表示不截断。

    输出:
        str，单行紧凑文本。

    例子:
        compact_text("A\\n\\n  B", 10) -> "A B"
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def title_key(text: str) -> str:
    """把标题标准化为匹配用 key。

    作用:
        第四层映射时用 key 做精确匹配，避免大小写和目录页码影响。

    输入:
        text: str，标题文本。

    输出:
        str，小写、去页码、压缩空白后的 key。

    例子:
        title_key("Chapter 1 Introduction 12") -> "chapter 1 introduction"
    """
    t = clean_title(text)
    # Markdown 转义会把正文里的 OS_DBG.C 变成 OS\_DBG.C，匹配目录时应视为同一标题。
    t = re.sub(r"\\(.)", r"\1", t)
    # 常见中英文标点归一，避免 "选项：iptables" 和 "选项: iptables" 匹配失败。
    t = t.translate(str.maketrans({
        "：": ":",
        "﹕": ":",
        "．": ".",
        "。": ".",
        "（": "(",
        "）": ")",
    }))
    # OCR 有时会把 OS_CFG_APP.C 识别成 OS_CFG_APP. C，这里只压紧英文/数字 token 内部的点号。
    t = re.sub(r"(?<=[A-Za-z0-9_])\s*\.\s*(?=[A-Za-z0-9_])", ".", t)
    # 统一 "第 9 章" 和 "第9章"，方便正文拆分章标题时匹配目录。
    t = re.sub(r"第\s*([一二三四五六七八九十百零0-9]+)\s*([章节篇卷部])", r"第\1\2", t)
    # 去所有空格 — OCR 空格不一致是匹配失败的主要原因
    t = re.sub(r"\s+", "", t)
    return t.lower()


def symbol_stripped_title_key(text: str) -> str:
    """标题宽匹配 key: 去掉 Markdown 标题符号和常见分隔符。"""
    t = title_key(_strip_atx_marker(text))
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", t).lower()


def display_title(text: str) -> str:
    """输出用标题文本，避免把目录里的 ATX 标记带进正文标题。"""
    t = _strip_atx_marker(text)
    return re.sub(r"\s+#+\s*$", "", t).strip()


def _is_toc_backmatter_key(key: str) -> bool:
    """判断目录标题是否已经进入附录/参考文献/索引等书尾结构。"""
    key = key or ""
    return bool(
        key.startswith(("附录", "appendix"))
        or key in {"参考书目", "参考文献", "bibliography", "references", "索引", "index"}
    )


def _toc_structural_marker(key: str) -> tuple[str, int] | None:
    """提取目录顶层结构序列，用于通用判断“正常推进”还是“重启倒退”。

    返回:
        ("part", 2): 第2篇 / 第二部分 / Part II
        ("chapter", 10): 第10章 / Chapter 10

    设计原则:
        同类结构 1 -> 2 -> 3 是目录继续；3 -> 1 才是重启信号。
        如果 part 正常前进，则允许其下 chapter 从 1 重新开始。
    """
    key = key or ""
    m = re.match(r"^(?:第)?([一二两三四五六七八九十百千零〇0-9]+)[篇卷部]", key)
    if m:
        n = _chinese_num_to_int(m.group(1))
        if n is not None:
            return "part", n
    m = re.match(r"^part\s*([ivx]+|\d+)", key, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).lower()
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
                 "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
        n = int(raw) if raw.isdigit() else roman.get(raw)
        if n is not None:
            return "part", n

    m = re.match(r"^第([一二两三四五六七八九十百千零〇0-9]+)[章节]", key)
    if m:
        n = _chinese_num_to_int(m.group(1))
        if n is not None:
            return "chapter", n
    m = re.match(r"^chapter\s*0*(\d+)", key, flags=re.IGNORECASE)
    if m:
        return "chapter", int(m.group(1))
    return None


def _drop_toc_tail_contamination(toc_headings: list[dict]) -> list[dict]:
    """截掉目录尾部混入的广告目录/另一套书目录。

    通用场景:
        真目录已经到“附录/参考书目”，后面又出现“Part 1 / 第1章 / 第2章 ...”。
        这通常不是本书目录，而是 PDF 尾部广告、样章或其他书的目录。
    """
    seen_backmatter = False
    cut_index = None

    for idx, h in enumerate(toc_headings):
        text = display_title(h.get("text", ""))
        key = title_key(text)
        level = int(h.get("level") or 2)
        if level > 2:
            continue

        if _is_toc_backmatter_key(key):
            seen_backmatter = True
            continue

        marker = _toc_structural_marker(key)
        if seen_backmatter and marker is not None and marker[1] <= 2:
            cut_index = idx
            break

    if cut_index is None:
        return toc_headings

    removed = toc_headings[cut_index:]
    first = display_title(removed[0].get("text", "")) if removed else ""
    print(
        f"  目录清理: 检测到尾部疑似广告/新书目录，"
        f"从第 {cut_index + 1} 个标题开始截断 {len(removed)} 个 ({first})"
    )
    return toc_headings[:cut_index]


def _is_frontmatter_toc_title(text: str) -> bool:
    """识别目录开头常见的前辅文条目。"""
    key = title_key(text)
    if _toc_structural_marker(key) is not None:
        return False
    return bool(
        "前言" in key
        or "序言" in key
        or key.endswith("序")
        or key in {"序", "自序", "代序", "引言", "导言", "致谢", "preface", "foreword"}
    )


def _is_main_body_toc_start(text: str) -> bool:
    """判断目录条目是否已经进入正文主结构。"""
    key = title_key(text)
    return bool(
        _toc_structural_marker(key) is not None
        or key in {"introduction", "intro", "引言", "导言"}
        or key.startswith(("chapter1", "chapter01", "第1章", "第一章"))
    )


def _format_toc_role_items(toc_headings: list[dict], max_items: int = 80) -> str:
    """把第二层目录标题格式化为角色分类输入。"""
    lines = []
    for idx, h in enumerate(toc_headings[:max_items]):
        text = display_title(h.get("text", ""))
        level = int(h.get("level") or 2)
        lines.append(f'{idx}. H{level} "{compact_text(text, 160)}"')
    return "\n".join(lines)


def _filter_layer2_toc_roles_by_llm(toc_headings: list[dict],
                                    file_name: str,
                                    max_items: int = 80) -> list[dict]:
    """第二层后处理: 让 LLM 判断目录条目角色，移除非目录树节点。

    规则只做安全保护:
        - LLM 不确定时默认保留。
        - 只允许移除 toc_container / publication_credit / frontmatter_note / noise。
        - 不允许移除第一个正文主结构标题，也不允许移除章/篇等强结构标题。
    """
    if not toc_headings:
        return toc_headings

    sample_count = min(max_items, len(toc_headings))
    expected = (
        '{"items": ['
        '{"index": 0, "role": "toc_container|book_heading|publication_credit|frontmatter_note|noise", '
        '"keep": true/false, "confidence": 0.0, "reason": "一句话"}'
        ']}'
    )
    prompt = f"""你是书籍目录条目角色分类器。第二层已经从全书正文目录区域解析出标题列表，但其中可能混入了目录页标题、出版署名、说明性文字或 OCR 噪声。

文件名: {file_name}

你的任务：判断每个条目是否应该进入最终“目录树”。

角色定义:
- book_heading: 应进入目录树的正式内容标题，例如 Introduction、Preface、Chapter、Section、Part、Appendix、References、Index 等。
- toc_container: 目录页自身的标题，不是书的内容标题，例如“目录”“Contents”“Table of Contents”“Contents at a Glance”等。
- publication_credit: 作者、审稿人、版权、出版人员、致谢等出版署名类条目。它们如果只是目录页前置出版信息，不进入正文目录树。
- frontmatter_note: 本书结构、读者对象、各章内容、内容简介等说明性条目，不是正文目录树节点。
- noise: OCR 噪声、页眉页脚、广告、乱码。

判断原则:
1. keep=true 只用于真正应进入最终目录树的 book_heading。
2. 目录容器标题必须 keep=false。
3. 出现在正文主结构之前的出版署名/说明性条目，如果不是正式正文标题，keep=false。
4. Introduction / Preface 如果是书中正式内容标题，应 keep=true。
5. Chapter 1 / 第1章 / Part I / Appendix 等强结构标题应 keep=true。
6. 不确定时 keep=true，避免误删真实目录标题。
7. 只返回 JSON，不要解释过程。

待分类目录条目（只分类这些 index）:
{_format_toc_role_items(toc_headings, sample_count)}

返回 JSON:
{expected}"""
    messages = [
        {"role": "system", "content": "你是目录条目角色分类器。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    result = call_llm_json_with_repair(
        messages,
        max_tokens=max(1200, min(6000, sample_count * 100)),
        stage="layer2_toc_role_filter",
        expected_format=expected,
        timeout=TIMEOUT,
    )

    raw_items = result.get("items", [])
    if not isinstance(raw_items, list):
        raise SkipMarkdownFile(
            "layer2_toc_role_filter_failed",
            "layer2_toc_role_filter",
            f"第二层角色分类未返回 items: {result}",
        )

    first_main_index = None
    for idx, h in enumerate(toc_headings[:sample_count]):
        if _is_main_body_toc_start(h.get("text", "")):
            first_main_index = idx
            break

    removable_roles = {"toc_container", "publication_credit", "frontmatter_note", "noise"}
    remove_indices: set[int] = set()
    role_by_index: dict[int, dict] = {}
    for item in raw_items:
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        if not (0 <= idx < sample_count):
            continue
        role = str(item.get("role") or "").strip().lower()
        keep = item.get("keep")
        role_by_index[idx] = item
        if keep is not False or role not in removable_roles:
            continue

        text = display_title(toc_headings[idx].get("text", ""))
        key = title_key(text)
        if idx == first_main_index:
            continue
        if first_main_index is not None and idx > first_main_index:
            # 进入 Introduction/第1章/第一部分 后，普通章节小标题不能再被当作
            # frontmatter/publication 类噪声删除。不确定时宁可保留。
            continue
        if first_main_index is None and idx > 8 and role != "noise":
            continue
        if _toc_structural_marker(key) is not None:
            continue
        if _is_main_body_toc_start(text):
            continue
        remove_indices.add(idx)

    if not remove_indices:
        return toc_headings

    if len(remove_indices) >= max(10, sample_count // 2):
        raise SkipMarkdownFile(
            "layer2_toc_role_filter_suspicious",
            "layer2_toc_role_filter",
            f"第二层角色分类要求删除过多标题: {len(remove_indices)}/{sample_count}",
        )

    kept = []
    removed = []
    for idx, h in enumerate(toc_headings):
        if idx in remove_indices:
            item = role_by_index.get(idx, {})
            removed.append(
                f"{display_title(h.get('text', ''))}({item.get('role', 'unknown')})"
            )
            continue
        kept.append(h)

    names = "、".join(removed[:8])
    if len(removed) > 8:
        names += "..."
    print(f"  第二层角色清理: LLM 移除 {len(removed)} 个非目录树标题 ({names})")
    return kept


def _heading_key_exists_between(md_text: str, target_key: str,
                                start_line: int, end_line: int) -> bool:
    """判断指定行号范围内是否存在与目标相近的 ATX 标题。"""
    if not target_key:
        return False
    target_symbol_key = symbol_stripped_title_key(target_key)
    lines = md_text.splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)
    pat = re.compile(r"^#{1,6}\s+(.+)")
    for line_no in range(start, end + 1):
        m = pat.match(lines[line_no - 1].strip())
        if not m:
            continue
        key = title_key(m.group(1))
        if key and (target_key in key or key in target_key):
            return True
        symbol_key = symbol_stripped_title_key(m.group(1))
        if (
            target_symbol_key
            and symbol_key
            and (target_symbol_key in symbol_key or symbol_key in target_symbol_key)
        ):
            return True
    return False


def _drop_pretoc_frontmatter_headings(toc_headings: list[dict],
                                      md_text: str,
                                      toc_range: dict) -> list[dict]:
    """移除正文已出现在目录前的前辅文目录条目。

    有些 PDF 的版式是“前言正文 → 目录 → 正文第一章”。目录里仍会列出“前言”，
    但当前转换流程的正文候选从目录后开始，无法再挂载目录前的前言正文。
    这类条目如果保留，会导致第五层后机制尝试在整本正文中硬补，范围巨大且不可靠。
    """
    if not md_text or not toc_range.get("toc_start_line"):
        return toc_headings

    toc_start = int(toc_range.get("toc_start_line") or 0)
    toc_end = int(toc_range.get("toc_end_line") or 0)
    lines = md_text.splitlines()
    total_lines = len(lines)
    after_frontmatter_end = total_lines
    pat = re.compile(r"^#{1,6}\s+(.+)")
    for line_no in range(max(1, toc_end + 1), total_lines + 1):
        m = pat.match(lines[line_no - 1].strip())
        if not m:
            continue
        if _is_main_body_toc_start(display_title(m.group(1))):
            after_frontmatter_end = line_no - 1
            break

    kept = []
    removed = []
    scanning_prefix = True

    for heading in toc_headings:
        text = display_title(heading.get("text", ""))
        key = title_key(text)

        if scanning_prefix and _is_frontmatter_toc_title(text):
            before_exists = _heading_key_exists_between(
                md_text,
                key,
                1,
                max(1, toc_start - 1),
            )
            after_exists = _heading_key_exists_between(
                md_text,
                key,
                toc_end + 1,
                after_frontmatter_end,
            )
            if before_exists and not after_exists:
                removed.append(text)
                continue
        else:
            if _toc_structural_marker(key) is not None:
                scanning_prefix = False
            elif not _is_frontmatter_toc_title(text):
                scanning_prefix = False

        kept.append(heading)

    if removed:
        names = "、".join(removed[:6])
        if len(removed) > 6:
            names += "..."
        print(f"  目录清理: 移除 {len(removed)} 个目录前已出现的前辅文标题 ({names})")
    return kept


# ============================================================
# LLM 调用
# ============================================================


def call_llm(messages: list[dict], max_tokens: int = OUTPUT_TOKENS,
             retries: int = 3, timeout: int = TIMEOUT) -> str:
    """调用 OpenAI 兼容的 Chat Completions API。

    作用:
        给本地模型或兼容接口发送 messages，返回 assistant 文本。
        默认关闭 thinking，并禁用代理，适合局域网 vLLM/兼容服务。

    输入:
        messages: list[dict]，OpenAI 格式消息，例如:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        max_tokens: int，模型最大输出 token。
        retries: int，失败后重试次数。
        timeout: int，单次 HTTP 超时时间，单位秒。

    输出:
        str，模型返回的 message.content。

    异常:
        RuntimeError: 重试后仍失败。

    例子:
        call_llm([{"role": "user", "content": "返回 {\\\"ok\\\": true}"}])
    """
    api_key = API_KEY_OVERRIDE or os.getenv(API_KEY_ENV) or "EMPTY"
    payload = {"model": MODEL, "messages": messages, "temperature": 0.0,
               "top_p": 0.1, "max_tokens": max_tokens}
    if LLM_USE_VLLM_EXTRAS:
        payload.update(
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": False},
        )

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                BASE_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [LLM错误] attempt={attempt+1}/{retries+1}: {e}")
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"LLM 调用失败")


def call_llm_json_with_repair(messages: list[dict], max_tokens: int,
                              stage: str, expected_format: str,
                              repair_attempts: int = 3,
                              timeout: int = TIMEOUT) -> dict:
    """调用 LLM 并要求 JSON；非 JSON 时最多修复 3 次，仍失败则跳过文件。"""
    current_messages = messages
    last_content = ""
    last_error = ""
    total_attempts = repair_attempts + 1

    original_prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            original_prompt = str(m.get("content", ""))
            break
    if len(original_prompt) > 6000:
        original_prompt = original_prompt[-6000:]

    for attempt in range(total_attempts):
        last_content = call_llm(
            current_messages,
            max_tokens=max(max_tokens, 400),
            timeout=timeout,
        )
        try:
            return parse_json(last_content)
        except Exception as e:
            last_error = str(e)
            if attempt >= repair_attempts:
                break
            print(
                f"    [{stage}] JSON解析失败 {attempt + 1}/{total_attempts}，"
                "要求 LLM 只修复为 JSON"
            )
            repair_prompt = f"""你刚才没有返回合法 JSON。
不要解释，不要推理，不要 Markdown，不要代码块。
只能返回一个 JSON 对象，格式如下：
{expected_format}

上一轮回复：
{last_content[:4000]}

原始待判断内容：
{original_prompt}"""
            current_messages = [
                {"role": "system", "content": "你是 JSON 修复器。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
                {"role": "user", "content": repair_prompt},
            ]

    raise SkipMarkdownFile(
        "llm_json_format_failed",
        stage,
        f"{stage} 连续 {total_attempts} 次未返回合法 JSON: {last_error}",
    )


# ============================================================
# 预处理: 提取 # 候选
# ============================================================


def extract_headings(md_text: str, context_chars: int = 200) -> list[dict]:
    """提取所有 ATX `#` 候选标题及其上下文。

    原理:
        转换后的 Markdown 里，代码块经常包含 `#define`、`#include`、`#endif`。
        这些不是文档标题，所以在提取候选时直接跳过 fenced code block。

    Args:
        md_text: 原始 Markdown 文本。
        context_chars: 每个候选标题前后上下文最多保留的字符数。

    Returns:
        列表元素包含 line/raw/text/prev_text/next_text/prev_headings/next_headings。
    """
    lines = md_text.split("\n")
    pat = re.compile(r"^(#{1,6})\s+(.+)")
    _clean_tail = re.compile(r"\s+#+\s*$")  # 去掉 ATX 闭合标记如 "## Introduction ##"
    all_h = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if m := pat.match(stripped):
            title_text = _clean_tail.sub("", m.group(2)).strip()
            all_h.append((i + 1, stripped, title_text))

    results = []
    for idx, (line_num, raw, text) in enumerate(all_h):
        pos = line_num - 1
        prev_text = "\n".join(lines[max(0, pos - 20):pos])
        if len(prev_text) > context_chars: prev_text = prev_text[-context_chars:]
        next_text = "\n".join(lines[pos + 1 : min(len(lines), pos + 21)])
        if len(next_text) > context_chars: next_text = next_text[:context_chars]
        prev_h = [f"行{all_h[j][0]}: {all_h[j][2]}" for j in range(idx - 1, max(idx - 3, -1), -1)][::-1]
        next_h = [f"行{all_h[j][0]}: {all_h[j][2]}" for j in range(idx + 1, min(idx + 3, len(all_h)))]

        results.append({
            "line": line_num, "raw": raw, "text": text,
            "prev_text": prev_text.strip(), "next_text": next_text.strip(),
            "prev_headings": prev_h, "next_headings": next_h,
        })
    return results


# ============================================================
# 第一层: 定位目录区域
# ============================================================


def _format_toc_marker_contexts(md_text: str, headings: list[dict],
                                max_markers: int = 6,
                                window_lines: int = 220,
                                max_chars: int = 14000) -> str:
    """为第一层 prompt 提供目录标记附近的真实原文窗口。"""
    if not md_text:
        return "（未提供目录标记附近原文）"

    lines = md_text.splitlines()
    marker_lines = []
    marker_types = {}
    for h in headings:
        key = title_key(h.get("text", ""))
        if _is_toc_marker_key(key):
            line_no = int(h.get("line") or 0)
            if line_no > 0:
                marker_lines.append(line_no)
                marker_types[line_no] = _toc_marker_candidate_type(key)

    if not marker_lines:
        return "（未在 # 标题大纲中发现明确的目录标记）"

    parts = []
    for marker_line in sorted(set(marker_lines))[:max_markers]:
        start = max(1, marker_line - 3)
        end = min(len(lines), marker_line + window_lines)
        snippet = [
            f"候选目录标记: 行{marker_line}",
            f"程序粗分类提示: {marker_types.get(marker_line, 'unknown')}（仅供参考，最终请结合上下文判断）",
        ]
        for line_no in range(start, end + 1):
            text = compact_text(lines[line_no - 1], 220)
            snippet.append(f"行{line_no}: {text}")
        parts.append("\n".join(snippet))

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...（目录标记附近原文已截断）"
    return text


def layer1_detect_toc(headings: list[dict], file_name: str, md_text: str = "") -> dict:
    """第一层 A: LLM 定位目录的起止行号。

    策略:
    1. 构建**全文标题大纲** (所有标题的行号+文本) — 紧凑格式，让 LLM 有全局视野
    2. 构建**前部标题详细上下文** — 只对前半部分的标题附带正文上下文
       (因为目录几乎总在文档前部，后半部分不需要详细上下文)
    3. 两者结合发给 LLM，让 LLM 精确判定目录起止行号
    4. toc_end_line 约束: 宁可偏大不可偏小

    注意:
        这个函数只给出 LLM 的候选判断。最终是否接受该范围，还要在 process_file()
        中经过重复标题证据和严格目录标记校验。

    输入:
        headings: list[dict]，extract_headings 输出。例如:
            [
              {"line": 67, "raw": "# 目录", "text": "目录", "prev_text": "...", "next_text": "..."},
              {"line": 73, "raw": "# 第1章 概述", "text": "第1章 概述", ...}
            ]
        file_name: str，当前文件名，用于 prompt。
        md_text: str，完整原文，只用于把目录标记附近的真实文本片段提供给 LLM。

    Returns:
        dict，格式:
        {
          "toc_marker_line": int | None,
          "toc_start_line": int | None,
          "toc_end_line": int | None,
          "found": bool
        }

    例子:
        layer1_detect_toc(headings, "嵌入式系统.md")
        -> {"toc_marker_line": 67, "toc_start_line": 67, "toc_end_line": 480, "found": True}
    """
    # ---- 全文大纲 (所有标题，紧凑格式，标注页码) ----
    outline_lines = []
    for h in headings:
        tag = " [页码]" if _has_toc_page_marker(h.get("text", "")) else ""
        outline_lines.append(f"行{h['line']}: {compact_text(h['text'])}{tag}")
    outline_text = "\n".join(outline_lines)
    toc_marker_contexts = _format_toc_marker_contexts(md_text, headings)

    # ---- 前部标题详细上下文 (目标: 整个目录检测 prompt 控制在约 30000 tokens 内) ----
    toc_prompt_budget = 30000
    empty_user_prompt = TOC_DETECT_USER.format(
        file_name=file_name,
        total_count=len(headings),
        headings_outline="",
        detailed_headings="",
        toc_marker_contexts=toc_marker_contexts,
    )
    fixed_tokens = (
        count_tokens(TOC_DETECT_SYSTEM)
        + count_tokens(empty_user_prompt)
        + count_tokens(outline_text)
    )
    budget_for_detail = max(0, toc_prompt_budget - fixed_tokens)

    # 至少取前 30 个，最多取前 200 个；第 30 个以后按实际 token 预算截断。
    detail_parts = []
    detail_tokens = 0
    min_detail_count = min(30, len(headings))
    for index, h in enumerate(headings[:200], start=1):
        part = (
            f"行{h['line']}: {compact_text(h['text'])}\n"
            f"  前文: {compact_text(h['prev_text'], 120)}\n"
            f"  后文: {compact_text(h['next_text'], 120)}"
        )
        part_tokens = count_tokens(part) + 2
        if index > min_detail_count and detail_tokens + part_tokens > budget_for_detail:
            break
        detail_parts.append(part)
        detail_tokens += part_tokens

    user_prompt = TOC_DETECT_USER.format(
        file_name=file_name,
        total_count=len(headings),
        headings_outline=outline_text,
        detailed_headings="\n\n".join(detail_parts),
        toc_marker_contexts=toc_marker_contexts,
    )

    total_tokens = (count_tokens(TOC_DETECT_SYSTEM) + count_tokens(user_prompt))
    print(f"  第一层 prompt: {total_tokens} tokens (大纲{len(outline_text)}字符 + {len(detail_parts)}个详细标题)")

    messages = [
        {"role": "system", "content": TOC_DETECT_SYSTEM},
        {"role": "user", "content": "/no_think\n" + user_prompt},
    ]

    result = call_llm_json_with_repair(
        messages,
        max_tokens=OUTPUT_TOKENS,
        stage="layer1_toc_detect",
        expected_format='{"found": true/false, "toc_type": "book_toc|figure_toc|table_toc|index|body_section|front_matter|unknown", "toc_marker_line": 行号或null, "toc_start_line": 行号或null, "toc_end_line": 行号或null, "confidence": 0.0到1.0, "reason": "一句话"}',
    )
    found = bool(result.get("found"))
    toc_type = str(result.get("toc_type") or "unknown").strip().lower()
    toc_line = result.get("toc_marker_line")
    toc_start = result.get("toc_start_line")
    toc_end = result.get("toc_end_line")
    confidence = result.get("confidence")

    if not found or toc_type != "book_toc":
        reason = str(result.get("reason", ""))[:120]
        print(f"  LLM 判定: 未找到全书正文目录 (type={toc_type}, confidence={confidence}, {reason})")
        return {"toc_marker_line": None, "toc_start_line": None,
                "toc_end_line": None, "found": False,
                "toc_type": toc_type, "confidence": confidence,
                "reason": result.get("reason", "")}

    if not all(isinstance(x, int) for x in (toc_line, toc_start, toc_end)):
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_detect",
            f"第一层目录行号字段不完整或不是整数: {result}",
        )

    span = toc_end - toc_start
    total_lines = len(md_text.splitlines()) if md_text else 0
    # 目录行数不能按 # 候选数量估算：很多目录小节没有 Markdown 标题符号。
    if span <= 5:
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_detect",
            f"第一层目录范围不合理: {toc_start}-{toc_end} (跨{span}行)",
        )

    print(
        f"  LLM 判定: book_toc 目录标记行{toc_line}, "
        f"范围行{toc_start}-{toc_end} (跨{span}行, confidence={confidence})"
    )
    result = {"toc_marker_line": toc_line, "toc_start_line": toc_start,
              "toc_end_line": toc_end, "found": True,
              "toc_type": "book_toc", "confidence": confidence,
              "reason": result.get("reason", "")}
    if total_lines and span >= total_lines * 0.85:
        result["layer1_overwide"] = True
        print("  第一层范围过宽，交给第二层范围确认复核")
    return result


def _strip_atx_marker(text: str) -> str:
    """去掉 Markdown ATX 标题前缀，保留标题文本。"""
    return re.sub(r"^#{1,6}\s+", "", (text or "").strip()).strip()


def _toc_marker_candidate_type(key: str) -> str:
    """对目录候选做粗分类；只用于提示和安全校验，最终分类仍交给 LLM。"""
    key = key or ""
    if not key:
        return "unknown"

    if any(word in key for word in (
        "图目录", "插图目录", "图目", "listoffigures", "figurescontents"
    )):
        return "figure_toc"

    if any(word in key for word in (
        "表目录", "表格目录", "表目", "listoftables", "tablescontents"
    )):
        return "table_toc"

    if any(word in key for word in (
        "查看目录", "目录和文件", "文件目录", "目录管理", "工作目录",
        "当前目录", "改变目录", "列出目录", "目录结构",
    )):
        return "body_section"

    if any(word in key for word in (
        "本书内容体系", "本书结构", "各章内容", "章节简介", "读者对象",
        "内容简介",
    )):
        return "front_matter"

    book_toc_words = (
        "目录",
        "总目录",
        "章节目录",
        "内容目录",
        "目录表",
        "目次",
        "目錄",
        "目録",
        "篇目",
        "章目",
        "contents",
        "tableofcontents",
    )
    if any(word in key for word in book_toc_words) or key in {"toc", "录", "錄"}:
        return "book_toc"

    return "unknown"


def _is_toc_marker_key(key: str) -> bool:
    """判断规范化后的标题是否值得作为目录候选交给 LLM。"""
    return _toc_marker_candidate_type(key) != "unknown"


def _is_book_toc_marker_key(key: str) -> bool:
    """判断标题是否像明确的全书正文目录标记。"""
    return _toc_marker_candidate_type(key) == "book_toc"


def _is_negative_toc_marker_key(key: str) -> bool:
    """判断标题是否明确像假目录标记，如图目录/表目录/正文目录操作小节。"""
    return _toc_marker_candidate_type(key) in {
        "figure_toc", "table_toc", "body_section", "front_matter"
    }


def _looks_like_toc_continuation_line(text: str) -> bool:
    """判断无页码目录中的短续行，如被 OCR 拆开的标题后半截。"""
    s = _strip_atx_marker(text)
    if not s:
        return False
    if len(s) > 80:
        return False
    if re.search(r"[。！？；;]$", s):
        return False
    if re.search(r"^(图|表)\s*\d", s):
        return False
    if re.search(r"精品学习资料|下载汇总|视频教程|考试时间|资源索引|电子书", s):
        return False
    return True


def _is_explicit_toc_marker_text(text: str) -> bool:
    """判断一行是否是明确的全书正文目录标记。"""
    return _is_book_toc_marker_key(title_key(_strip_atx_marker(text)))


def _looks_like_body_structure_heading(text: str) -> bool:
    """判断标题是否像已经进入正文后的章/节结构。"""
    key = title_key(_strip_atx_marker(text))
    if not key:
        return False
    return bool(
        re.match(r"^第[一二两三四五六七八九十百千零〇0-9]+[章节篇卷部]", key)
        or re.match(r"^\d+\.\d+(?:\.\d+)*", key)
    )


def _count_toc_like_lines(lines: list[str], start: int, end: int) -> int:
    """粗略统计目录范围内像目录条目的短行数量。"""
    count = 0
    for i in range(max(1, start) - 1, min(len(lines), end)):
        raw = lines[i].strip()
        if not raw:
            continue
        title = _strip_atx_marker(raw)
        key = title_key(title)
        if len(title) <= 120 and (
            raw.startswith("#")
            or _has_toc_page_marker(raw)
            or _looks_like_body_structure_heading(title)
            or re.match(r"^(附录|参考文献|索引|本阶段总结|习题|练习)", key)
        ):
            count += 1
    return count


def _has_prior_body_structure(lines: list[str], before_line: int) -> bool:
    """目录候选之前若已有多个正文结构标题，说明候选很可能是正文局部标题列表。"""
    hits = 0
    for raw in lines[:max(0, before_line - 1)]:
        stripped = raw.strip()
        if not stripped.startswith("#"):
            continue
        if _looks_like_body_structure_heading(stripped):
            hits += 1
            if hits >= 2:
                return True
    return False


def validate_toc_range_or_raise(md_text: str, toc_range: dict, file_name: str) -> None:
    """合法 JSON 之后的目录范围安全校验；可疑则跳过当前文件。"""
    if not toc_range.get("found"):
        return

    toc_type = str(toc_range.get("toc_type") or "book_toc").strip().lower()
    if toc_type != "book_toc":
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_range",
            f"{file_name}: 第一层候选不是全书正文目录 toc_type={toc_type}",
        )

    lines = md_text.splitlines()
    total = len(lines)
    start = int(toc_range.get("toc_start_line") or 0)
    end = int(toc_range.get("toc_end_line") or 0)
    marker = int(toc_range.get("toc_marker_line") or start or 0)

    if start <= 0 or end <= 0 or start > end or end > total:
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_range",
            f"目录范围行号不合理: start={start}, end={end}, total={total}",
        )

    marker_text = lines[marker - 1] if 1 <= marker <= total else ""
    marker_key = title_key(_strip_atx_marker(marker_text))
    marker_guess = _toc_marker_candidate_type(marker_key)
    if _is_negative_toc_marker_key(marker_key):
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_range",
            f"{file_name}: 目录候选行{marker}像 {marker_guess}，不是全书正文目录: {compact_text(marker_text, 80)}",
        )

    explicit_marker = _is_explicit_toc_marker_text(marker_text)
    toc_like_count = _count_toc_like_lines(lines, start, end)
    span = end - start + 1

    marker_path = _toc_number_path(marker_text)
    if marker > 200 and marker_path and "." in marker_path and _has_prior_body_structure(lines, marker):
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_range",
            f"{file_name}: 目录候选行{marker}从中途小节 {marker_path} 开始，且前文已有正文结构，疑似正文小节",
        )

    if not explicit_marker and toc_like_count < 5:
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_range",
            f"{file_name}: 无明确目录标记且目录条目过少 ({toc_like_count} 条, 跨 {span} 行)",
        )

    if not explicit_marker and marker > 200 and _has_prior_body_structure(lines, marker):
        raise SkipMarkdownFile(
            "toc_range_suspicious",
            "layer1_toc_range",
            f"{file_name}: 目录候选行{marker}无明确目录标记，且前文已有正文章/节结构，疑似假目录",
        )

def _scan_unpaged_toc_tail(md_text: str, toc_range: dict) -> dict:
    """扩展无页码目录尾部。

    有些书的目录没有页码，第一层容易在中途章节处截断。这里用目录编号结构继续
    向后扫描: 如果后续仍是递增的章号/小节号，就继续算作目录；遇到目录第一章
    再次出现时，说明正文开始。
    """
    lines = md_text.splitlines()
    total_lines = len(lines)
    toc_start = int(toc_range.get("toc_start_line") or 0)
    current_end = int(toc_range.get("toc_end_line") or 0)
    if toc_start <= 0 or current_end <= 0 or current_end >= total_lines:
        return toc_range

    first_chapter_key = ""
    first_root_num = None
    highest_root = None

    for i in range(toc_start - 1, min(current_end, total_lines)):
        title = _strip_atx_marker(lines[i])
        path = _toc_number_path(title)
        if not path:
            continue
        root = path.split(".", 1)[0]
        if not root.isdigit():
            continue
        root_num = int(root)
        if first_root_num is None:
            first_root_num = root_num
            first_chapter_key = title_key(title)
        highest_root = root_num if highest_root is None else max(highest_root, root_num)

    if first_root_num is None or highest_root is None:
        return toc_range

    last_toc_line = current_end
    extended = False
    in_tail = False
    nonempty_since_last_number = 0
    scan_limit = min(total_lines, current_end + 1800)

    for line_no in range(current_end + 1, scan_limit + 1):
        raw = lines[line_no - 1]
        stripped = raw.strip()
        if not stripped:
            continue

        title = _strip_atx_marker(stripped)
        key = title_key(title)
        path = _toc_number_path(title)

        if (
            extended
            and first_chapter_key
            and key == first_chapter_key
            and (highest_root or 0) > first_root_num
        ):
            break

        if path:
            root = path.split(".", 1)[0]
            if root.isdigit():
                root_num = int(root)

                # 已经扫到后续章节后，编号回落到第1章/1.1，通常就是正文开始。
                if extended and root_num <= first_root_num and highest_root > first_root_num:
                    break

                # 允许同章小节继续，也允许下一章/少量跳章继续。
                if root_num == highest_root or highest_root < root_num <= highest_root + 2:
                    highest_root = max(highest_root, root_num)
                    last_toc_line = line_no
                    extended = True
                    in_tail = True
                    nonempty_since_last_number = 0
                    continue

            elif in_tail and len(root) == 1 and root.isalpha():
                # 附录 A / A.1 这类尾部结构。
                last_toc_line = line_no
                extended = True
                nonempty_since_last_number = 0
                continue

        if in_tail and _looks_like_toc_continuation_line(stripped):
            last_toc_line = line_no
            nonempty_since_last_number += 1
            if nonempty_since_last_number <= 3:
                continue

        # 出现长正文段落或连续非编号短行，认为目录结束。
        if extended:
            break
        # 还没扩展到任何新目录项，说明原 end 后面不是无页码目录。
        break

    if last_toc_line > current_end:
        new_range = dict(toc_range)
        new_range["toc_end_line"] = last_toc_line
        new_range["unpaged_tail_scanned"] = True
        return new_range
    return toc_range


def _toc_path_sort_key(path: str) -> tuple | None:
    """把目录编号路径转成可比较元组，仅用于判断编号是否向后延续。"""
    parts = (path or "").split(".")
    if not parts:
        return None
    if all(part.isdigit() for part in parts):
        return ("N",) + tuple(int(part) for part in parts)
    if len(parts[0]) == 1 and parts[0].isalpha() and all(
        part.isdigit() for part in parts[1:]
    ):
        return ("A", parts[0].upper()) + tuple(int(part) for part in parts[1:])
    return None


def _is_forward_toc_number_path(previous: str, candidate: str) -> bool:
    """判断 candidate 是否像 previous 后面的连续目录编号。"""
    previous_key = _toc_path_sort_key(previous)
    candidate_key = _toc_path_sort_key(candidate)
    if not previous_key or not candidate_key or previous_key[0] != candidate_key[0]:
        return False
    if candidate_key <= previous_key:
        return False

    if previous_key[0] == "N":
        previous_root = previous_key[1]
        candidate_root = candidate_key[1]
        return candidate_root == previous_root or candidate_root <= previous_root + 2

    previous_root = ord(previous_key[1])
    candidate_root = ord(candidate_key[1])
    return candidate_root == previous_root or candidate_root <= previous_root + 1


def _find_numbered_toc_tail_continuation(
    lines: list[str],
    current_end: int,
    scan_end: int,
    *,
    min_hits: int = 2,
) -> dict | None:
    """查找 current_end 后仍在向前推进的连续编号目录条目。

    该检查只作为 tail 停止前的保护。它不会自行寻找目录起点，也不会跨过明显的
    长正文段落，避免把正文中的后续章节重新并入目录。
    """
    if current_end <= 0 or current_end >= len(lines):
        return None

    reference_path = ""
    for line_no in range(current_end, max(0, current_end - 120), -1):
        reference_path = _toc_number_path(_strip_atx_marker(lines[line_no - 1]))
        if reference_path:
            break
    if not reference_path:
        return None

    sequence_path = reference_path
    hits: list[tuple[int, str]] = []
    scan_end = min(scan_end, len(lines))

    for line_no in range(current_end + 1, scan_end + 1):
        raw = lines[line_no - 1].strip()
        if not raw:
            continue

        title = _strip_atx_marker(raw)
        # 连续长正文是可靠的目录结束信号，不继续越过正文寻找相似编号。
        if len(title) >= 180 and re.search(r"[。！？.!?；;]", title):
            break

        path = _toc_number_path(title)
        if not path or not _is_forward_toc_number_path(sequence_path, path):
            continue

        hits.append((line_no, path))
        sequence_path = path

    if len(hits) < min_hits:
        return None

    return {
        "reference_path": reference_path,
        "first_line": hits[0][0],
        "last_line": hits[-1][0],
        "paths": [path for _, path in hits],
    }


def _scan_toc_tail(md_text: str, toc_range: dict, file_name: str,
                   chunk_size: int = TOC_TAIL_CHUNK_LINES,
                   overlap_lines: int = TOC_TAIL_OVERLAP_LINES) -> dict:
    """第一层 B: 从 LLM 给出的 toc_end_line 向后逐段扫描，找目录底部。

    有页码：chunk 有页码调 LLM，没页码停；LLM 停后用规则收尾。
    没页码：全程调 LLM 按结构判断，LLM 的终点就是最终结果。
    每轮携带少量重叠上下文；接受停止结果前先检查后面是否仍有连续编号目录。
    """
    lines = md_text.splitlines()
    total_lines = len(lines)
    toc_start = int(toc_range.get("toc_start_line") or 0)
    current_end = int(toc_range.get("toc_end_line") or 0)

    if current_end <= 0:
        return toc_range

    # 阶段 0: 全局预判 — TOC 区域有没有页码
    _toc_raw = "\n".join(lines[toc_range["toc_start_line"] - 1 : current_end])
    _is_pageless = not any(_has_toc_page_marker(line) for line in _toc_raw.split("\n"))
    if _is_pageless:
        print("    tail: 目录无页码，全程 LLM 结构判断")

    max_iterations = 20
    for _ in range(max_iterations):
        chunk_start = current_end + 1
        if chunk_start > total_lines:
            break

        chunk_end = min(chunk_start + chunk_size - 1, total_lines)
        context_start = max(toc_start, current_end - overlap_lines + 1)
        context_lines = []
        for i in range(context_start - 1, current_end):
            context_lines.append(f"行{i + 1}: {lines[i]}")
        context_text = "\n".join(context_lines)

        chunk_lines = []
        for i in range(chunk_start - 1, chunk_end):  # 0-indexed
            chunk_lines.append(f"行{i + 1}: {lines[i]}")
        chunk_text = "\n".join(chunk_lines)

        numbered_continuation = _find_numbered_toc_tail_continuation(
            lines, current_end, chunk_end
        )

        # 有页码的书：局部 chunk 没有点线页码时，仍先检查连续编号目录。
        if not _is_pageless and not any(
            _has_toc_page_marker(line) for line in lines[chunk_start - 1:chunk_end]
        ):
            if numbered_continuation:
                print(
                    "    tail连续编号保护: "
                    f"{numbered_continuation['reference_path']} → "
                    f"{numbered_continuation['paths'][-1]}, "
                    f"推进到行{numbered_continuation['last_line']}"
                )
                current_end = int(numbered_continuation["last_line"])
                continue
            break

        # 无页码目录也交给同一个三态边界 prompt；由 LLM 在窗口内判断。
        _system = TOC_TAIL_SYSTEM
        user_prompt = TOC_TAIL_USER.format(
            file_name=file_name,
            current_end=current_end,
            context_text=context_text,
            chunk_text=chunk_text,
        )
        messages = [
            {"role": "system", "content": _system},
            {"role": "user", "content": "/no_think\n" + user_prompt},
        ]

        result = call_llm_json_with_repair(
            messages,
            max_tokens=300,
            stage="toc_tail_scan",
            expected_format='{"decision": "continue|stop|not_toc|uncertain", "last_toc_line": 行号或null, "may_continue_after_chunk": true/false, "reason": "一句话"}',
        )

        decision = str(result.get("decision", "")).strip().lower()
        if decision not in {"continue", "stop", "not_toc", "uncertain"}:
            raise SkipMarkdownFile(
                "toc_tail_suspicious",
                "toc_tail_scan",
                f"tail扫描返回 decision 无效: {result}",
            )

        if decision == "uncertain":
            raise SkipMarkdownFile(
                "toc_tail_suspicious",
                "toc_tail_scan",
                f"tail扫描无法确认目录尾部: {result}",
            )

        if decision in {"continue", "stop"}:
            new_end = result.get("last_toc_line")
            if not isinstance(new_end, int):
                raise SkipMarkdownFile(
                    "toc_tail_suspicious",
                    "toc_tail_scan",
                    f"tail扫描 {decision} 但 last_toc_line 无效: {result}",
                )
            if current_end < new_end <= chunk_end:
                if decision == "continue":
                    current_end = new_end
                    continue

                continuation_after_stop = _find_numbered_toc_tail_continuation(
                    lines, new_end, chunk_end
                )
                if continuation_after_stop:
                    print(
                        "    tail停止结果被连续编号否决: "
                        f"行{new_end}后仍延续到行{continuation_after_stop['last_line']}"
                    )
                    current_end = int(continuation_after_stop["last_line"])
                    continue
                current_end = new_end
                break
            if decision == "stop" and new_end == current_end:
                if numbered_continuation:
                    print(
                        "    tail停止结果被连续编号否决: "
                        f"行{current_end}后仍延续到行{numbered_continuation['last_line']}"
                    )
                    current_end = int(numbered_continuation["last_line"])
                    continue
                break
            raise SkipMarkdownFile(
                "toc_tail_suspicious",
                "toc_tail_scan",
                f"tail扫描 last_toc_line 不在当前检查窗口内: {result}, chunk={chunk_start}-{chunk_end}",
            )

        if decision == "not_toc":
            if numbered_continuation:
                print(
                    "    tail not_toc 被连续编号否决: "
                    f"行{current_end}后仍延续到行{numbered_continuation['last_line']}"
                )
                current_end = int(numbered_continuation["last_line"])
                continue
            break

    extended = dict(toc_range)
    extended.update({
        "toc_marker_line": toc_range.get("toc_marker_line"),
        "toc_start_line": toc_range.get("toc_start_line"),
        "toc_end_line": current_end,
        "found": toc_range.get("found"),
        "toc_type": toc_range.get("toc_type", "book_toc"),
        "tail_scanned": True,
    })
    # 有页码：LLM 停后用规则扫几行收尾（补 chunk 边界切掉的条目）
    # 没页码：LLM 的终点即最终结果，不跑规则
    if _is_pageless:
        return extended
    return _scan_unpaged_toc_tail(md_text, extended)


# ============================================================
# 第二层: 解析目录内容 → 标题层级
# ============================================================


def _is_toc_junk_line(text: str) -> bool:
    """判断目录区域中的低风险格式噪声行。

    只删除确定没有目录语义的内容，例如装饰分隔线、图片、details 标记、代码围栏、
    纯页码。不要删除“本章涉及的文件”“Index”“References”等可能是真目录条目的文本。
    """
    s = (text or "").strip()
    if not s:
        return True

    compact = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[#*_=\-—·.。…]+", compact) and len(compact) >= 3:
        return True

    low = s.lower()
    if re.match(r"^!\[.*\]\(.*\)$", s):
        return True
    if low.startswith(("<details", "</details>", "<summary", "</summary>")):
        return True
    if low.startswith(("```", "~~~")):
        return True

    if re.fullmatch(r"\d{1,5}", compact):
        return True

    return False


def _clean_toc_lines(md_text: str, toc_range: dict) -> tuple[list[tuple[int, str]], int, int]:
    """从 Markdown 中提取并清洗目录区域原文。

    作用:
        第二层只需要目录区域文本。这里按第一层行号截取，删除空行并压缩空白。

    输入:
        md_text: str，完整 Markdown。
        toc_range: dict，目录范围，例如:
            {"toc_start_line": 67, "toc_end_line": 480, "found": True}

    输出:
        tuple:
            cleaned_lines: list[tuple[int, str]]
                每项为 (原文行号, 清洗后文本)，例如 [(73, "# 第1章 概述"), (74, "1.1 背景 3")]
            start_line: int，实际截取起始行号。
            end_line: int，实际截取结束行号。

    例子:
        _clean_toc_lines("# 目录\\n\\n# 第1章 概述", {"toc_start_line": 1, "toc_end_line": 3})
        -> ([(1, "# 目录"), (3, "# 第1章 概述")], 1, 3)
    """
    lines = md_text.split("\n")
    start = max(1, toc_range["toc_start_line"] - 1)
    end = min(len(lines), toc_range["toc_end_line"])

    raw_lines = lines[start:end]
    # 第一步: 清洗空行、压缩空白，并删除确定无目录语义的格式噪声。
    cleaned_lines = []
    junk_count = 0
    for line_no, line in enumerate(raw_lines, start=start + 1):
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue
        if _is_toc_junk_line(stripped):
            junk_count += 1
            continue
        cleaned_lines.append((line_no, stripped))

    if junk_count:
        print(f"  目录清洗: 删除 {junk_count} 行格式噪声")

    return cleaned_lines, start + 1, end


def _format_toc_lines(cleaned_lines: list[tuple[int, str]]) -> str:
    """把清洗后的目录行格式化为 LLM 输入文本。

    输入:
        cleaned_lines: list[tuple[int, str]]，例如 [(73, "# 第1章 概述")]

    输出:
        str，每行形如 "行73: # 第1章 概述"。

    例子:
        _format_toc_lines([(73, "# 第1章 概述")]) -> "行73: # 第1章 概述"
    """
    return "\n".join(f"行{line_no}: {text}" for line_no, text in cleaned_lines)


def _call_toc_parse_llm(messages: list[dict], max_tokens: int, label: str) -> list[dict]:
    """调用 LLM 解析一个目录块。

    作用:
        第二层单批或分块都会调用它。它负责发送请求、解析 JSON、发现截断、触发修复重试。

    输入:
        messages: list[dict]，OpenAI 消息格式。
        max_tokens: int，本批最大输出 token。
        label: str，日志标签，例如 "batch 3/9"。

    输出:
        list[dict]，标题列表，例如:
            [{"text": "第1章 概述", "level": 2}, {"text": "1.1 背景", "level": 3}]

    异常:
        RuntimeError: JSON 截断或修复失败。

    例子:
        _call_toc_parse_llm(messages, 8000, "batch 1/3")
    """
    content = ""
    for attempt in range(2):
        try:
            content = call_llm(messages, max_tokens=max_tokens, timeout=900)
            result = parse_json(content)
            headings_list = result.get("headings", [])
            if headings_list:
                h1_count = sum(1 for h in headings_list if h.get("level") == 1)
                print(f"    {label}: {len(headings_list)} 个标题, H1={h1_count}")
                return headings_list
            raise ValueError("headings 列表为空")
        except ValueError as e:
            stripped_content = (content or "").strip()
            is_truncated = (
                bool(stripped_content)
                and (
                    stripped_content.startswith("{")
                    or stripped_content.startswith("```")
                )
                and (
                    not stripped_content.endswith("}")
                    or content.count("{") > content.count("}")
                )
            )
            if is_truncated and attempt == 0:
                raise RuntimeError(
                    f"第二层 {label} JSON 被截断 (max_tokens={max_tokens}不足)。"
                    f"请增大 --toc-chunk-max-tokens 或减小 --toc-chunk-lines。"
                    f"响应前200字符: {content[:200] if content else 'N/A'}"
                )
            if attempt < 1:
                messages += [
                    {"role": "assistant", "content": content[:1200] if content else ""},
                    {"role": "user", "content": REPAIR_PROMPT.format(
                        expected_format='{"headings": [{"text": "...", "level": 2}, ...]}')}]
            else:
                raise RuntimeError(f"第二层 {label} 目录解析失败: {e}")
        except Exception as e:
            if attempt < 1 and not isinstance(e, RuntimeError):
                time.sleep(2)
            else:
                raise
    return []


def _heading_leading_number(text: str) -> tuple[int, int] | None:
    """提取标题开头的 x.y 编号，用于判断第二层是否从中途编号开始。"""
    m = re.match(r"^\s*(\d+)\.(\d+)(?:\D|$)", display_title(text or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _looks_like_toc_prefix_line(text: str) -> bool:
    """判断一行是否像目录前缀标题，用于截取漏解析窗口。"""
    s = display_title(text or "").strip()
    if not s:
        return False
    key = title_key(s)
    if _toc_marker_candidate_type(key) == "book_toc":
        return True
    return bool(
        re.match(r"^第[一二两三四五六七八九十百千零〇0-9]+[章节篇卷部]", key)
        or re.match(r"^\d+(?:\.\d+)+", key)
        or re.match(r"^chapter\s*\d+", key, flags=re.IGNORECASE)
        or _has_toc_page_marker(s)
    )


def _repair_layer2_missing_prefix_by_llm(headings: list[dict],
                                         cleaned_lines: list[tuple[int, str]],
                                         file_name: str,
                                         chunk_max_tokens: int) -> list[dict]:
    """第二层后置自检: 解析结果从 1.5 等中途编号开始时，用 LLM 补回前缀。"""
    if not headings or not cleaned_lines:
        return headings

    first_num = _heading_leading_number(headings[0].get("text", ""))
    if not first_num:
        return headings
    chapter_no, section_no = first_num
    if chapter_no != 1 or section_no <= 1:
        return headings

    target_prefix = f"1.{section_no}"
    prefix_end_index = None
    saw_chapter = False
    saw_early_section = False
    for idx, (_, text) in enumerate(cleaned_lines):
        key = title_key(display_title(text))
        if re.match(r"^第[一二两三四五六七八九十百千零〇0-9]+[章节篇卷部]", key):
            saw_chapter = True
        m = re.match(r"^1\.(\d+)", key)
        if m and int(m.group(1)) < section_no:
            saw_early_section = True
        if re.match(rf"^{re.escape(target_prefix)}(?:\D|$)", key):
            prefix_end_index = idx
            break

    if prefix_end_index is None or not (saw_chapter and saw_early_section):
        return headings

    prefix_lines = [
        (ln, text)
        for ln, text in cleaned_lines[:prefix_end_index]
        if _looks_like_toc_prefix_line(text)
    ]
    if not prefix_lines:
        return headings

    expected = '{"headings": [{"text": "...", "level": 2}, ...]}'
    prompt = f"""第二层目录解析结果疑似漏掉了开头前缀。

文件名: {file_name}

已解析结果的第一个标题是:
{headings[0].get("text", "")}

这说明前面的第1章/1.1...1.{section_no - 1} 可能被误删。

请只解析下面这段“漏掉的目录前缀”，返回应插入到现有 headings 最前面的标题列表。

规则:
1. 只解析下面给出的前缀行，不要输出后续已经解析到的 {target_prefix}。
2. 即使局部没有页码，只要是连续编号/层级目录项，也必须提取。
3. 去掉页码、点线、省略号。
4. 第X章为 level=2；1.1/1.2 为 level=3；1.1.1 为 level=4。
5. 只返回 JSON，不要解释。

漏掉的目录前缀:
{_format_toc_lines(prefix_lines)}

返回格式:
{expected}"""
    messages = [
        {"role": "system", "content": "你是书籍目录结构修复器。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    result = call_llm_json_with_repair(
        messages,
        max_tokens=max(1000, min(chunk_max_tokens, len(prefix_lines) * 120)),
        stage="layer2_missing_prefix_repair",
        expected_format=expected,
        timeout=TIMEOUT,
    )
    prefix_headings = result.get("headings", [])
    if not isinstance(prefix_headings, list) or not prefix_headings:
        raise SkipMarkdownFile(
            "layer2_missing_prefix_repair_failed",
            "layer2_missing_prefix_repair",
            f"第二层前缀修复未返回 headings: {result}",
        )

    existing_keys = {title_key(h.get("text", "")) for h in headings}
    repaired = []
    for h in prefix_headings:
        key = title_key(h.get("text", ""))
        if not key or key in existing_keys:
            continue
        repaired.append({
            "text": display_title(h.get("text", "")),
            "level": int(h.get("level") or 3),
        })

    if repaired:
        print(
            f"  第二层前缀修复: 检测到结果从 {headings[0].get('text', '')} 开始，"
            f"补回 {len(repaired)} 个前缀标题"
        )
        return repaired + headings
    return headings


def _toc_number_depth_from_path(path: str) -> int:
    """计算编号路径深度，如 16 -> 1, 16.1.7 -> 3, A.2 -> 2。"""
    path = (path or "").strip()
    if not path:
        return 0
    return path.count(".") + 1


def _toc_number_root(path: str) -> str:
    """取编号根路径，用于判断两个目录编号是否在同一章/附录下。"""
    path = (path or "").strip()
    if not path:
        return ""
    return path.split(".", 1)[0].upper()


def _is_short_unnumbered_toc_title(text: str) -> bool:
    """判断目录标题是否像 OCR 拆出的无编号短续行。"""
    title = display_title(clean_title(text or ""))
    if not title:
        return False
    if len(title) > 80:
        return False
    key = title_key(title)
    if not key or _toc_marker_candidate_type(key) == "book_toc":
        return False
    if _toc_number_path(title):
        return False
    if re.match(r"^(第|chapter|part|appendix|附录)", key, flags=re.IGNORECASE):
        return False
    if re.search(r"(。|！|？|；|;)$", title):
        return False
    if re.search(r"https?://|www\.", title, flags=re.IGNORECASE):
        return False
    return True


def _toc_paths_related_for_wrap(prev_path: str, next_path: str) -> bool:
    """判断前后两个编号是否足以构成“续行夹在编号标题之间”的上下文。"""
    prev_path = (prev_path or "").strip()
    next_path = (next_path or "").strip()
    if not prev_path or not next_path:
        return False
    if _toc_number_root(prev_path) != _toc_number_root(next_path):
        return False
    if next_path.startswith(prev_path + ".") or prev_path.startswith(next_path + "."):
        return True
    return True


def _layer2_wrap_group_needs_llm(prev: dict, conts: list[dict], next_item: dict) -> bool:
    """判断一个无编号夹心组是否值得交给 LLM 做逻辑标题恢复。"""
    prev_path = _toc_number_path(prev.get("text", ""))
    next_path = _toc_number_path(next_item.get("text", ""))
    if not _toc_paths_related_for_wrap(prev_path, next_path):
        return False

    prev_depth = _toc_number_depth_from_path(prev_path)
    next_depth = _toc_number_depth_from_path(next_path)
    prev_level = int(prev.get("level") or 6)
    next_level = int(next_item.get("level") or 6)
    cont_levels = [int(x.get("level") or 6) for x in conts]

    if next_path.startswith(prev_path + "."):
        return True
    if next_depth <= prev_depth:
        return True
    if any(level <= prev_level for level in cont_levels):
        return True
    if any(level <= next_level for level in cont_levels):
        return True
    return False


def _find_layer2_wrapped_title_groups(headings: list[dict]) -> list[dict]:
    """找出“编号标题 + 无编号短标题 + 相关编号标题”的可疑断行组。"""
    groups = []
    i = 1
    group_id = 1
    while i < len(headings) - 1:
        prev_index = i - 1
        prev = headings[prev_index]
        if not _toc_number_path(prev.get("text", "")):
            i += 1
            continue

        cont_indices = []
        j = i
        while j < len(headings) and len(cont_indices) < 3 \
                and _is_short_unnumbered_toc_title(headings[j].get("text", "")):
            cont_indices.append(j)
            j += 1

        if not cont_indices or j >= len(headings):
            i += 1
            continue

        next_item = headings[j]
        if _toc_number_path(next_item.get("text", "")) and _layer2_wrap_group_needs_llm(
            prev,
            [headings[k] for k in cont_indices],
            next_item,
        ):
            groups.append({
                "id": group_id,
                "prev_index": prev_index,
                "cont_indices": cont_indices,
                "next_index": j,
            })
            group_id += 1
            i = j
            continue

        i += 1
    return groups


def _fallback_merge_toc_title(prev_text: str, cont_texts: list[str]) -> str:
    """不信任 LLM merged_text 时，用本地规则拼出保守合并标题。"""
    merged = display_title(clean_title(prev_text or ""))
    merged_key = title_key(merged)
    for text in cont_texts:
        part = display_title(clean_title(text or ""))
        part_key = title_key(part)
        if not part_key or part_key in merged_key:
            continue
        merged = f"{merged} {part}".strip()
        merged_key = title_key(merged)
    return merged


def _call_layer2_wrap_repair_llm(groups: list[dict], headings: list[dict],
                                 file_name: str, chunk_max_tokens: int) -> dict[int, int]:
    """让 LLM 判断可疑无编号短标题是否应合并到上一条编号标题。"""
    expected = (
        '{"fixes": [{"id": 1, '
        '"relation": "continuation_of_previous|child_of_previous|sibling_of_previous|independent", '
        '"confidence": 0.95, "merge_count": 1, "merged_text": "...", "reason": "..."}]}'
    )
    group_blocks = []
    for group in groups:
        prev = headings[group["prev_index"]]
        next_item = headings[group["next_index"]]
        cont_lines = []
        for offset, idx in enumerate(group["cont_indices"], 1):
            item = headings[idx]
            cont_lines.append(
                f"  {offset}. index={idx + 1}, H{int(item.get('level') or 2)}: "
                f"{display_title(item.get('text', ''))}"
            )
        group_blocks.append(
            "\n".join([
                f"id={group['id']}",
                (
                    f"previous index={group['prev_index'] + 1}, "
                    f"H{int(prev.get('level') or 2)}: {display_title(prev.get('text', ''))}"
                ),
                "candidate_continuations:",
                "\n".join(cont_lines),
                (
                    f"next index={group['next_index'] + 1}, "
                    f"H{int(next_item.get('level') or 2)}: {display_title(next_item.get('text', ''))}"
                ),
            ])
        )

    prompt = f"""禁止分析过程。你的回复第一个字符必须是 {{，最后一个字符必须是 }}。

文件名: {file_name}

任务：判断目录解析结果中的无编号短标题，是否其实是上一条编号标题被 OCR 换行拆出的续行/副标题。

判断规则：
1. 默认采取保守关系 independent / child_of_previous / sibling_of_previous。只有 previous 的语义明显没有完成，
   并且 candidate_continuations 能直接补全 previous，才 relation=continuation_of_previous。
2. 合并时保留 previous 的等级；被合并的 candidate 不再作为独立目录标题。
3. 如果连续出现多个语义完整、句式相似、主题并列的 candidate，它们通常是 previous 下面的独立子标题，
   必须 action=keep, merge_count=0；不要把它们拼成一个超长标题。
4. 如果只有前几个 candidate 是续行，merge_count 写需要合并的数量。
5. merged_text 必须是 previous + 被合并 candidate 后的完整逻辑标题，去掉页码/点线。
6. 不要创造不存在的标题，不要改变 next。
7. next 与 previous 属于同一章或编号连续，只能说明候选位于两者之间，不能单独证明候选应合并。

正确示例：
- previous: 16.1.7 创建GUI的其他方法
  candidate: 平台无关的窗口API
  next: 16.2 GTK+简介
  => continuation_of_previous, merged_text="16.1.7 创建GUI的其他方法 平台无关的窗口API"
- previous: 16.2.4 安装
  candidate: GNOME/GTK+开发库
  next: 16.3 事件、信号和回调函数
  => continuation_of_previous, merged_text="16.2.4 安装 GNOME/GTK+开发库"
- previous: 13.5.2 把管道用作标准输入
  candidate: 和标准输出
  next: 13.6 命名管道：FIFO
  => continuation_of_previous, merged_text="13.5.2 把管道用作标准输入 和标准输出"
- previous: 2.3 Tokenization and LLM capabilities
  candidates: LLMs are bad at word games / LLMs are challenged by mathematics / LLMs and language equity
  next: 2.4 Special tokens
  => child_of_previous 或 sibling_of_previous；三个 candidate 都是语义完整的独立标题，不是 2.3 的续行
- previous: 5.2 Backing up data
  candidates: Installing the command-line interface / Configuring your account / Creating your first bucket
  next: 5.3 Scheduling backups
  => child_of_previous；三个完整操作短语必须保持为三个独立标题，不能拼成长标题

可疑组：
{chr(10).join(group_blocks)}

返回 JSON：
{expected}"""

    messages = [
        {"role": "system", "content": "你是目录逻辑标题修复器。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    result = call_llm_json_with_repair(
        messages,
        max_tokens=max(800, min(chunk_max_tokens, 400 + len(groups) * 180)),
        stage="layer2_wrapped_title_repair",
        expected_format=expected,
        timeout=TIMEOUT,
    )
    fixes = result.get("fixes", [])
    if not isinstance(fixes, list):
        raise SkipMarkdownFile(
            "layer2_wrapped_title_repair_failed",
            "layer2_wrapped_title_repair",
            f"第二层逻辑标题修复未返回 fixes 列表: {result}",
        )

    merge_counts: dict[int, int] = {}
    groups_by_id = {int(g["id"]): g for g in groups}
    for item in fixes:
        try:
            item_id = int(item.get("id"))
        except Exception:
            continue
        group = groups_by_id.get(item_id)
        if not group:
            continue
        relation = str(item.get("relation", "")).strip().lower()
        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if relation != "continuation_of_previous" or confidence < 0.90:
            merge_counts[item_id] = 0
            continue
        try:
            merge_count = int(item.get("merge_count") or 0)
        except Exception:
            merge_count = 0
        merge_counts[item_id] = max(0, min(merge_count, len(group["cont_indices"])))
    return merge_counts


def _repair_layer2_wrapped_titles_by_llm(headings: list[dict],
                                         file_name: str,
                                         chunk_max_tokens: int) -> list[dict]:
    """第二层后置自检: 用 LLM 合并被 OCR 断成多行的逻辑目录标题。"""
    if len(headings) < 3:
        return headings

    groups = _find_layer2_wrapped_title_groups(headings)
    if not groups:
        return headings

    merge_counts: dict[int, int] = {}
    batch_size = 20
    for start in range(0, len(groups), batch_size):
        batch = groups[start:start + batch_size]
        merge_counts.update(_call_layer2_wrap_repair_llm(
            batch,
            headings,
            file_name,
            chunk_max_tokens,
        ))

    skip_indices: set[int] = set()
    replacement_text: dict[int, str] = {}
    merged_groups = 0
    for group in groups:
        merge_count = merge_counts.get(int(group["id"]), 0)
        if merge_count <= 0:
            continue

        prev_index = int(group["prev_index"])
        selected_indices = group["cont_indices"][:merge_count]
        prev_text = headings[prev_index].get("text", "")
        cont_texts = [headings[idx].get("text", "") for idx in selected_indices]
        merged_text = _fallback_merge_toc_title(prev_text, cont_texts)
        if not merged_text:
            continue

        replacement_text[prev_index] = merged_text
        skip_indices.update(selected_indices)
        merged_groups += 1

    if not skip_indices:
        return headings

    repaired = []
    for idx, heading in enumerate(headings):
        if idx in skip_indices:
            continue
        item = dict(heading)
        if idx in replacement_text:
            item["text"] = replacement_text[idx]
            item.setdefault("_fixed", "")
            item["_fixed"] += "第二层逻辑标题续行合并"
        repaired.append(item)

    print(
        f"  第二层逻辑标题修复: 检测到 {len(groups)} 个疑似断行组，"
        f"合并 {merged_groups} 组，删除 {len(skip_indices)} 个续行节点"
    )
    return repaired


def _should_layer2_confirm_toc_tail(cleaned_lines: list[tuple[int, str]]) -> tuple[bool, int | None]:
    """判断第一层给出的目录尾部是否需要第二层再确认。

    触发信号:
        目录已经进入较后章节后，尾部又突然回到第1章/第2章。
        这通常表示第一层把正文开头、样章、广告页或章节首页也吃进了目录范围。
    """
    max_seen: dict[str, int] = {}
    for idx, (_, text) in enumerate(cleaned_lines):
        key = title_key(display_title(text))
        marker = _toc_structural_marker(key)
        if not marker:
            continue
        kind, number = marker
        if number <= 0:
            continue
        previous_max = max_seen.get(kind, 0)
        if previous_max >= 2 and number < previous_max:
            return True, idx
        if number > previous_max:
            max_seen[kind] = number
            if kind == "part":
                # 新篇/部分正常推进时，其下章节允许从 1 重新计数。
                max_seen["chapter"] = 0
    return False, None


def _toc_tail_line_score(text: str) -> int:
    """给目录尾部候选行打分，用于防止 LLM 把真实目录条目截掉。"""
    raw = display_title(text or "").strip()
    if not raw:
        return 0
    key = title_key(raw)
    score = 0
    if _has_toc_page_marker(raw) or re.search(r"\s+\d{1,4}\s*$", raw):
        score += 3
    if raw.startswith("#"):
        score += 2
    if _is_toc_backmatter_key(key):
        score += 4
    if _toc_structural_marker(key) is not None:
        score += 3
    if re.match(r"^\d+\s*[-.]\s*\d+", key) or re.match(r"^\d+(?:\.\d+)+", key):
        score += 2
    if 2 <= len(raw) <= 90 and not re.search(r"[。！？；;]$", raw):
        score += 1
    if len(raw) > 120:
        score -= 3
    return score


def _last_plausible_toc_line_before_reset(cleaned_lines: list[tuple[int, str]],
                                          suspicious_idx: int) -> int | None:
    """找到重启到第1章之前最后一个明显仍像目录条目的行。"""
    for idx in range(suspicious_idx - 1, max(-1, suspicious_idx - 260), -1):
        line_no, text = cleaned_lines[idx]
        if _toc_tail_line_score(text) >= 4:
            return int(line_no)
    return None


def _confirm_layer2_toc_range_by_llm(md_text: str, toc_range: dict,
                                     file_name: str) -> dict:
    """第二层范围确认: 第一层给宽范围后，由 LLM 再判断真实目录尾部。

    第一层负责“宁可多给，不要漏目录”；第二层在解析目录前负责把明显混入的
    正文开头/样章目录/广告页截掉，避免后续目录树被污染。
    """
    if not toc_range.get("found"):
        return toc_range

    candidate_range = dict(toc_range)
    original_end = int(candidate_range.get("layer2_original_toc_end_line") or 0)
    current_end = int(candidate_range.get("toc_end_line") or 0)
    if original_end > current_end:
        candidate_range["toc_end_line"] = original_end

    cleaned_lines, start_line, end_line = _clean_toc_lines(md_text, candidate_range)
    if len(cleaned_lines) < 20:
        return toc_range

    need_confirm, suspicious_idx = _should_layer2_confirm_toc_tail(cleaned_lines)
    if not need_confirm or suspicious_idx is None:
        return toc_range

    context_start = max(0, suspicious_idx - 180)
    context_end = min(len(cleaned_lines), suspicious_idx + 80)
    context_lines = cleaned_lines[context_start:context_end]
    current_end = int(candidate_range.get("toc_end_line") or end_line)
    safe_last_before_reset = _last_plausible_toc_line_before_reset(cleaned_lines, suspicious_idx)
    expected = '{"last_toc_line": 123, "trim": true, "reason": "一句话"}'
    prompt = f"""禁止分析过程。你的回复第一个字符必须是 {{，最后一个字符必须是 }}。

文件名: {file_name}

第一层给出的目录范围是行 {start_line}-{current_end}。第一层的职责是给宽范围，所以可能把目录后的正文开头也包含进来。

你的任务：在下面的尾部上下文中确认“真实目录最后一行”。

判断规则：
1. 目录通常是密集标题列表，常带页码或连续章节/小节编号。
2. 如果目录已经到较后章节，然后尾部又突然出现“第1章/第2章/1 1/1-1”等回到开头的结构，
   并且后面开始出现正文段落，那么这通常不是目录继续，而是正文首页/样章/版式残留。
3. last_toc_line 应该返回最后一个仍属于真实目录的原文行号。
4. 注意：不能把重启信号之前的正常目录条目截掉。例如已经进入第10章后，
   第10章及其小节/专栏仍然是目录；如果第1章重复出现在它们后面，应截到第1章重复之前，
   而不是截到第10章开始处。
5. 如果目录尾部之后出现“版权信息/ISBN/出版社/作者/前言正文/网址/日期”等非目录段落，
   last_toc_line 应该是这些非目录段落之前的最后一个目录条目，而不是后面再次出现的正文标题。
6. 如果当前范围没有吃多，返回 last_toc_line={current_end}, trim=false。
7. 不要因为普通“专栏/小结/习题”就截断；它们可能是目录条目。
8. 书籍可能先给出 `brief contents / contents at a glance / 简目 / 章节速览`，随后重新从 Part 1、Chapter 1、
   第1章或 `1 Title` 开始给出更密集、更深层级的详细目录。此时重启不是正文开始，不得在详细目录之前截断。
9. 判断重启是“详细目录”还是“正文”的关键是后续形态：后面持续出现密集短标题、子标题、页码或编号序列，
   属于详细目录；标题后开始出现连续段落、代码、表格或操作说明，才属于正文。
10. 如果前面是简略目录、后面是详细目录，应优先保留详细目录；不能仅因章节编号从头开始就 trim。
11. 只返回 JSON，不要解释。

尾部上下文：
{_format_toc_lines(context_lines)}

返回 JSON：
{expected}"""

    messages = [
        {"role": "system", "content": "你是目录范围复核器。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    result = call_llm_json_with_repair(
        messages,
        max_tokens=800,
        stage="layer2_toc_range_confirm",
        expected_format=expected,
        timeout=TIMEOUT,
    )

    try:
        last_toc_line = int(result.get("last_toc_line"))
    except Exception:
        raise SkipMarkdownFile(
            "layer2_toc_range_confirm_failed",
            "layer2_toc_range_confirm",
            f"第二层目录范围确认未返回有效 last_toc_line: {result}",
        )

    if not (start_line <= last_toc_line <= current_end):
        raise SkipMarkdownFile(
            "layer2_toc_range_confirm_failed",
            "layer2_toc_range_confirm",
            f"第二层目录范围确认返回行号越界: {result}",
        )

    suspicious_line = int(cleaned_lines[suspicious_idx][0])
    if safe_last_before_reset and last_toc_line >= suspicious_line:
        print(
            f"  第二层范围确认: LLM 截到正文重启行 {last_toc_line}，"
            f"按重启信号前最后目录条目修正为 {safe_last_before_reset}"
        )
        last_toc_line = safe_last_before_reset
    elif safe_last_before_reset and last_toc_line < safe_last_before_reset:
        print(
            f"  第二层范围确认: LLM 回退过早 {last_toc_line}，"
            f"按重启信号前最后目录条目修正为 {safe_last_before_reset}"
        )
        last_toc_line = safe_last_before_reset

    if last_toc_line >= current_end:
        print("  第二层范围确认: 第一层目录尾部保留")
        return toc_range

    trimmed = dict(toc_range)
    trimmed["toc_end_line"] = last_toc_line
    trimmed["layer2_range_confirmed"] = True
    trimmed["layer2_original_toc_end_line"] = current_end
    print(
        f"  第二层范围确认: 目录尾部回退 {current_end} → {last_toc_line} "
        f"({str(result.get('reason', ''))[:60]})"
    )
    return trimmed


def _dedupe_adjacent_headings(headings: list[dict]) -> list[dict]:
    """去掉相邻重复标题。

    作用:
        第二层分块时，若未来加入重叠窗口，边界处可能重复解析同一标题。
        当前无重叠时也保留该保护，不影响正常数据。

    输入:
        headings: list[dict]，例如:
            [{"text": "1.1 背景", "level": 3}, {"text": "1.1 背景", "level": 3}]

    输出:
        list[dict]，相邻重复项只保留一个。

    例子:
        _dedupe_adjacent_headings([{"text": "A", "level": 2}, {"text": "A", "level": 2}])
        -> [{"text": "A", "level": 2}]
    """
    deduped = []
    last_key = None
    for heading in headings:
        key = (title_key(str(heading.get("text", ""))), heading.get("level"))
        if key[0] and key == last_key:
            continue
        deduped.append(heading)
        last_key = key
    return deduped


def _validate_layer2_logical_recovery_result(
    result: dict,
    valid_lines: set[int],
    core_lines: set[int],
    label: str,
    require_complete: bool = True,
) -> tuple[list[dict], set[int], set[int]]:
    """校验第二层逻辑标题恢复结果，不参与判断标题边界。"""
    raw_headings = result.get("headings", [])
    if not isinstance(raw_headings, list):
        raise SkipMarkdownFile(
            "layer2_logical_recovery_invalid",
            "layer2_logical_recovery",
            f"{label}: headings 不是列表",
        )

    headings = []
    for item in raw_headings:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        raw_sources = item.get("source_lines", [])
        if not text or not isinstance(raw_sources, list):
            continue
        try:
            source_lines = sorted(set(int(x) for x in raw_sources))
        except Exception:
            source_lines = []
        valid_source_lines = [line for line in source_lines if line in valid_lines]
        source_inferred = False
        if not valid_source_lines:
            # source_lines 只用于追踪和分块归属。模型偶尔会把书页页码误当成
            # 原文行号；保留标题内容，并使用当前核心块起点作为诊断占位。
            valid_source_lines = [min(core_lines)]
            source_inferred = True
        source_lines = valid_source_lines
        # 重叠窗口只负责最早来源行落在核心跨度内的标题。允许 LLM 把来源
        # 标到相邻空白行；覆盖诊断仍只检查清洗后的非空目录行。
        if not (min(core_lines) <= source_lines[0] <= max(core_lines)):
            continue
        heading = {
            "text": text,
            "source_lines": source_lines,
            "_source_inferred": source_inferred,
        }
        relation = str(item.get("relation") or "").strip().lower()
        if relation in {
            "independent",
            "continuation_of_previous",
            "child_of_previous",
            "sibling_of_previous",
            "non_title",
        }:
            heading["_layer2_relation"] = relation
            try:
                heading["_layer2_relation_confidence"] = max(
                    0.0, min(float(item.get("confidence", 0.0)), 1.0)
                )
            except Exception:
                heading["_layer2_relation_confidence"] = 0.0
            merged_text = str(item.get("merged_text") or "").strip()
            if merged_text:
                heading["_layer2_merged_text"] = merged_text
        headings.append(heading)

    def parse_line_set(key: str) -> set[int]:
        raw = result.get(key, [])
        if not isinstance(raw, list):
            raise SkipMarkdownFile(
                "layer2_logical_recovery_invalid",
                "layer2_logical_recovery",
                f"{label}: {key} 不是列表",
            )
        try:
            values = {int(x) for x in raw}
        except Exception:
            raise SkipMarkdownFile(
                "layer2_logical_recovery_invalid",
                "layer2_logical_recovery",
                f"{label}: {key} 包含非法行号",
            )
        # ignored/carried 只是核心区覆盖诊断。重叠上下文由相邻块负责，直接过滤。
        return values & core_lines

    ignored_lines = parse_line_set("ignored_lines")
    carried_lines = parse_line_set("carried_lines")
    accounted = {
        line
        for heading in headings
        for line in heading["source_lines"]
        if line in core_lines
    } | ignored_lines | carried_lines
    missing = sorted(core_lines - accounted)
    if missing:
        message = f"{label}: LLM 未说明这些目录原文行的归属: {missing[:20]}"
        if require_complete:
            raise SkipMarkdownFile(
                "layer2_logical_recovery_incomplete",
                "layer2_logical_recovery",
                message,
            )
        print(f"    [layer2_logical_review] 覆盖诊断警告: {message}")
    return headings, ignored_lines, carried_lines


def _apply_layer2_inline_relations(headings: list[dict]) -> list[dict]:
    """只执行 LLM 高置信度确认的续行合并，其他关系一律保留独立标题。"""
    resolved: list[dict] = []
    merged_count = 0
    for item in headings:
        relation = str(item.get("_layer2_relation") or "").strip().lower()
        try:
            confidence = float(item.get("_layer2_relation_confidence", 0.0))
        except Exception:
            confidence = 0.0
        merged_text = str(item.get("_layer2_merged_text") or "").strip()
        previous_text = str(resolved[-1].get("text") or "").strip() if resolved else ""
        merged_key = title_key(merged_text)
        previous_key = title_key(previous_text)
        current_key = title_key(str(item.get("text") or ""))
        merge_preserves_fragments = bool(
            merged_key
            and previous_key
            and current_key
            and previous_key in merged_key
            and current_key in merged_key
        )
        if (
            relation == "continuation_of_previous"
            and confidence >= 0.85
            and merged_text
            and resolved
            and merge_preserves_fragments
        ):
            previous = resolved[-1]
            previous["text"] = merged_text
            previous["source_lines"] = sorted(
                set(previous.get("source_lines", []))
                | set(item.get("source_lines", []))
            )
            previous["_layer2_relation"] = "continuation_merged"
            previous["_layer2_relation_confidence"] = confidence
            merged_count += 1
            continue
        if relation == "continuation_of_previous" and confidence >= 0.85:
            item["_layer2_relation"] = "continuation_rejected"
        resolved.append(item)
    if merged_count:
        print(f"    第二层内联关系: 合并 {merged_count} 个明确续行片段")
    return resolved


def _resolve_layer2_missing_coverage_by_llm(
    reviewed_result: dict,
    context_lines: list[tuple[int, str]],
    core_lines: list[tuple[int, str]],
    file_name: str,
    chunk_index: int,
    chunk_count: int,
    chunk_max_tokens: int,
) -> dict:
    """让 LLM 专门判断复核结果中仍未说明归属的原文行。"""
    core_numbers = {line_no for line_no, _ in core_lines}
    valid_context_numbers = {line_no for line_no, _ in context_lines}
    covered = set()
    for item in reviewed_result.get("headings", []):
        if not isinstance(item, dict):
            continue
        for line in item.get("source_lines", []):
            try:
                covered.add(int(line))
            except Exception:
                pass
    for key in ("ignored_lines", "carried_lines"):
        for line in reviewed_result.get(key, []):
            try:
                covered.add(int(line))
            except Exception:
                pass
    missing = sorted(core_numbers - covered)
    if not missing:
        return reviewed_result

    expected = """{
  "attachments": [{"line": 115, "heading_texts": ["完整标题"]}],
  "add_headings": [{"text": "遗漏的完整标题", "source_lines": [135]}],
  "ignored_lines": [],
  "carried_lines": []
}"""
    prompt = f"""禁止分析过程，只返回严格 JSON。

文件名: {file_name}
当前目录覆盖修复块: {chunk_index}/{chunk_count}

上一轮目录逻辑标题复核后，以下核心原文行仍未说明归属：
{missing}

任务：只处理这些缺失行，判断它们如何归入目录逻辑标题。

动作说明：
1. 如果缺失行属于已经恢复出的一个或多个标题，在 attachments 中返回该行和对应 heading_texts。
2. 如果缺失行包含上一轮完全遗漏的逻辑标题，在 add_headings 中返回完整标题和全部 source_lines。
3. 如果缺失行确定不是目录标题内容，放入 ignored_lines。
4. 如果缺失行只是前一个分块已开始标题的续行，放入 carried_lines。
5. 每个缺失行必须且只能通过上述动作之一得到解释。
6. 不要修改其他标题，不要分配等级，不要创造原文不存在的标题。

目录原文：
{_format_toc_lines(context_lines)}

上一轮复核结果：
{json.dumps(reviewed_result, ensure_ascii=False)}

返回格式：
{expected}"""
    result = call_llm_json_with_repair(
        [
            {
                "role": "system",
                "content": (
                    "你是目录原文行归属解析器。只解决未覆盖行，"
                    "禁止分析过程，只能输出严格 JSON。/no_think"
                ),
            },
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
        max_tokens=max(1000, min(chunk_max_tokens, 1200 + len(missing) * 300)),
        stage="layer2_logical_coverage_repair",
        expected_format=expected,
        timeout=TIMEOUT,
    )

    repaired = json.loads(json.dumps(reviewed_result, ensure_ascii=False))
    headings = repaired.setdefault("headings", [])
    headings_by_key: dict[str, list[dict]] = {}
    for heading in headings:
        if isinstance(heading, dict):
            headings_by_key.setdefault(title_key(heading.get("text", "")), []).append(heading)

    resolved_lines: set[int] = set()
    for attachment in result.get("attachments", []):
        if not isinstance(attachment, dict):
            continue
        try:
            line = int(attachment.get("line"))
        except Exception:
            continue
        if line not in missing:
            continue
        target_texts = attachment.get("heading_texts", [])
        if not isinstance(target_texts, list) or not target_texts:
            continue
        attached = False
        for target_text in target_texts:
            matches = headings_by_key.get(title_key(str(target_text)), [])
            for match in matches:
                source_lines = match.setdefault("source_lines", [])
                if line not in source_lines:
                    source_lines.append(line)
                    source_lines.sort()
                attached = True
        if attached:
            resolved_lines.add(line)

    for item in result.get("add_headings", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        try:
            source_lines = sorted(set(int(x) for x in item.get("source_lines", [])))
        except Exception:
            continue
        if (
            not text
            or not source_lines
            or not set(source_lines) <= valid_context_numbers
            or not (set(source_lines) & set(missing))
        ):
            continue
        headings.append({"text": text, "source_lines": source_lines})
        resolved_lines.update(set(source_lines) & set(missing))

    for key in ("ignored_lines", "carried_lines"):
        values = repaired.setdefault(key, [])
        for raw_line in result.get(key, []):
            try:
                line = int(raw_line)
            except Exception:
                continue
            if line in missing and line not in values:
                values.append(line)
                resolved_lines.add(line)

    unresolved = sorted(set(missing) - resolved_lines)
    if unresolved:
        print(
            "    [layer2_logical_coverage_repair] "
            f"仍有 {len(unresolved)} 行未获得精确 source_lines 归属，保留 LLM 标题结果"
        )
    return repaired


def _call_layer2_logical_recovery_llm(
    file_name: str,
    context_lines: list[tuple[int, str]],
    core_lines: list[tuple[int, str]],
    chunk_index: int,
    chunk_count: int,
    chunk_max_tokens: int,
) -> tuple[list[dict], set[int]]:
    """用两次 LLM 调用把目录物理文本恢复成逻辑标题并复核完整性。"""
    valid_lines = set(range(context_lines[0][0], context_lines[-1][0] + 1))
    core_line_numbers = {line_no for line_no, _ in core_lines}
    context_text = _format_toc_lines(context_lines)
    core_start = core_lines[0][0]
    core_end = core_lines[-1][0]
    expected = (
        '{"headings": [{"text": "原子标题候选", "source_lines": [109], '
        '"relation": "independent|continuation_of_previous|child_of_previous|sibling_of_previous", '
        '"confidence": 0.95, "merged_text": ""}], '
        '"ignored_lines": [73], "carried_lines": []}'
    )

    recovery_prompt = f"""禁止分析过程，只返回严格 JSON。

文件名: {file_name}
当前目录恢复块: {chunk_index}/{chunk_count}
当前核心原文行: {core_start}-{core_end}

任务：把目录区域中的物理文本恢复成完整、有序的原子标题候选。

你必须综合使用语义、页码、换行、排版符号和上下文自行判断：
1. “逻辑标题”是目录里的每一个独立条目，不是整章内容块。章标题、1.1/2.3 等编号小节、
   无编号子标题都必须分别输出；绝不能把一个章标题及其后续小节折叠成一个标题。
2. source_lines 只能包含物理上构成当前标题文字的行，不能包含它的子标题、兄弟标题或后代条目的行。
3. 一条物理行可能包含多个逻辑标题，必须完整拆出，并允许多个标题共享同一个 source_line。
4. 一个标题可能跨越多条物理行。只有后续片段明确无法独立成题、并且直接补全前一标题时才拼接；
   不能把语义完整的子主题、操作或步骤拼入父标题。
5. 数字可能是页码，也可能是标题编号；必须保留标题开头的结构编号，例如 `1`、`1.1`、`2.3.4`、
   `Chapter 2`、`第3章`。只删除标题末尾表示页数的数字，不得删除结构编号。
6. 标题 text 必须去掉末尾页码、点线和用于分隔标题的排版符号，只保留完整标题文字。
7. 多个语义完整、彼此并列的短语必须拆成多个原子标题候选。若无法确定是续行还是独立标题，
   默认分别保留，后续关系判断会决定是否合并。
8. 如果同一区域先出现简略目录、随后出现更详细目录，必须只恢复后面的详细目录条目；
   简略目录容器及其较早的重复摘要行全部放入 ignored_lines，不能同时输出两套目录。
   同一章标题在两套目录中重复时，必须保留后面详细目录中的那一次，不能因为它与前面重复就把详细目录章根删除。
9. 只输出“最早 source_line 位于核心原文行 {core_start}-{core_end}”的逻辑标题。
10. 核心区内确定不是目录标题内容的行放入 ignored_lines。
11. 核心区内属于前一块已开始标题的续行放入 carried_lines。
12. 核心区每一条输入行必须出现在某个 source_lines、ignored_lines 或 carried_lines 中。
13. 不要分配标题等级，不要创造原文中不存在的标题。
14. 每个标题必须返回 relation、confidence 和 merged_text：
    - independent：新的独立标题；
    - continuation_of_previous：当前片段本身不是完整标题，只是上一标题被换行截断后的续写；
    - child_of_previous：当前标题是上一标题的独立子主题；
    - sibling_of_previous：当前标题与上一标题并列。
15. 只有明确 continuation_of_previous 才填写合并后的 merged_text；其他关系 merged_text 必须为空。
16. 不确定时保留独立标题，使用 independent、child_of_previous 或 sibling_of_previous，不能提前合并。
17. 标题开头已有的结构编号必须逐字保留，例如 `5.2`、`1.2.3`、`Chapter 4`、`第5章`；
    不得因为判断了父子关系就删除编号。
18. 目录原文中的每个实义文字片段必须出现在某个标题 text 或 merged_text 中。若无法确定短片段归属，
    将它作为独立标题保留；不能因为同一物理行已被其他标题引用就把剩余文字静默删除。
通用版式示例：
原文：
行10: Data processing 4 | Working with command-line
行12: tools 5 | Managing files 9
应恢复为三个独立标题：
- `Data processing`，source_lines=[10]
- `Working with command-line tools`，source_lines=[10,12]
- `Managing files`，source_lines=[12]
不能把 `tools` 单独保留为标题，也不能遗漏行12后半段的新标题。

带前后重叠上下文的目录原文：
{context_text}

返回格式：
{expected}"""
    recovery_result = call_llm_json_with_repair(
        [
            {
                "role": "system",
                "content": (
                    "你是目录逻辑标题恢复器。根据完整语义和版式恢复标题，"
                    "禁止分析过程，只能输出严格 JSON。/no_think"
                ),
            },
            {"role": "user", "content": "/no_think\n" + recovery_prompt},
        ],
        max_tokens=chunk_max_tokens,
        stage="layer2_logical_recovery",
        expected_format=expected,
        timeout=TIMEOUT,
    )

    review_base_prompt = f"""禁止分析过程，只返回严格 JSON。

文件名: {file_name}
当前目录复核块: {chunk_index}/{chunk_count}
当前核心原文行: {core_start}-{core_end}

任务：对照目录原文，完整复核上一轮原子标题候选，并返回修正后的完整结果。

重点检查：
1. 是否把章标题及其后续编号小节、无编号子标题错误折叠成一个标题；若有，必须拆成每个独立目录条目。
2. 是否遗漏了同一物理行中的其他标题。
3. 是否遗漏了跨行标题的前半段或后半段；仅保留原子候选，不在存在歧义时提前合并。
4. 是否把标题残片错误保留为独立标题。
5. 是否错误拼接了本来独立的多个标题；多个完整操作、步骤或并列主题必须拆开保留。
6. 是否错误删除了标题开头的 `1/1.1/2.3.4/Chapter 2/第3章` 等结构编号。
7. 如果已经出现后面的详细目录，是否仍错误保留了前面的简略目录；简略目录必须放入 ignored_lines。
   对重复章标题必须保留后面详细目录中的那一次，删除较早的简略摘要，不能反向删除详细目录章根。
8. 是否创造了原文不存在的标题。
9. source_lines 是否只覆盖构成当前标题文字的物理行，不能把后代标题行算入父标题。
10. 只输出最早 source_line 位于核心区的标题。
11. 核心区每一条输入行必须出现在 source_lines、ignored_lines 或 carried_lines 中。
12. 不要分配标题等级。
13. 每个标题必须保留或修正 relation、confidence、merged_text。只有当前片段明确无法独立成题、
    并直接补全前一标题时，才使用 continuation_of_previous 并填写 merged_text。
14. 多个语义完整的主题、操作或步骤必须保持独立；不确定时不能合并。
15. 必须逐字保留标题开头已有的结构编号，不能删除 `5.2`、`1.2.3`、`Chapter 4`、`第5章`。
16. 原文中的每个实义文字片段必须保留在某个标题 text 或 merged_text 中；无法确定归属时独立保留，
    不能静默删除。
关键版式复核示例：
若上一行末尾是语义未完成的 `Working with command-line`，下一行开头是 `tools 5`，
同时该行后半段又开始另一个标题，则必须恢复 `Working with command-line tools`，
并继续单独恢复后半段新标题；不能留下孤立的 `tools`。

目录原文：
{context_text}

上一轮恢复结果：
{json.dumps(recovery_result, ensure_ascii=False)}

返回修正后的完整 JSON：
{expected}"""
    correction = ""
    last_error = None
    reviewed_result = {}
    for review_attempt in range(3):
        reviewed_result = call_llm_json_with_repair(
            [
                {
                    "role": "system",
                    "content": (
                        "你是目录逻辑标题完整性审核器。必须对照原文修复遗漏、误拆和误拼，"
                        "禁止分析过程，只能输出严格 JSON。/no_think"
                    ),
                },
                {
                    "role": "user",
                    "content": "/no_think\n" + review_base_prompt + correction,
                },
            ],
            max_tokens=chunk_max_tokens,
            stage="layer2_logical_review",
            expected_format=expected,
            timeout=TIMEOUT,
        )
        try:
            headings, ignored_lines, _ = _validate_layer2_logical_recovery_result(
                reviewed_result,
                valid_lines,
                core_line_numbers,
                f"第二层逻辑复核块 {chunk_index}/{chunk_count}",
            )
            return _apply_layer2_inline_relations(headings), ignored_lines
        except SkipMarkdownFile as exc:
            last_error = exc
            if review_attempt >= 2:
                break
            print(
                f"    [layer2_logical_review] 完整性校验失败 "
                f"{review_attempt + 1}/3，要求 LLM 按行号修复"
            )
            correction = f"""

上一轮复核结果仍不完整或行号非法：
{exc.message}

请重新对照原文返回完整结果。必须修复上述问题，不得遗漏任何核心原文行。
上一轮复核 JSON：
{json.dumps(reviewed_result, ensure_ascii=False)}
"""

    if not reviewed_result:
        raise last_error or SkipMarkdownFile(
            "layer2_logical_recovery_invalid",
            "layer2_logical_recovery",
            f"第二层逻辑复核块 {chunk_index}/{chunk_count} 未通过完整性校验",
        )
    repaired_result = _resolve_layer2_missing_coverage_by_llm(
        reviewed_result,
        context_lines,
        core_lines,
        file_name,
        chunk_index,
        chunk_count,
        chunk_max_tokens,
    )
    headings, ignored_lines, _ = _validate_layer2_logical_recovery_result(
        repaired_result,
        valid_lines,
        core_line_numbers,
        f"第二层逻辑覆盖修复块 {chunk_index}/{chunk_count}",
        require_complete=False,
    )
    return _apply_layer2_inline_relations(headings), ignored_lines


def _call_layer2_title_relations_by_llm(
    file_name: str,
    context_lines: list[tuple[int, str]],
    candidate_headings: list[dict],
    core_lines: set[int],
    chunk_index: int,
    chunk_count: int,
    chunk_max_tokens: int,
) -> list[dict]:
    """让 LLM 拆分原子标题并判断相邻关系，仅合并明确的续行片段。"""
    if not candidate_headings:
        return []

    valid_lines = {line_no for line_no, _ in context_lines}
    expected = """{
  "items": [
    {
      "text": "完整或原子的标题片段",
      "source_lines": [164],
      "relation": "independent|continuation_of_previous|child_of_previous|sibling_of_previous|non_title",
      "confidence": 0.95,
      "merged_text": ""
    }
  ]
}"""
    prompt = f"""禁止分析过程，只返回严格 JSON。

文件名: {file_name}
当前标题关系复核块: {chunk_index}/{chunk_count}

任务：对照目录原文，先把候选恢复为最小且语义完整的目录标题，再判断相邻标题之间的语义关系。

关系定义：
- independent：新的独立标题。
- continuation_of_previous：当前文字本身不是完整标题，只是上一标题被换行截断后的续写。
- child_of_previous：当前标题是上一标题展开出的子主题。
- sibling_of_previous：当前标题与上一标题并列。
- non_title：确定不是目录标题文字。

必须遵守：
1. 每个语义完整的主题、操作、步骤或栏目必须单独输出，不能折叠进父标题。
2. 如果当前候选把多个语义完整、彼此并列的主题合成了长标题，必须拆成多个 items。
3. 只有当前片段单独无法构成自然标题，并且确实直接补全上一标题时，才标记 continuation_of_previous。
4. 多个句式相似、语义完整的短语通常是独立子标题或兄弟标题，不能标记为续行。
5. 不确定时必须保留为 independent、child_of_previous 或 sibling_of_previous，不能合并或删除。
6. relation 描述当前 item 与紧邻的前一个输出 item 的关系。第一项必须是 independent。
7. continuation_of_previous 时，merged_text 必须给出两项合并后的完整标题；其他关系的 merged_text 为空字符串。
8. source_lines 只能使用目录原文行号；同一物理行可以产生多个独立 items。
9. 页码、换行和排版符号只能作为辅助证据，最终以标题语义是否完整、是否表达独立主题为准。
10. 不要分配标题等级，不要依据正文是否存在对应标题做判断。

语义示例：
- 一个主题后连续出现多个完整的操作短语，这些操作短语应作为独立子标题保留。
- 上一标题以不完整短语结束，下一项只有一个能够直接补全它的短片段，才属于续行。

目录原文：
{_format_toc_lines(context_lines)}

当前候选标题：
{json.dumps({"headings": candidate_headings}, ensure_ascii=False)}

返回格式：
{expected}"""
    try:
        result = call_llm_json_with_repair(
            [
                {
                    "role": "system",
                    "content": (
                        "你是目录标题语义关系判断器。先拆开独立主题，"
                        "只有明确续行才允许合并。禁止分析过程，只能输出严格 JSON。/no_think"
                    ),
                },
                {"role": "user", "content": "/no_think\n" + prompt},
            ],
            max_tokens=chunk_max_tokens,
            stage="layer2_title_relations",
            expected_format=expected,
            timeout=TIMEOUT,
        )
    except SkipMarkdownFile as exc:
        print(
            f"    [layer2_title_relations] 关系判断失败，保留原候选: "
            f"{exc.message[:140]}"
        )
        return [
            dict(item)
            for item in candidate_headings
            if item.get("source_lines") and min(item["source_lines"]) in core_lines
        ]

    raw_items = result.get("items", [])
    if not isinstance(raw_items, list):
        raw_items = []
    parsed: list[dict] = []
    allowed_relations = {
        "independent",
        "continuation_of_previous",
        "child_of_previous",
        "sibling_of_previous",
        "non_title",
    }
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        text = str(raw_item.get("text") or "").strip()
        try:
            source_lines = sorted(
                set(int(x) for x in raw_item.get("source_lines", []))
            )
        except Exception:
            continue
        source_lines = [line for line in source_lines if line in valid_lines]
        if not text or not source_lines:
            continue
        relation = str(raw_item.get("relation") or "independent").strip().lower()
        if relation not in allowed_relations:
            relation = "independent"
        try:
            confidence = float(raw_item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        parsed.append({
            "text": text,
            "source_lines": source_lines,
            "relation": relation,
            "confidence": max(0.0, min(confidence, 1.0)),
            "merged_text": str(raw_item.get("merged_text") or "").strip(),
        })

    minimum_count = max(1, int(len(candidate_headings) * 0.70))
    if len(parsed) < minimum_count:
        print(
            "    [layer2_title_relations] 返回标题过少，保留原候选: "
            f"{len(candidate_headings)} → {len(parsed)}"
        )
        return [
            dict(item)
            for item in candidate_headings
            if item.get("source_lines") and min(item["source_lines"]) in core_lines
        ]

    resolved: list[dict] = []
    merged_count = 0
    split_delta = max(0, len(parsed) - len(candidate_headings))
    for item in parsed:
        relation = item["relation"]
        confidence = item["confidence"]
        if relation == "non_title" and confidence >= 0.90:
            continue
        if (
            relation == "continuation_of_previous"
            and confidence >= 0.85
            and resolved
            and item["merged_text"]
        ):
            previous = resolved[-1]
            previous["text"] = item["merged_text"]
            previous["source_lines"] = sorted(
                set(previous["source_lines"]) | set(item["source_lines"])
            )
            previous["_layer2_relation"] = "continuation_merged"
            merged_count += 1
            continue
        resolved.append({
            "text": item["text"],
            "source_lines": item["source_lines"],
            "_layer2_relation": relation,
            "_layer2_relation_confidence": confidence,
        })

    owned = [
        item for item in resolved
        if item.get("source_lines") and min(item["source_lines"]) in core_lines
    ]
    original_owned = [
        dict(item)
        for item in candidate_headings
        if item.get("source_lines") and min(item["source_lines"]) in core_lines
    ]
    minimum_owned = max(0, len(original_owned) - merged_count)
    if len(owned) < minimum_owned:
        print(
            f"    [layer2_title_relations] 关系复核丢失标题，保留原候选: "
            f"原有 {len(original_owned)}, 合并 {merged_count}, 返回 {len(owned)}"
        )
        return original_owned
    print(
        f"    第二层语义关系 {chunk_index}/{chunk_count}: "
        f"拆出 +{split_delta}, 明确续行合并 {merged_count}, 保留 {len(owned)} 个"
    )
    return owned


def _refine_layer2_title_relations_by_llm(
    cleaned_lines: list[tuple[int, str]],
    logical_titles: list[dict],
    file_name: str,
    chunk_max_tokens: int,
    core_size: int = 24,
    context_size: int = 8,
) -> list[dict]:
    """分窗口复核标题语义关系，避免把完整子标题合并成长标题。"""
    if not cleaned_lines or not logical_titles:
        return logical_titles

    core_size = max(12, core_size)
    context_size = max(1, context_size)
    chunk_count = (len(cleaned_lines) + core_size - 1) // core_size
    refined: list[dict] = []
    for chunk_index, start in enumerate(range(0, len(cleaned_lines), core_size), 1):
        end = min(len(cleaned_lines), start + core_size)
        context_start = max(0, start - context_size)
        context_end = min(len(cleaned_lines), end + context_size)
        core = cleaned_lines[start:end]
        context = cleaned_lines[context_start:context_end]
        core_numbers = {line_no for line_no, _ in core}
        context_numbers = {line_no for line_no, _ in context}
        candidates = [
            item for item in logical_titles
            if set(item.get("source_lines", [])) & context_numbers
        ]
        refined.extend(
            _call_layer2_title_relations_by_llm(
                file_name,
                context,
                candidates,
                core_numbers,
                chunk_index,
                chunk_count,
                chunk_max_tokens,
            )
        )

    deduped: list[dict] = []
    seen = set()
    for item in refined:
        key = (tuple(item.get("source_lines", [])), title_key(item.get("text", "")))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: (min(item["source_lines"]), max(item["source_lines"])))

    minimum_count = max(1, int(len(logical_titles) * 0.75))
    if len(deduped) < minimum_count:
        print(
            "  第二层语义关系复核结果过少，保留复核前标题: "
            f"{len(logical_titles)} → {len(deduped)}"
        )
        return logical_titles
    print(
        f"  第二层语义关系复核: {len(logical_titles)} → {len(deduped)} 个标题"
    )
    return deduped


def _recover_layer2_logical_titles_by_llm(
    cleaned_lines: list[tuple[int, str]],
    file_name: str,
    chunk_max_tokens: int,
    core_size: int = LAYER2_LOGICAL_CORE_LINES,
    context_size: int = LAYER2_LOGICAL_CONTEXT_LINES,
) -> list[dict]:
    """第二层 A/B：统一恢复所有逻辑标题，再由 LLM 复核完整性。"""
    if not cleaned_lines:
        return []

    core_size = max(20, core_size)
    context_size = max(1, context_size)
    chunk_count = (len(cleaned_lines) + core_size - 1) // core_size
    recovered: list[dict] = []
    ignored_lines: set[int] = set()

    for chunk_index, start in enumerate(range(0, len(cleaned_lines), core_size), 1):
        end = min(len(cleaned_lines), start + core_size)
        context_start = max(0, start - context_size)
        context_end = min(len(cleaned_lines), end + context_size)
        core = cleaned_lines[start:end]
        context = cleaned_lines[context_start:context_end]
        batch_headings, batch_ignored = _call_layer2_logical_recovery_llm(
            file_name,
            context,
            core,
            chunk_index,
            chunk_count,
            chunk_max_tokens,
        )
        recovered.extend(batch_headings)
        ignored_lines.update(batch_ignored)
        print(
            f"  第二层逻辑恢复 {chunk_index}/{chunk_count}: "
            f"原文 {len(core)} 行 → {len(batch_headings)} 个逻辑标题"
        )

    # 重叠窗口可能让 LLM 重复返回同一逻辑标题，按来源行和文本确定性去重。
    deduped = []
    seen = set()
    for item in recovered:
        key = (tuple(item["source_lines"]), title_key(item["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    locally_covered_lines = {
        line
        for item in deduped
        for line in item.get("source_lines", [])
    }
    confirmed_ignored_lines = ignored_lines - locally_covered_lines
    deduped = _review_layer2_logical_titles_globally_by_llm(
        cleaned_lines,
        deduped,
        file_name,
        chunk_max_tokens,
        confirmed_ignored_lines,
    )
    all_lines = {line_no for line_no, _ in cleaned_lines}
    covered_lines = {
        line
        for item in deduped
        for line in item.get("source_lines", [])
    } | ignored_lines
    missing = sorted(all_lines - covered_lines)
    if missing:
        print(
            "  第二层逻辑恢复覆盖警告: "
            f"{len(missing)} 条目录原文未获得精确 source_lines 归属: {missing[:20]}"
        )

    deduped.sort(key=lambda item: (min(item["source_lines"]), max(item["source_lines"])))
    print(
        f"  第二层逻辑恢复完成: {len(cleaned_lines)} 条物理行 → "
        f"{len(deduped)} 个逻辑标题, 忽略 {len(ignored_lines)} 行"
    )
    return deduped


def _review_layer2_logical_titles_globally_by_llm(
    cleaned_lines: list[tuple[int, str]],
    logical_titles: list[dict],
    file_name: str,
    chunk_max_tokens: int,
    confirmed_ignored_lines: set[int] | None = None,
) -> list[dict]:
    """第二层 B 全局复核：统一修正跨块误拆、遗漏和简略目录重复。"""
    if not logical_titles:
        return logical_titles

    confirmed_ignored_lines = set(confirmed_ignored_lines or set())
    expected = (
        '{"headings": [{"text": "原子标题候选", "source_lines": [109], '
        '"relation": "independent|continuation_of_previous|child_of_previous|sibling_of_previous", '
        '"confidence": 0.95, "merged_text": ""}], '
        '"ignored_lines": [73], "carried_lines": []}'
    )
    prompt = f"""禁止分析过程，只返回严格 JSON。

文件名: {file_name}

任务：对照完整目录原文，对分块恢复后的原子标题候选列表做最终全局复核。

必须完成：
1. 每个独立目录条目单独输出，不能把父标题及其子标题折叠。
2. 修复跨物理行和跨分块的标题遗漏，但存在歧义时优先保留原子候选；
   不能把语义完整的子主题、操作、步骤或并列标题提前合并。
3. 一条物理行包含多个标题时必须全部拆出。
4. 保留标题开头的结构编号，例如 `1`、`1.1`、`2.3.4`、`Chapter 2`、`第3章`；
   删除末尾页码、点线和排版分隔符。
5. 如果先出现简略目录、后出现详细目录，只保留后面的详细目录；必须保留详细目录中的章根标题，
   删除较早的简略摘要及目录容器。
6. source_lines 只包含构成当前标题文字的原文行；允许多个标题共享同一物理行。
7. 多个语义完整、彼此并列的短语必须拆开；不确定是否为续行时默认分别保留。
8. 不要分配等级，不要创造原文不存在的标题。
9. 每个标题必须返回 relation、confidence、merged_text。只有当前片段明确无法独立成题并直接补全前一标题时，
   才使用 continuation_of_previous 并填写 merged_text；其他关系 merged_text 为空。
10. 必须逐字保留标题开头已有的结构编号；原文中的每个实义文字片段都必须保留在某个标题 text 或
    merged_text 中，无法确定归属时独立保留，不能静默删除。
关键版式示例：
`Working with command-line` 与下一行开头的 `tools 5` 应合并为
`Working with command-line tools`；如果 `tools 5` 后面同一行又出现新标题，
还必须把新标题单独恢复，不能遗漏。

完整目录原文：
{_format_toc_lines(cleaned_lines)}

分块恢复结果：
{json.dumps({"headings": logical_titles}, ensure_ascii=False)}

局部恢复已确认忽略、且未被任何标题使用的原文行：
{sorted(confirmed_ignored_lines)}
这些行不能在全局复核中重新生成标题。

返回最终完整结果：
{expected}"""
    try:
        global_max_tokens = max(
            chunk_max_tokens,
            min(24000, max(12000, len(logical_titles) * 100)),
        )
        result = call_llm_json_with_repair(
            [
                {
                    "role": "system",
                    "content": (
                        "你是全书目录逻辑标题总审核器。必须统一修复跨块误拆、遗漏和重复，"
                        "禁止分析过程，只能输出严格 JSON。/no_think"
                    ),
                },
                {"role": "user", "content": "/no_think\n" + prompt},
            ],
            max_tokens=global_max_tokens,
            stage="layer2_logical_global_review",
            expected_format=expected,
            timeout=TIMEOUT,
        )
        valid_lines = set(range(cleaned_lines[0][0], cleaned_lines[-1][0] + 1))
        core_lines = {line_no for line_no, _ in cleaned_lines}
        headings, _, _ = _validate_layer2_logical_recovery_result(
            result,
            valid_lines,
            core_lines,
            "第二层全目录逻辑复核",
            require_complete=False,
        )
        if confirmed_ignored_lines:
            before_filter = len(headings)
            headings = [
                item for item in headings
                if not set(item.get("source_lines", [])) <= confirmed_ignored_lines
            ]
            removed = before_filter - len(headings)
            if removed:
                print(f"  第二层全目录逻辑复核: 移除 {removed} 个由已忽略原文行复活的标题")
        minimum_count = max(1, int(len(logical_titles) * 0.75))
        if len(headings) < minimum_count:
            raise SkipMarkdownFile(
                "layer2_logical_global_review_incomplete",
                "layer2_logical_global_review",
                (
                    f"全目录复核标题数量下降过多: "
                    f"{len(logical_titles)} -> {len(headings)}, 最少应保留 {minimum_count}"
                ),
            )
        if headings:
            print(
                f"  第二层全目录逻辑复核: "
                f"{len(logical_titles)} → {len(headings)} 个逻辑标题"
            )
            return _apply_layer2_inline_relations(headings)
    except SkipMarkdownFile as exc:
        print(f"  第二层全目录逻辑复核失败，保留分块复核结果: {exc.message[:160]}")
    return logical_titles


def _assign_layer2_initial_levels_by_llm(
    logical_titles: list[dict],
    file_name: str,
    chunk_max_tokens: int,
    chunk_size: int = LAYER2_LEVEL_CHUNK_TITLES,
) -> list[dict]:
    """第二层 C：只给已确认的逻辑标题分配初始等级，不允许改写标题。"""
    if not logical_titles:
        return []

    levels: dict[int, int] = {}
    chunk_size = max(20, chunk_size)
    chunk_count = (len(logical_titles) + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, len(logical_titles), chunk_size), 1):
        end = min(len(logical_titles), start + chunk_size)
        context_start = max(0, start - 10)
        context_end = min(len(logical_titles), end + 10)
        lines = []
        for idx in range(context_start, context_end):
            role = "CORE" if start <= idx < end else "CONTEXT"
            lines.append(f'{role} index={idx + 1}: "{logical_titles[idx]["text"]}"')
        expected = '{"items": [{"index": 1, "level": 2}]}'
        prompt = f"""禁止分析过程，只返回严格 JSON。

文件名: {file_name}
当前初始等级块: {chunk_index}/{chunk_count}

任务：仅为 CORE 逻辑标题分配初始 Markdown 等级。
标题已经完成恢复，禁止改写、合并、拆分、遗漏或创造标题。
第三层稍后会根据完整语义重建 parent tree，因此这里给出稳定的初始等级即可。

规则：
1. 必须为每个 CORE index 恰好返回一项，不能返回 CONTEXT index。
2. level 只能是 2-6。
3. 章标题通常为 H2；1.1/2.3 通常为 H3；1.1.1/2.3.4 通常为 H4。
4. 无编号子标题根据语义给出合理初始等级。
5. 不要返回标题文本，只返回 index 和 level。

逻辑标题：
{chr(10).join(lines)}

返回格式：
{expected}"""
        result = call_llm_json_with_repair(
            [
                {
                    "role": "system",
                    "content": (
                        "你是目录初始等级分配器。不能改变标题列表，"
                        "禁止分析过程，只能输出严格 JSON。/no_think"
                    ),
                },
                {"role": "user", "content": "/no_think\n" + prompt},
            ],
            max_tokens=max(1000, min(chunk_max_tokens, 400 + (end - start) * 40)),
            stage="layer2_initial_levels",
            expected_format=expected,
            timeout=TIMEOUT,
        )
        items = result.get("items", [])
        if not isinstance(items, list):
            raise SkipMarkdownFile(
                "layer2_initial_levels_invalid",
                "layer2_initial_levels",
                f"第二层初始等级块 {chunk_index}/{chunk_count}: items 不是列表",
            )
        expected_indices = set(range(start + 1, end + 1))
        batch_levels: dict[int, int] = {}
        for item in items:
            try:
                index = int(item.get("index"))
                level = int(item.get("level"))
            except Exception:
                continue
            if index in expected_indices and 2 <= level <= 6 and index not in batch_levels:
                batch_levels[index] = level
        if set(batch_levels) != expected_indices:
            missing = sorted(expected_indices - set(batch_levels))
            raise SkipMarkdownFile(
                "layer2_initial_levels_invalid",
                "layer2_initial_levels",
                f"第二层初始等级块 {chunk_index}/{chunk_count} 缺少标题: {missing[:20]}",
            )
        levels.update(batch_levels)
        print(
            f"  第二层初始等级 {chunk_index}/{chunk_count}: "
            f"{end - start} 个逻辑标题"
        )

    return [
        {
            "text": item["text"],
            "level": levels[index],
            "source_lines": item["source_lines"],
            "_logical_recovered": True,
        }
        for index, item in enumerate(logical_titles, 1)
    ]


def _layer2_parse_toc_legacy(
    md_text: str,
    toc_range: dict,
    file_name: str,
    single_max_lines: int = TOC_PARSE_SINGLE_MAX_LINES,
    chunk_lines: int = TOC_PARSE_CHUNK_LINES,
    chunk_max_tokens: int = TOC_PARSE_CHUNK_MAX_TOKENS,
    cache_input_path: str | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """第二层: LLM 解析目录区域，输出全文标题及其等级。

    短目录一次发送；长目录按清洗后的目录行数分块发送，避免超上下文或输出过长。

    分块原则:
        每个 batch 只解析自己的目录行，并返回同一 JSON 格式。程序按顺序合并。
        batch 缓存包含分块参数、行号范围和内容 hash，所以中途失败后重跑时只补失败批次。

    Args:
        md_text: 原始 Markdown 文本。
        toc_range: 第一层确认后的目录范围。
        file_name: 文件名，仅用于 prompt。
        single_max_lines: 清洗后目录行数不超过该值时单批解析。
        chunk_lines: 长目录每批最多解析的清洗后目录行数。
        chunk_max_tokens: 每批 LLM 输出 token 上限。
        cache_input_path: 用于生成 batch 缓存路径的输入文件路径。
        use_cache: 是否启用第二层总缓存和 batch 缓存。

    Returns:
        目录标题列表，元素形如 {"text": "...", "level": 2}。
    """
    cleaned_lines, start_line, end_line = _clean_toc_lines(md_text, toc_range)
    toc_content = _format_toc_lines(cleaned_lines)
    print(f"  目录内容: {len(toc_content)} 字符 (清洗前{end_line-start_line+1}行, 清洗后{len(cleaned_lines)}行)")

    # 全局扫描：整个 TOC 有没有页码 — 决定情况 A/B
    _has_any_page = any(_has_toc_page_marker(ln) for _, ln in cleaned_lines)
    _page_hint = (
        "【全局提示】本目录包含页码。优先提取带页码的行；"
        "但如果局部页码 OCR 丢失，连续编号/层级标题也必须提取，"
        "不能从 1.5、2.3 这类中途编号才开始。"
        if _has_any_page else
        "【全局提示】本目录没有页码，按标题密度判断："
        "标题后紧跟另一个标题 → 目录条目，提取。"
        "标题后紧跟大段正文段落 → 前辅文/非目录，跳过。"
    )

    single_max_lines = max(1, single_max_lines)
    chunk_lines = max(1, chunk_lines)
    chunk_max_tokens = max(1000, chunk_max_tokens)

    if len(cleaned_lines) <= single_max_lines:
        user_prompt = TOC_PARSE_USER.format(
            file_name=file_name,
            toc_start=start_line,
            toc_end=end_line,
            toc_content=toc_content,
            page_hint=_page_hint,
        )
        messages = [
            {"role": "system", "content": TOC_PARSE_SYSTEM},
            {"role": "user", "content": "/no_think\n" + user_prompt},
        ]
        dyn_max_tokens = min(chunk_max_tokens, max(1000, len(cleaned_lines) * 60))
        print(f"  第二层: 单批解析 {len(cleaned_lines)} 行, max_tokens={dyn_max_tokens}")
        try:
            headings = _call_toc_parse_llm(messages, dyn_max_tokens, "LLM解析")
            headings = _repair_layer2_missing_prefix_by_llm(
                headings,
                cleaned_lines,
                file_name,
                chunk_max_tokens,
            )
            return headings
        except RuntimeError as e:
            fallback_chunk_lines = min(chunk_lines, 80)
            if len(cleaned_lines) <= fallback_chunk_lines:
                fallback_chunk_lines = max(1, (len(cleaned_lines) + 1) // 2)
            print(
                "  第二层单批解析失败，自动降级分块: "
                f"{str(e)[:180]}；fallback_chunk_lines={fallback_chunk_lines}"
            )
            chunk_lines = fallback_chunk_lines

    chunks = [
        cleaned_lines[i:i + chunk_lines]
        for i in range(0, len(cleaned_lines), chunk_lines)
    ]
    print(
        f"  第二层: 分块解析 {len(cleaned_lines)} 行 → {len(chunks)} 批 "
        f"(每批最多{chunk_lines}行, max_tokens上限{chunk_max_tokens})"
    )

    all_headings = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_content = _format_toc_lines(chunk)
        chunk_start = chunk[0][0]
        chunk_end = chunk[-1][0]
        batch_cache = None
        if use_cache and cache_input_path:
            chunk_digest = hashlib.sha1(chunk_content.encode("utf-8")).hexdigest()[:10]
            batch_stage = (
                f"l2b.s{single_max_lines}.l{chunk_lines}.t{chunk_max_tokens}."
                f"b{index:03d}of{len(chunks):03d}.{chunk_start}-{chunk_end}.{chunk_digest}"
            )
            batch_cache = _cache_key(cache_input_path, batch_stage)
            if cached := _load_cache(batch_cache):
                cached_headings = cached.get("headings", [])
                print(
                    f"  第二层 batch {index}/{len(chunks)}: "
                    f"[缓存] {len(cached_headings)} 个标题, 原文行{chunk_start}-{chunk_end}"
                )
                all_headings.extend(cached_headings)
                continue

        user_prompt = TOC_PARSE_CHUNK_USER.format(
            file_name=file_name,
            chunk_index=index,
            chunk_count=len(chunks),
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            toc_content=chunk_content,
            page_hint=_page_hint,
        )
        messages = [
            {"role": "system", "content": TOC_PARSE_SYSTEM},
            {"role": "user", "content": "/no_think\n" + user_prompt},
        ]
        dyn_max_tokens = chunk_max_tokens
        prompt_tokens = count_tokens(TOC_PARSE_SYSTEM) + count_tokens(user_prompt)
        print(
            f"  第二层 batch {index}/{len(chunks)}: "
            f"{len(chunk)}行, 原文行{chunk_start}-{chunk_end}, "
            f"prompt≈{prompt_tokens} tokens, max_tokens={dyn_max_tokens}"
        )
        batch_headings = _call_toc_parse_llm(
            messages,
            dyn_max_tokens,
            f"batch {index}/{len(chunks)}",
        )
        if batch_cache:
            _save_cache(batch_cache, {
                "chunk_index": index,
                "chunk_count": len(chunks),
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk_line_count": len(chunk),
                "headings": batch_headings,
            })
        all_headings.extend(batch_headings)

    all_headings = _dedupe_adjacent_headings(all_headings)
    all_headings = _repair_layer2_missing_prefix_by_llm(
        all_headings,
        cleaned_lines,
        file_name,
        chunk_max_tokens,
    )
    print(f"  第二层合并: {len(all_headings)} 个目录标题")
    return all_headings


def layer2_parse_toc(
    md_text: str,
    toc_range: dict,
    file_name: str,
    single_max_lines: int = TOC_PARSE_SINGLE_MAX_LINES,
    chunk_lines: int = TOC_PARSE_CHUNK_LINES,
    chunk_max_tokens: int = TOC_PARSE_CHUNK_MAX_TOKENS,
    cache_input_path: str | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """第二层：快速解析目录，仅对少量疑似续行组做保守语义修复。"""
    return _layer2_parse_toc_legacy(
        md_text,
        toc_range,
        file_name,
        single_max_lines=single_max_lines,
        chunk_lines=chunk_lines,
        chunk_max_tokens=chunk_max_tokens,
        cache_input_path=cache_input_path,
        use_cache=use_cache,
    )


# ============================================================
# 第三层: 规则校验标题等级
# ============================================================


def _extract_number_prefix(text: str) -> str:
    """从标题中提取编号前缀的"模式", 用于分组同类标题。

    作用:
        第三层需要知道哪些标题属于同一结构模式，例如所有"第1章/第2章"
        都归一化为 "第N章"，所有 "1.1/2.3" 都归一化为 "N.N"。

    输入:
        text: str，目录标题文本。

    输出:
        str，结构模式。

    返回的是模式的抽象表示而非原文:
    - "第1章 xxx" → "第N章"
    - "第 2 章 xxx" → "第N章"
    - "第一章 xxx" → "第N章"
    - "Chapter 3 xxx" → "Chapter N"
    - "1.1 xxx" → "N.N"
    - "2.3.4 xxx" → "N.N.N"
    - "Part I xxx" → "Part N"
    - "附录A xxx" → "附录N"
    - "小结" → "小结"  (无编号则保留原文本)

    例子:
        _extract_number_prefix("2.3.4 中断处理") -> "N.N.N"
    """
    # 统一空白: 全角空格、换行、多空格 → 压缩为单空格
    text = clean_title(re.sub(r"[\s　]+", " ", text.strip()))
    # 中文序数章节: 第X章 / 第X部分 / 第X节
    m = re.match(r"^(第\s*[一二三四五六七八九十百零0-9]+\s*[章节篇卷部])", text)
    if m: return re.sub(r"\s+", "", re.sub(r"[一二三四五六七八九十百零0-9]+", "N", m.group(1)))
    # Chapter X / Part X
    m = re.match(r"^(Chapter|Part)\s+[IVX\d]+", text, re.IGNORECASE)
    if m: return f"{m.group(1)} N"
    # 附录X / Appendix X
    m = re.match(r"^(附录|Appendix)\s*[A-Za-z0-9]", text, re.IGNORECASE)
    if m: return f"{m.group(1)} N"
    # 十进制编号: 1.1 / 2.3.4
    m = re.match(r"^(\d+(?:\.\d+)+)", text)
    if m: return re.sub(r"\d+", "N", m.group(1))
    # 纯数字: 1. / 2. (单级数字)
    m = re.match(r"^(\d+)\.?\s", text)
    if m: return "N."
    # 无编号标题: 用原文本作为模式 (同类文本会自然聚合)
    return text


def _compute_decimal_depth(text: str) -> int:
    """计算十进制编号标题的层级深度。

    输入:
        text: str，标题文本。

    输出:
        int:
            0 表示无十进制编号；
            1 表示 "1." 或 "1 标题"；
            2 表示 "1.1"；
            3 表示 "2.3.4"。

    例子:
        _compute_decimal_depth("1.2 GPIO") -> 2
        _compute_decimal_depth("小结") -> 0
    """
    m = re.match(r"^(\d+(?:\.\d+)+)", clean_title(text))
    if m:
        return m.group(1).count(".") + 1
    m = re.match(r"^(\d+)\.?\s", clean_title(text))
    if m:
        return 1
    return 0


def _expected_decimal_level_from_numbered_ancestor(
    toc_headings: list[dict],
    index: int,
) -> int | None:
    """按编号路径寻找十进制标题的真实父级，不受无编号标题干扰。

    例如 `2.3.4` 依次寻找 `2.3`、`2`；找到父级后，按缺失的编号深度
    推导当前标题的最低等级。找不到编号祖先时返回 None，保留 LLM 等级。
    """
    current_path = _toc_number_path(display_title(toc_headings[index].get("text", "")))
    parts = current_path.split(".") if current_path else []
    if not parts or not all(part.isdigit() for part in parts):
        return None

    # `1 Title / 2 Title` 可能是章，也可能是局部编号列表。只有位于明确
    # Part/篇/卷下面时，第三层才安全地把它视为该容器的下一层。
    if len(parts) == 1:
        for j in range(index - 1, -1, -1):
            marker = _toc_structural_marker(
                title_key(display_title(toc_headings[j].get("text", "")))
            )
            if marker is None:
                continue
            if marker[0] == "part":
                return min(int(toc_headings[j].get("level") or 2) + 1, 6)
            break
        return None

    # 优先寻找最长编号父路径：2.3.4 -> 2.3 -> 2。
    for parent_depth in range(len(parts) - 1, 0, -1):
        parent_path = ".".join(parts[:parent_depth])
        for j in range(index - 1, -1, -1):
            candidate_path = _toc_number_path(display_title(toc_headings[j].get("text", "")))
            if candidate_path != parent_path:
                continue
            parent_level = int(toc_headings[j].get("level") or 2)
            missing_depth = len(parts) - parent_depth
            return min(parent_level + missing_depth, 6)

    return None


def _extract_arabic_chapter_number(text: str) -> str:
    """提取 `第 18 章` 这类标题中的阿拉伯数字章号。"""
    m = re.match(r"^第\s*([0-9]+)\s*[章节篇卷部]", clean_title(text))
    return m.group(1) if m else ""


def _extract_decimal_root(text: str) -> str:
    """提取 `18.1` 这类十进制编号的根编号。"""
    m = re.match(r"^(\d+)(?:\.\d+)+", clean_title(text))
    return m.group(1) if m else ""


def _looks_like_toc_wrapped_continuation(parent: dict, current: dict,
                                         next_item: dict | None) -> bool:
    """判断目录项是否是上一条章节标题的换行续文。

    典型 OCR/目录解析结果:
        H2 第 18 章 虚拟化云计算平台
        H3 Proxmox VE
        H4 18.1 OpenVZ 简介

    这里的 `Proxmox VE` 应合并回上一条，而不是作为单独目录节点。
    """
    if not parent or not current or not next_item:
        return False

    parent_lv = int(parent.get("level") or 0)
    curr_lv = int(current.get("level") or 0)
    if curr_lv != parent_lv + 1:
        return False

    curr_text = clean_title(current.get("text", ""))
    if not curr_text or len(curr_text) > 80:
        return False
    if _compute_decimal_depth(curr_text) > 0:
        return False
    if re.match(r"^第\s*[一二三四五六七八九十百零0-9]+\s*[章节篇卷部]", curr_text):
        return False

    parent_no = _extract_arabic_chapter_number(parent.get("text", ""))
    next_root = _extract_decimal_root(next_item.get("text", ""))
    return bool(parent_no and parent_no == next_root)


def _merge_toc_wrapped_continuations(toc_headings: list[dict]) -> list[dict]:
    """合并目录中被换行拆开的章节标题，并修正其后子级层级。"""
    if len(toc_headings) < 3:
        return toc_headings

    merged = []
    i = 0
    while i < len(toc_headings):
        curr = dict(toc_headings[i])
        if i + 2 < len(toc_headings) and _looks_like_toc_wrapped_continuation(
            curr, toc_headings[i + 1], toc_headings[i + 2]
        ):
            cont = toc_headings[i + 1]
            curr["text"] = f"{curr.get('text', '').rstrip()} {cont.get('text', '').strip()}".strip()
            curr.setdefault("_fixed", "")
            curr["_fixed"] += "目录换行续题合并"
            parent_lv = int(curr.get("level") or 2)
            merged.append(curr)
            i += 2

            while i < len(toc_headings):
                item = dict(toc_headings[i])
                item_lv = int(item.get("level") or 0)
                if item_lv <= parent_lv:
                    break
                if item_lv > parent_lv + 1:
                    item["_orig_level"] = item.get("_orig_level", item_lv)
                    item["level"] = item_lv - 1
                    item.setdefault("_fixed", "")
                    item["_fixed"] += f"续题合并后层级→H{item['level']}"
                merged.append(item)
                i += 1
            continue

        merged.append(curr)
        i += 1
    return merged


def _is_layer3_semantic_anchor(heading: dict) -> bool:
    """编号、篇章和书尾结构属于第三层稳定骨架，不交给语义层随意改写。"""
    text = display_title(heading.get("text", ""))
    key = title_key(text)
    return bool(
        _toc_number_path(text)
        or _toc_structural_marker(key) is not None
        or _is_toc_backmatter_key(key)
    )


def _find_layer3_unnumbered_groups(toc_headings: list[dict]) -> list[dict]:
    """收集编号锚点之间连续出现的无编号标题组。"""
    groups = []
    group_id = 1
    i = 0
    while i < len(toc_headings):
        if _is_layer3_semantic_anchor(toc_headings[i]):
            i += 1
            continue

        start = i
        while i < len(toc_headings) and not _is_layer3_semantic_anchor(toc_headings[i]):
            i += 1
        end = i

        prev_anchor = start - 1 if start > 0 and _is_layer3_semantic_anchor(toc_headings[start - 1]) else None
        next_anchor = end if end < len(toc_headings) and _is_layer3_semantic_anchor(toc_headings[end]) else None
        if prev_anchor is None:
            continue

        groups.append({
            "id": group_id,
            "start": start,
            "end": end,
            "candidate_indices": list(range(start, end)),
            "prev_anchor": prev_anchor,
            "next_anchor": next_anchor,
        })
        group_id += 1
    return groups


def _format_layer3_semantic_group(group: dict, toc_headings: list[dict]) -> str:
    """格式化一个局部语义判定窗口，保留全局索引和当前等级。"""
    start = max(0, int(group["start"]) - 4)
    end = min(len(toc_headings), int(group["end"]) + 5)
    candidate_indices = set(group["candidate_indices"])
    lines = [f"group_id={group['id']}"]
    for idx in range(start, end):
        h = toc_headings[idx]
        if idx in candidate_indices:
            role = "CANDIDATE"
        elif _is_layer3_semantic_anchor(h):
            role = "ANCHOR"
        else:
            role = "CONTEXT"
        lines.append(
            f'{role} index={idx + 1} current=H{int(h.get("level") or 2)} '
            f'text="{compact_text(display_title(h.get("text", "")), 180)}"'
        )
    return "\n".join(lines)


def _propagate_numbered_sibling_level(toc_headings: list[dict],
                                       start_index: int,
                                       target_level: int) -> None:
    """父标题由语义层恢复后，把后续同根同深度编号标题保持为同级。"""
    start_path = _toc_number_path(display_title(toc_headings[start_index].get("text", "")))
    parts = start_path.split(".") if start_path else []
    if len(parts) < 2 or not all(part.isdigit() for part in parts):
        return

    root = parts[0]
    depth = len(parts)
    for idx in range(start_index, len(toc_headings)):
        text = display_title(toc_headings[idx].get("text", ""))
        key = title_key(text)
        marker = _toc_structural_marker(key)
        if idx > start_index and marker is not None:
            break

        path = _toc_number_path(text)
        if not path:
            continue
        path_parts = path.split(".")
        if not all(part.isdigit() for part in path_parts):
            break
        if path_parts[0] != root or len(path_parts) < depth:
            break
        if len(path_parts) != depth:
            continue

        old_level = int(toc_headings[idx].get("level") or 2)
        if old_level == target_level:
            continue
        toc_headings[idx]["_orig_level"] = toc_headings[idx].get("_orig_level", old_level)
        toc_headings[idx]["level"] = target_level
        toc_headings[idx].setdefault("_fixed", "")
        toc_headings[idx]["_fixed"] += f"语义父标题后同级编号→H{target_level}"


def _semantic_parent_level_for_next_anchor(toc_headings: list[dict],
                                           candidate_index: int,
                                           next_index: int) -> int:
    """把 LLM 的 parent_of_next 语义关系换算成稳定 H 等级。"""
    next_path = _toc_number_path(display_title(toc_headings[next_index].get("text", "")))
    next_parts = next_path.split(".") if next_path else []

    # 章级基准：存在 Part/篇/卷时章位于其下一层，否则章通常为 H2。
    chapter_level = 2
    for idx in range(candidate_index - 1, -1, -1):
        text = display_title(toc_headings[idx].get("text", ""))
        marker = _toc_structural_marker(title_key(text))
        if marker is None:
            continue
        if marker[0] == "part":
            chapter_level = min(int(toc_headings[idx].get("level") or 2) + 1, 5)
        break

    # next=N.1 时 candidate 是章；next=N.1.1 时 candidate 是 N.1，以此类推。
    if len(next_parts) >= 2 and all(part.isdigit() for part in next_parts):
        return min(chapter_level + len(next_parts) - 2, 5)
    return max(2, min(int(toc_headings[next_index].get("level") or 3) - 1, 5))


def _refine_cross_root_missing_parents_by_llm(toc_headings: list[dict],
                                               groups: list[dict],
                                               file_name: str) -> tuple[int, int]:
    """复核编号根切换处的混合组，识别漏掉编号的新章父标题。"""
    suspicious = []
    for group in groups:
        prev_index = group.get("prev_anchor")
        next_index = group.get("next_anchor")
        if prev_index is None or next_index is None:
            continue
        prev_path = _toc_number_path(display_title(toc_headings[int(prev_index)].get("text", "")))
        next_path = _toc_number_path(display_title(toc_headings[int(next_index)].get("text", "")))
        prev_parts = prev_path.split(".") if prev_path else []
        next_parts = next_path.split(".") if next_path else []
        if len(prev_parts) < 2 or len(next_parts) < 2:
            continue
        if not all(x.isdigit() for x in prev_parts + next_parts):
            continue
        if prev_parts[0] == next_parts[0]:
            continue
        suspicious.append(group)

    if not suspicious:
        return 0, 0

    blocks = []
    for group in suspicious:
        prev_index = int(group["prev_anchor"])
        next_index = int(group["next_anchor"])
        candidates = "\n".join(
            f'  index={idx + 1} current=H{int(toc_headings[idx].get("level") or 2)} '
            f'text="{compact_text(display_title(toc_headings[idx].get("text", "")), 180)}"'
            for idx in group["candidate_indices"]
        )
        blocks.append(
            "\n".join([
                f"group_id={group['id']}",
                (
                    f'previous_anchor index={prev_index + 1} '
                    f'H{int(toc_headings[prev_index].get("level") or 2)} '
                    f'text="{display_title(toc_headings[prev_index].get("text", ""))}"'
                ),
                "candidates:",
                candidates,
                (
                    f'next_anchor index={next_index + 1} '
                    f'H{int(toc_headings[next_index].get("level") or 2)} '
                    f'text="{display_title(toc_headings[next_index].get("text", ""))}"'
                ),
            ])
        )

    expected = (
        '{"groups": [{"id": 1, "parent_index": 12, "confidence": 0.95, '
        '"reason": "该标题语义覆盖后续新编号章节"}]}'
    )
    prompt = f"""禁止分析过程。只返回严格 JSON 对象。

文件名: {file_name}

任务：前一个编号标题和后一个编号标题的根编号发生了变化，例如从 6.6 跳到 7.1。
两者之间夹着若干无编号标题，其中可能只有一个标题是漏掉编号的“第7章”标题，
其他标题仍属于前一个 6.6。

请根据语义判断：哪个 candidate 的主题能够覆盖 next_anchor 及其后续同根小节，
因此应作为 next_anchor 的直接父标题。

规则：
1. 不能机械选择最后一个 candidate，必须比较标题语义。
2. parent_index 只能从该组 candidates 的 index 中选择。
3. 如果没有候选能够作为 next_anchor 的父标题，parent_index 返回 null。
4. 不要选择只是 previous_anchor 的方法、组成部分、案例或说明的标题。
5. 只返回 JSON，不要解释。

待复核组：
{chr(10).join(blocks)}

返回 JSON：
{expected}"""
    messages = [
        {
            "role": "system",
            "content": "你是目录缺号父标题识别器。根据语义判断新编号章节的父标题，只输出严格 JSON。/no_think",
        },
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    result = call_llm_json_with_repair(
        messages,
        max_tokens=max(800, min(4000, 400 + len(suspicious) * 180)),
        stage="layer3_cross_root_missing_parent",
        expected_format=expected,
        timeout=TIMEOUT,
    )

    by_id = {int(group["id"]): group for group in suspicious}
    changed = 0
    recovered = 0
    for item in result.get("groups", []):
        try:
            group_id = int(item.get("id"))
            confidence = float(item.get("confidence", 0))
            parent_index = int(item.get("parent_index")) - 1
        except Exception:
            continue
        group = by_id.get(group_id)
        if not group or confidence < 0.70 or parent_index not in group["candidate_indices"]:
            continue
        next_index = int(group["next_anchor"])
        level = _semantic_parent_level_for_next_anchor(toc_headings, parent_index, next_index)
        old_level = int(toc_headings[parent_index].get("level") or 2)
        if old_level != level:
            toc_headings[parent_index]["_orig_level"] = toc_headings[parent_index].get("_orig_level", old_level)
            toc_headings[parent_index]["level"] = level
            toc_headings[parent_index].setdefault("_fixed", "")
            toc_headings[parent_index]["_fixed"] += f"跨编号根语义父标题→H{level}"
            changed += 1
        _propagate_numbered_sibling_level(toc_headings, next_index, min(level + 1, 6))
        recovered += 1
    return changed, recovered


def _refine_unnumbered_levels_by_llm(toc_headings: list[dict],
                                      file_name: str = "") -> None:
    """让 LLM 根据局部语义关系判断无编号标题等级。"""
    groups = _find_layer3_unnumbered_groups(toc_headings)
    if not groups:
        return

    expected = (
        '{"groups": [{"id": 1, "items": ['
        '{"index": 4, "level": 5, '
        '"relation": "child_of_previous|sibling_of_previous|parent_of_next|independent|keep", '
        '"confidence": 0.95, "reason": "一句话"}]}]}'
    )
    batches = []
    current = []
    current_candidates = 0
    current_chars = 0
    for group in groups:
        block = _format_layer3_semantic_group(group, toc_headings)
        candidate_count = len(group["candidate_indices"])
        if current and (current_candidates + candidate_count > 45 or current_chars + len(block) > 18000):
            batches.append(current)
            current = []
            current_candidates = 0
            current_chars = 0
        current.append((group, block))
        current_candidates += candidate_count
        current_chars += len(block)
    if current:
        batches.append(current)

    changed = 0
    parent_of_next: list[tuple[int, int]] = []
    for batch_no, batch in enumerate(batches, start=1):
        blocks = "\n\n".join(block for _, block in batch)
        candidate_count = sum(len(group["candidate_indices"]) for group, _ in batch)
        prompt = f"""禁止分析过程。只返回严格 JSON 对象。

文件名: {file_name}

任务：编号标题、Part、Chapter、篇章标题已经形成稳定骨架。请根据标题语义和前后结构，
判断每个 CANDIDATE 无编号标题在目录树中的真实等级。

规则：
1. ANCHOR 是稳定结构，只用于理解上下文，不要为 ANCHOR 返回结果。
2. 不能仅因物理位置紧邻，就认定 CANDIDATE 是前一个编号标题的子标题。
3. 根据语义范围判断关系：
   - child_of_previous：是前一个标题覆盖主题下的具体解释、组成部分、方法、案例或问题，
     level=父标题level+1；
   - sibling_of_previous：与前一个标题同级，level相同；
   - parent_of_next：当前无编号标题实际上是后一个编号小节的父标题，当前 level 应比 next 低一级；
   - independent：属于当前 Part/章下的独立栏目；
   - keep：证据不足，保留 current 等级。
4. 连续多个并列、语义完整的 CANDIDATE 通常互为兄弟，不能形成逐项加深的阶梯。
5. “小结/习题/Summary/Exercises”等可能是章级栏目，也可能属于局部章节；必须结合语义和前后锚点判断，
   不要依赖固定关键词。
6. 如果无编号标题后出现 6.1、7.1 等，但目录中没有对应的“第6章/6 Title”，
   无编号标题通常是章标题漏掉了编号，应优先使用 parent_of_next。
   尤其当前后编号根从 6.x 切换到 7.1，且中间只有一个语义完整的无编号标题时，
   该标题通常就是缺号的第7章标题。
7. level 只能是 2-6。没有充分把握时使用 relation=keep 和当前等级。
8. 每个 CANDIDATE 必须恰好返回一项，index 使用输入中的全局 index。

语义示例：
- H4 `1.1 Curse of the LLMs and the idea of RAG`
  后接 `LLMs are not trained for facts`、`What is RAG?`
  再接 H4 `1.2 The novelty of RAG`
  => 两个无编号标题都在解释 1.1 的组成主题，应为 child_of_previous、H5，不是与 1.1 同级。
- H3 `1.3 最后一节` 后接 `本章小结`，再接 H2 `第2章`
  => `本章小结` 是章内独立栏目，通常与 1.3 同级，不是 1.3 的子标题。
- H2 `Part 3` 后接无编号标题 `Progression of RAG systems`，再接 H3 `6.1 Limitations`
  => 无编号标题是漏掉章号的章标题，应为 parent_of_next、H3，并让 6.1 成为 H4。

局部目录窗口：
{blocks}

返回 JSON：
{expected}"""
        messages = [
            {
                "role": "system",
                "content": "你是书籍目录语义层级判断器。根据标题语义确定父子和同级关系，只输出严格 JSON。/no_think",
            },
            {"role": "user", "content": "/no_think\n" + prompt},
        ]
        result = call_llm_json_with_repair(
            messages,
            max_tokens=max(1200, min(8000, 500 + candidate_count * 150)),
            stage="layer3_unnumbered_semantic_levels",
            expected_format=expected,
            timeout=TIMEOUT,
        )

        allowed_by_group = {
            int(group["id"]): {idx + 1 for idx in group["candidate_indices"]}
            for group, _ in batch
        }
        next_by_group = {
            int(group["id"]): group.get("next_anchor")
            for group, _ in batch
        }
        prev_by_group = {
            int(group["id"]): group.get("prev_anchor")
            for group, _ in batch
        }
        for result_group in result.get("groups", []):
            try:
                group_id = int(result_group.get("id"))
            except Exception:
                continue
            allowed = allowed_by_group.get(group_id, set())
            for item in result_group.get("items", []):
                try:
                    one_based_index = int(item.get("index"))
                    level = int(item.get("level"))
                    confidence = float(item.get("confidence", 0))
                except Exception:
                    continue
                if one_based_index not in allowed or not 2 <= level <= 6 or confidence < 0.60:
                    continue
                idx = one_based_index - 1
                relation = str(item.get("relation") or "").strip().lower()
                prev_index = prev_by_group.get(group_id)
                next_index = next_by_group.get(group_id)

                # LLM 负责语义关系；程序负责把关系转换成自洽、无跳级的等级。
                if relation == "child_of_previous" and prev_index is not None:
                    level = min(int(toc_headings[int(prev_index)].get("level") or 2) + 1, 6)
                elif relation == "sibling_of_previous" and prev_index is not None:
                    level = int(toc_headings[int(prev_index)].get("level") or 2)
                elif relation == "parent_of_next" and next_index is not None:
                    level = _semantic_parent_level_for_next_anchor(
                        toc_headings,
                        idx,
                        int(next_index),
                    )

                old_level = int(toc_headings[idx].get("level") or 2)
                if old_level != level:
                    toc_headings[idx]["_orig_level"] = toc_headings[idx].get("_orig_level", old_level)
                    toc_headings[idx]["level"] = level
                    toc_headings[idx].setdefault("_fixed", "")
                    toc_headings[idx]["_fixed"] += f"第三层语义→H{level}"
                    changed += 1

                if relation == "parent_of_next" and next_index is not None and confidence >= 0.70:
                    parent_of_next.append((int(next_index), min(level + 1, 6)))

        print(
            f"  第三层语义判断 batch {batch_no}/{len(batches)}: "
            f"{candidate_count} 个无编号标题"
        )

    for next_index, target_level in parent_of_next:
        _propagate_numbered_sibling_level(toc_headings, next_index, target_level)

    cross_changed, cross_recovered = _refine_cross_root_missing_parents_by_llm(
        toc_headings,
        groups,
        file_name,
    )
    changed += cross_changed

    if changed or parent_of_next or cross_recovered:
        print(
            f"  第三层语义判断: 调整 {changed} 个无编号标题，"
            f"恢复 {len(parent_of_next) + cross_recovered} 个缺号父标题关系"
        )


def _is_layer3_part_root(heading: dict) -> bool:
    """判断标题是否是 Part/篇/卷/部级语义块根节点。"""
    marker = _toc_structural_marker(title_key(display_title(heading.get("text", ""))))
    return bool(marker is not None and marker[0] == "part")


def _is_layer3_chapter_root(heading: dict) -> bool:
    """识别适合作为第三层分块边界的明确章标题。"""
    text = display_title(heading.get("text", ""))
    marker = _toc_structural_marker(title_key(text))
    if marker is not None and marker[0] == "chapter":
        return True

    path = _toc_number_path(text)
    return bool(
        path
        and "." not in path
        and path.isdigit()
        and _compute_decimal_depth(text) == 1
        and int(heading.get("level") or 2) <= 3
    )


def _layer3_parent_block_size(toc_headings: list[dict], indices: list[int]) -> tuple[int, int]:
    """返回 parent-tree 语义块的标题数和近似字符数。"""
    chars = sum(len(display_title(toc_headings[idx].get("text", ""))) + 36 for idx in indices)
    return len(indices), chars


def _split_layer3_parent_blocks(toc_headings: list[dict],
                                max_items: int = 180,
                                max_chars: int = 50000) -> list[dict]:
    """按完整 Part 或章节构造第三层 parent-tree 语义块。

    优先把整个 Part 交给 LLM。Part 过大时按章切分，并把 Part 根节点作为
    上下文重复放入各章块；没有 Part 时直接按章切分。
    """
    if not toc_headings:
        return []

    blocks: list[dict] = []
    block_id = 1

    def add_block(indices: list[int], label: str) -> None:
        nonlocal block_id
        indices = list(dict.fromkeys(indices))
        if not indices:
            return
        count, chars = _layer3_parent_block_size(toc_headings, indices)
        if count <= max_items and chars <= max_chars:
            blocks.append({"id": block_id, "label": label, "indices": indices})
            block_id += 1
            return

        # 极大章节继续拆分时，重复首个 Part/章标题作为稳定上下文。
        context = [indices[0]] if (
            _is_layer3_part_root(toc_headings[indices[0]])
            or _is_layer3_chapter_root(toc_headings[indices[0]])
        ) else []
        content = indices[1:] if context else indices
        room = max(20, max_items - len(context))
        for offset in range(0, len(content), room):
            chunk = context + content[offset:offset + room]
            blocks.append({
                "id": block_id,
                "label": f"{label} chunk {offset // room + 1}",
                "indices": list(dict.fromkeys(chunk)),
            })
            block_id += 1

    def first_backmatter(start: int, end: int) -> int | None:
        for idx in range(start, end):
            if _is_toc_backmatter_key(title_key(display_title(toc_headings[idx].get("text", "")))):
                return idx
        return None

    def add_segment(start: int, end: int, part_root: int | None, label: str) -> None:
        if start >= end:
            return
        backmatter = first_backmatter(start + (1 if part_root == start else 0), end)
        main_end = backmatter if backmatter is not None else end
        main_indices = list(range(start, main_end))
        count, chars = _layer3_parent_block_size(toc_headings, main_indices)
        if main_indices and count <= max_items and chars <= max_chars:
            add_block(main_indices, label)
        elif main_indices:
            chapter_start_from = start + (1 if part_root == start else 0)
            chapter_positions = [
                idx for idx in range(chapter_start_from, main_end)
                if _is_layer3_chapter_root(toc_headings[idx])
            ]
            if chapter_positions:
                first_chapter = chapter_positions[0]
                if start < first_chapter:
                    add_block(list(range(start, first_chapter)), f"{label} prelude")
                for pos, chapter_start in enumerate(chapter_positions):
                    chapter_end = (
                        chapter_positions[pos + 1]
                        if pos + 1 < len(chapter_positions)
                        else main_end
                    )
                    context = [part_root] if part_root is not None else []
                    add_block(
                        context + list(range(chapter_start, chapter_end)),
                        f"{label} chapter {pos + 1}",
                    )
            else:
                add_block(main_indices, label)

        if backmatter is not None:
            add_block(list(range(backmatter, end)), f"{label} backmatter")

    part_positions = [
        idx for idx, heading in enumerate(toc_headings)
        if _is_layer3_part_root(heading)
    ]
    if part_positions:
        if part_positions[0] > 0:
            add_segment(0, part_positions[0], None, "frontmatter")
        for pos, part_start in enumerate(part_positions):
            part_end = part_positions[pos + 1] if pos + 1 < len(part_positions) else len(toc_headings)
            add_segment(part_start, part_end, part_start, f"part {pos + 1}")
    else:
        add_segment(0, len(toc_headings), None, "book")

    covered = {idx for block in blocks for idx in block["indices"]}
    missing = [idx for idx in range(len(toc_headings)) if idx not in covered]
    if missing:
        add_block(missing, "uncovered")
    return blocks


def _format_layer3_parent_block(toc_headings: list[dict], block: dict) -> str:
    """格式化完整语义块，index 使用全书 1-based 标题序号。"""
    lines = []
    for idx in block["indices"]:
        heading = toc_headings[idx]
        text = compact_text(display_title(heading.get("text", "")), 240)
        number_path = _toc_number_path(text) or "-"
        lines.append(
            f'index={idx + 1} current=H{int(heading.get("level") or 2)} '
            f'number_path={number_path} text="{text}"'
        )
    return "\n".join(lines)


def _validate_layer3_parent_block_result(toc_headings: list[dict],
                                         block: dict,
                                         result: dict) -> tuple[dict[int, int | None],
                                                                dict[int, float],
                                                                list[str]]:
    """严格验证一个语义块返回的 parent_index 结果。"""
    allowed = {idx + 1 for idx in block["indices"]}
    parents: dict[int, int | None] = {}
    confidences: dict[int, float] = {}
    errors: list[str] = []
    items = result.get("items", [])
    if not isinstance(items, list):
        return {}, {}, ["items 不是列表"]

    for item in items:
        try:
            index = int(item.get("index"))
        except Exception:
            errors.append("存在无效 index")
            continue
        if index not in allowed:
            errors.append(f"index={index} 不属于当前语义块")
            continue
        if index in parents:
            errors.append(f"index={index} 重复返回")
            continue

        raw_parent = item.get("parent_index")
        if raw_parent is None:
            parent = None
        else:
            try:
                parent = int(raw_parent)
            except Exception:
                errors.append(f"index={index} 的 parent_index 无效")
                continue
            if parent not in allowed:
                errors.append(f"index={index} 的父节点 {parent} 不在当前语义块")
                continue
            if parent >= index:
                errors.append(f"index={index} 的父节点必须位于它之前")
                continue

        parents[index] = parent
        try:
            confidences[index] = float(item.get("confidence", 0.0))
        except Exception:
            confidences[index] = 0.0

    missing = sorted(allowed - set(parents))
    if missing:
        errors.append(f"缺少 index: {missing[:20]}")

    # 明确编号路径存在父标题时，父节点必须指向该编号父标题。
    path_to_indices: dict[str, list[int]] = {}
    for index in sorted(allowed):
        path = _toc_number_path(display_title(toc_headings[index - 1].get("text", "")))
        if path:
            path_to_indices.setdefault(path, []).append(index)

    for index, parent in parents.items():
        path = _toc_number_path(display_title(toc_headings[index - 1].get("text", "")))
        parts = path.split(".") if path else []
        if len(parts) < 2:
            continue
        parent_path = ".".join(parts[:-1])
        candidates = [x for x in path_to_indices.get(parent_path, []) if x < index]
        if candidates:
            expected_parent = candidates[-1]
            if parent != expected_parent:
                errors.append(
                    f"index={index} 编号父节点应为 {expected_parent}，实际为 {parent}"
                )

    for index in sorted(allowed):
        if _is_layer3_part_root(toc_headings[index - 1]) and parents.get(index) is not None:
            errors.append(f"Part 根标题 index={index} 的 parent_index 必须为 null")

    return parents, confidences, errors


def _call_layer3_parent_tree_llm(toc_headings: list[dict],
                                 block: dict,
                                 file_name: str) -> tuple[dict[int, int | None], dict[int, float]]:
    """让 LLM 为完整 Part/章节块返回每个标题的直接父节点。"""
    expected = (
        '{"items": [{"index": 1, "parent_index": null, "confidence": 0.95}]}'
    )
    block_text = _format_layer3_parent_block(toc_headings, block)
    correction = ""

    for attempt in range(4):
        prompt = f"""禁止分析过程。只返回严格 JSON 对象。

文件名: {file_name}
当前语义块: {block.get('label', '')}

任务：根据完整语义块中的编号结构和标题语义，为每个标题选择直接父标题 parent_index。
不要直接判断 Markdown H 等级；程序会根据父子树自动计算等级。

规则：
1. 必须为输入中的每一个 index 恰好返回一项，不能遗漏、重复或创造标题。
2. parent_index 必须是 null，或当前语义块中位于该标题之前的 index。
3. Part/第X篇/第X部分通常是语义块根节点，parent_index=null。
4. 完整编号是强结构证据：
   - `3 Title` 通常是 Part 的子标题，或无 Part 时的根标题；
   - `3.2 Title` 的父标题通常是第3章；
   - `3.2.1 Title` 的父标题通常是 `3.2 Title`。
5. 无编号标题必须根据语义选择父节点：
   - 具体解释、组成部分、方法、案例或问题，属于语义覆盖它的父标题；
   - 多个并列主题应共享同一个父节点，不能形成逐项加深的阶梯；
   - 不能仅因为物理位置最近就选择父节点；但编号区间是强结构证据，必须与语义一起判断。
6. 同级编号标题之间的无编号标题，通常属于前一个编号标题覆盖的局部主题：
   - 如果若干无编号标题位于 `2.2 Title` 与下一个同级标题 `2.3 Title` 之间，
     并且它们语义上是 `2.2` 的具体过程、组成部分、方法、风险、案例或问题，
     则这些无编号标题的直接父节点应共同指向 `2.2`，不能全部直接挂到第2章；
   - 同理，位于 `2.3` 与 `2.4` 之间、语义解释 `2.3` 的无编号标题，应共同指向 `2.3`；
   - 这不是机械就近规则。如果无编号标题语义上明显是章级独立栏目、总结或后一个编号标题的父标题，
     应根据真实语义选择其他父节点。
7. 纸质目录可能省略更深层编号。无编号标题在正文中可能实际是 `2.2.1/2.2.2/2.2.3`；
   当它们在同级编号区间内形成并列子主题时，应恢复为前一编号标题的子节点。
8. OCR 可能丢失章号。例如无编号标题后直接出现 `6.1/6.2`，该无编号标题可能就是缺号的第6章父标题。
9. 前言、序、致谢、参考文献、索引等独立书籍结构可以返回 parent_index=null。
10. current H 等级只是第二层参考，不是最终答案；parent_index 才是最终语义判断。

编号区间示例：
- `2.2 Data preparation`
  后接无编号标题 `Loading data`、`Cleaning data`、`Validating data`
  再接同级编号标题 `2.3 Model training`
  => 三个无编号标题都是 `2.2` 的并列子主题，parent_index 必须共同指向 `2.2`。
- `4.1 Security risks`
  后接无编号标题 `Weak passwords`、`Unsafe permissions`、`Unpatched services`
  再接同级编号标题 `4.2 Security controls`
  => 三个无编号标题都是 `4.1` 的并列子主题，parent_index 必须共同指向 `4.1`。
{correction}
完整语义块：
{block_text}

返回 JSON：
{expected}"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是书籍目录父子树重建器。根据完整语义块判断每个标题的直接父节点，"
                    "只能输出严格 JSON。/no_think"
                ),
            },
            {"role": "user", "content": "/no_think\n" + prompt},
        ]
        result = call_llm_json_with_repair(
            messages,
            max_tokens=TOC_PARSE_CHUNK_MAX_TOKENS,
            stage="layer3_parent_tree",
            expected_format=expected,
            timeout=TIMEOUT,
        )
        parents, confidences, errors = _validate_layer3_parent_block_result(
            toc_headings,
            block,
            result,
        )
        if not errors:
            return parents, confidences
        correction = (
            "\n上一轮 parent tree 不合法，必须修复以下问题后重新返回完整 items：\n- "
            + "\n- ".join(errors[:20])
            + "\n"
        )
        print(
            f"    [layer3_parent_tree] 语义块 {block['id']} 校验失败 "
            f"{attempt + 1}/4，要求 LLM 修复"
        )

    raise SkipMarkdownFile(
        "layer3_parent_tree_invalid",
        "layer3_parent_tree",
        f"第三层语义块 {block.get('label')} 连续 4 次未返回合法 parent tree",
    )


def _validate_complete_layer3_parent_tree(toc_headings: list[dict],
                                          parents: dict[int, int | None]) -> None:
    """验证合并后的全书 parent tree 完整、向前引用且无循环。"""
    expected = set(range(1, len(toc_headings) + 1))
    missing = sorted(expected - set(parents))
    if missing:
        raise SkipMarkdownFile(
            "layer3_parent_tree_incomplete",
            "layer3_parent_tree",
            f"第三层 parent tree 缺少 {len(missing)} 个标题: {missing[:20]}",
        )

    for index, parent in parents.items():
        if parent is None:
            continue
        if parent not in expected or parent >= index:
            raise SkipMarkdownFile(
                "layer3_parent_tree_invalid",
                "layer3_parent_tree",
                f"第三层 parent tree 非法: index={index}, parent_index={parent}",
            )


def _apply_layer3_parent_tree_levels(toc_headings: list[dict],
                                     parents: dict[int, int | None],
                                     confidences: dict[int, float]) -> list[dict]:
    """根据 parent_index 确定性计算 H 等级，并保存父节点诊断信息。"""
    levels: dict[int, int] = {}
    h1_used = False
    for index, heading in enumerate(toc_headings, 1):
        parent = parents.get(index)
        old_level = int(heading.get("level") or 2)
        if parent is None:
            if old_level == 1 and not h1_used:
                level = 1
                h1_used = True
            else:
                level = 2
        else:
            level = min(levels[parent] + 1, 6)

        levels[index] = level
        heading["parent_index"] = parent
        heading["parent_text"] = (
            display_title(toc_headings[parent - 1].get("text", ""))
            if parent is not None
            else None
        )
        heading["semantic_confidence"] = round(float(confidences.get(index, 0.0)), 4)
        if old_level != level:
            heading["_orig_level"] = heading.get("_orig_level", old_level)
            heading["level"] = level
            heading.setdefault("_fixed", "")
            heading["_fixed"] += f"parent_tree→H{level}"
        else:
            heading["level"] = level
    return toc_headings


def _rebuild_layer3_parent_tree(toc_headings: list[dict],
                                file_name: str) -> list[dict]:
    """第三层核心：完整语义块判断父节点，再由程序计算最终等级。"""
    blocks = _split_layer3_parent_blocks(toc_headings)
    parents: dict[int, int | None] = {}
    confidences: dict[int, float] = {}
    print(f"  第三层 parent tree: {len(toc_headings)} 个标题 → {len(blocks)} 个完整语义块")

    for pos, block in enumerate(blocks, 1):
        block_parents, block_confidences = _call_layer3_parent_tree_llm(
            toc_headings,
            block,
            file_name,
        )
        for index, parent in block_parents.items():
            if index in parents and parents[index] != parent:
                raise SkipMarkdownFile(
                    "layer3_parent_tree_conflict",
                    "layer3_parent_tree",
                    (
                        f"第三层重复上下文标题父节点冲突: index={index}, "
                        f"{parents[index]} != {parent}"
                    ),
                )
            parents[index] = parent
            confidences[index] = max(confidences.get(index, 0.0), block_confidences.get(index, 0.0))
        print(
            f"    第三层语义块 {pos}/{len(blocks)}: "
            f"{block.get('label')}，{len(block['indices'])} 个标题"
        )

    _validate_complete_layer3_parent_tree(toc_headings, parents)
    return _apply_layer3_parent_tree_levels(toc_headings, parents, confidences)


def layer3_validate(toc_headings: list[dict], file_name: str = "") -> list[dict]:
    """第三层: 用完整语义块重建 parent tree，再确定性计算标题等级。"""
    if not toc_headings:
        return toc_headings

    toc_headings = _merge_toc_wrapped_continuations(toc_headings)
    return _rebuild_layer3_parent_tree(toc_headings, file_name)


# ============================================================
# 第四层 A: 正文候选准备
# ============================================================


def _build_body_candidate_lines(md_text: str, start_line: int) -> list[dict]:
    """从目录结束后构造正文候选行，并跳过整个 details 块。

    返回的 line 始终是原始 Markdown 行号，后续输出和 JSON 可追踪回源文件。
    """
    lines = md_text.split("\n")
    body = []
    in_details = False
    start = max(1, start_line)

    for i in range(start - 1, len(lines)):
        text = lines[i].rstrip("\n")
        stripped = text.strip()
        if stripped.startswith("<details"):
            in_details = True
            continue
        if stripped.startswith("</details>"):
            in_details = False
            continue
        if in_details:
            continue
        body.append({"line": i + 1, "text": text})
    return body


def _compress_body_empty_lines(body_lines: list[dict]) -> list[dict]:
    """压缩正文候选中的连续空行，保留第一条空行的原始行号。"""
    compressed = []
    prev_empty = False
    for rec in body_lines:
        if rec.get("text", "").strip() == "":
            if not prev_empty:
                compressed.append(rec)
                prev_empty = True
        else:
            compressed.append(rec)
            prev_empty = False
    return compressed


def _greedy_fix_split_body_headings(body_lines: list[dict],
                                    toc_headings: list[dict]) -> tuple[list[dict], int]:
    """按目录标题贪心修复正文中被 OCR 拆断的标题。

    典型场景:
        "# 第1章" + "# 学习Linux的经验与技巧" -> "# 第1章 学习Linux的经验与技巧"
        "# 第20章" + "负载均衡集群 LVS 与 HAProxy" -> "# 第20章 负载均衡集群 LVS 与 HAProxy"
    """
    records = [dict(x) for x in body_lines]
    toc_keys = {title_key(h.get("text", "")) for h in toc_headings if h.get("text")}
    pat = re.compile(r"^(#{1,6})\s+(.+)")

    head_positions = []
    for i, rec in enumerate(records):
        m = pat.match(rec.get("text", "").strip())
        if m:
            head_positions.append((i, m.group(2).strip()))

    fixed_count = 0
    for idx, (pos, text) in enumerate(head_positions):
        if title_key(text) in toc_keys:
            continue

        merged = None
        merged_lines = []

        # A: 下一个 # 标题在 5 个候选行内。
        if idx + 1 < len(head_positions):
            npos, ntext = head_positions[idx + 1]
            if 0 < npos - pos <= 5 and title_key(f"{text} {ntext}") in toc_keys:
                merged = f"{text} {ntext}"
                merged_lines = [npos]

        # B: 下方最多 2 条短纯文本续文。
        if merged is None:
            plain_lines = []
            j = pos + 1
            while j < len(records) and len(plain_lines) < 2:
                s = records[j].get("text", "").strip()
                if not s:
                    j += 1
                    continue
                if s.startswith("#"):
                    break
                if len(s) <= 80:
                    plain_lines.append((j, s))
                j += 1

            for take in (1, 2):
                if len(plain_lines) >= take:
                    extra = " ".join(x[1] for x in plain_lines[:take])
                    if title_key(f"{text} {extra}") in toc_keys:
                        merged = f"{text} {extra}"
                        merged_lines = [x[0] for x in plain_lines[:take]]
                        break

        if merged:
            records[pos]["text"] = records[pos]["text"].rstrip() + " " + merged[len(text) + 1:]
            for line_idx in merged_lines:
                records[line_idx]["text"] = ""
            fixed_count += 1

    return _compress_body_empty_lines(records), fixed_count


def _extract_headings_from_body_lines(body_lines: list[dict],
                                      toc_headings: list[dict] | None = None,
                                      context_lines: int = 5) -> list[dict]:
    """从正文行中提取 `#` 候选标题。

    正常代码块里的 `#` 仍然跳过；如果 OCR/转换造成代码块没有正确闭合，
    但某个 `#` 行能匹配目录标题，就把它放回第四层候选。
    """
    pat = re.compile(r"^(#{1,6})\s+(.+)")
    headings = []
    in_code = False
    toc_keys = [title_key(h.get("text", "")) for h in (toc_headings or [])]

    for i, rec in enumerate(body_lines):
        stripped = rec.get("text", "").strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            continue

        m = pat.match(stripped)
        if not m:
            continue
        title = m.group(2).strip()

        code_match_kind = None
        if in_code:
            body_key = title_key(title)
            for toc_key in toc_keys:
                kind = _toc_match_kind(body_key, toc_key)
                if kind:
                    code_match_kind = kind
                    break
            if not code_match_kind:
                continue

        prev_text = "\n".join(
            x.get("text", "").strip()
            for x in body_lines[max(0, i - context_lines):i]
        )
        next_text = "\n".join(
            x.get("text", "").strip()
            for x in body_lines[i + 1:i + 1 + context_lines]
        )
        headings.append({
            "line": rec["line"],
            "raw": stripped,
            "text": title,
            "prev_text": prev_text,
            "next_text": next_text,
            "inside_code_block": in_code,
            "code_toc_match_kind": code_match_kind,
        })
    return headings


def _find_body_title_start_by_llm(body_lines: list[dict],
                                  toc_headings: list[dict],
                                  search_limit: int = 800,
                                  batch: int = 12,
                                  step: int = 8) -> int | None:
    """让 LLM 在正文候选区中寻找真正正文起点的标题行下标。"""
    if not body_lines or not toc_headings:
        return None

    first_toc_text = toc_headings[0].get("text", "") if toc_headings else ""
    first_toc_level = int(toc_headings[0].get("level", 2) or 2) if toc_headings else 2
    toc_ref = "\n".join(
        f"order={i} H{h.get('level', 2)}: {h.get('text', '')}"
        for i, h in enumerate(toc_headings[:40])
    )
    pat = re.compile(r"^(#{1,6})\s+(.+)")
    limit = min(search_limit, len(body_lines))
    heading_indices = [
        i for i, rec in enumerate(body_lines[:limit])
        if pat.match(rec.get("text", "").strip())
    ]
    api_failures = 0

    # 先用目录前部父级标题做确定性匹配。LLM 只兜底，避免把正文起点误选到很后面的子标题。
    first_toc_key = title_key(toc_headings[0].get("text", "")) if toc_headings else ""
    parent_toc_candidates = []
    fallback_toc_candidates = []
    for toc_order, toc_h in enumerate(toc_headings[:24]):
        toc_key = title_key(toc_h.get("text", ""))
        if not toc_key:
            continue
        level = int(toc_h.get("level", 2) or 2)
        item = (toc_order, level, toc_key, toc_h.get("text", ""))
        if toc_order == 0 or (toc_order <= 12 and level <= 2):
            parent_toc_candidates.append(item)
        elif toc_order <= 12 and level <= 3:
            fallback_toc_candidates.append(item)

    for body_idx in heading_indices[:120]:
        m = pat.match(body_lines[body_idx].get("text", "").strip())
        if not m:
            continue
        cand_title = m.group(2).strip()
        if _is_noise_heading_text(cand_title):
            continue
        body_key = title_key(cand_title)
        for toc_order, level, toc_key, toc_text in parent_toc_candidates:
            if _toc_match_kind(body_key, toc_key):
                print(
                    f"  正文起点规则匹配: 行{body_lines[body_idx]['line']} "
                    f"{cand_title[:60]} (toc_order={toc_order} H{level})"
                )
                return body_idx

    earliest_fallback = None
    for body_idx in heading_indices[:120]:
        m = pat.match(body_lines[body_idx].get("text", "").strip())
        if not m:
            continue
        cand_title = m.group(2).strip()
        if _is_noise_heading_text(cand_title):
            continue
        body_key = title_key(cand_title)
        for toc_order, level, toc_key, toc_text in fallback_toc_candidates:
            if _toc_match_kind(body_key, toc_key):
                earliest_fallback = (body_idx, toc_order, level, cand_title)
                break
        if earliest_fallback:
            break

    for cand_start in range(0, len(heading_indices), step):
        candidate_indices = heading_indices[cand_start:cand_start + batch]
        valid_indices = set(candidate_indices)
        valid_lines = {body_lines[i]["line"]: i for i in candidate_indices}

        blocks = []
        for body_idx in candidate_indices:
            rec = body_lines[body_idx]
            after = []
            for next_rec in body_lines[body_idx + 1:body_idx + 9]:
                s = next_rec.get("text", "").strip()
                if s:
                    after.append(f"    line={next_rec['line']}: {s[:120]}")
                if len(after) >= 5:
                    break
            m = pat.match(rec.get("text", "").strip())
            cand_title = m.group(2).strip() if m else rec.get("text", "").strip()
            cand_key = title_key(cand_title)
            relation = []
            if first_toc_key and _toc_match_kind(cand_key, first_toc_key):
                relation.append("matches_first_toc")
            for toc_order, toc_h in enumerate(toc_headings[:12]):
                toc_key = title_key(toc_h.get("text", ""))
                if toc_key and _toc_match_kind(cand_key, toc_key):
                    relation.append(f"matches_toc_order_{toc_order}_H{toc_h.get('level', 2)}")
                    break
            if _has_numbered_prefix(cand_key):
                relation.append("numbered_subsection")
            if _is_noise_heading_text(rec.get("text", "")):
                relation.append("noise_like")
            relation_text = ", ".join(relation) if relation else "no_direct_toc_relation"
            blocks.append(
                f"[candidate idx={body_idx} line={rec['line']} relation={relation_text}] "
                f"{rec.get('text', '').strip()[:180]}\n" + "\n".join(after)
            )
        chunk_text = "\n\n".join(blocks)

        prompt = f"""禁止分析过程。你的回复第一个字符必须是 {{，最后一个字符必须是 }}。
只输出一个 JSON 对象；不要输出解释、推理、Markdown 或代码块。

## 目标目录起点
order=0 H{first_toc_level}: {first_toc_text}

## 目录标题列表
{toc_ref}

## 候选 Markdown 标题
idx 是当前正文候选列表下标；line 是原始 Markdown 行号；relation 是程序根据目录结构给出的辅助标签。
每个候选后面列出后续上下文。
{chunk_text}

任务：从候选标题中选择“目标目录起点”对应的真正正文开始位置。

判定规则：
1. start_index 必须是上方某个 candidate idx；不要返回普通导读段、空行或上下文行。
2. 优先选择 relation 包含 matches_first_toc 的候选；它就是目标目录起点。
3. 如果存在目标目录起点的父级/章标题，不要选择 1.1、A.1、10.1 这类 numbered_subsection 子标题。
4. 只有当本批没有目标父级标题、且更早候选都是封面/广告/噪声时，才允许选择子标题作为兜底。
5. 跳过封面、书名、篇章列表、广告页、图片标记、图片说明，以及 OCR 噪声标题（如 # LINUX / # JINUX）。
6. 章节标题后面紧跟章节插图，但之后进入小节/正文段落，仍然算正文起点。
7. 如果本批候选没有目标目录起点或可信兜底，返回 found=false。
8. 最终只输出 JSON 对象。

返回格式：
{{"found": true/false, "start_index": idx数字或null, "start_line": 原始行号或null, "matched_relation": "候选relation或null", "reason": "一句话说明"}}"""

        messages = [
            {"role": "system", "content": "你是文档结构分析师。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
            {"role": "user", "content": prompt},
        ]

        try:
            result = parse_json(call_llm(messages, max_tokens=3000))
            print(f"  正文起点扫描候选[{cand_start:>4}]: found={result.get('found')} "
                  f"{str(result.get('reason', ''))[:60]}")
            if not result.get("found"):
                continue

            body_idx = None
            if result.get("start_index") is not None:
                body_idx = int(result["start_index"])
            elif result.get("start_line") is not None:
                line_no = int(result["start_line"])
                body_idx = valid_lines.get(line_no)

            if body_idx not in valid_indices:
                raise ValueError(f"LLM 返回的正文起点不在范围内: {result}")

            if not pat.match(body_lines[body_idx].get("text", "").strip()):
                raise ValueError(f"LLM 返回的正文起点不是 # 标题行: {result}")

            if earliest_fallback and body_idx > earliest_fallback[0]:
                fb_idx, toc_order, level, cand_title = earliest_fallback
                print(
                    f"  正文起点约束回退: LLM 选择 idx={body_idx}，"
                    f"回退到更早目录匹配 idx={fb_idx} "
                    f"行{body_lines[fb_idx]['line']} {cand_title[:50]} "
                    f"(toc_order={toc_order} H{level})"
                )
                return fb_idx

            return body_idx
        except Exception as e:
            api_failures += 1
            print(f"  正文起点扫描候选[{cand_start:>4}] 错误: {e}")
            if api_failures >= 3:
                print("  正文起点 LLM 连续失败，停止扫描")
                break

    return None


def _prepare_layer4_body_lines(md_text: str, toc_range: dict,
                               toc_headings: list[dict]) -> tuple[int, list[dict]]:
    """第四层准备正文区: 清洗、修复、压缩，并用 LLM 找真正正文标题起点。"""
    candidate_start = int(toc_range.get("toc_end_line", 0) or 0) + 1
    body_lines = _build_body_candidate_lines(md_text, candidate_start)
    print(f"  正文候选: 从行{candidate_start}开始, 去 details 后 {len(body_lines)} 行")

    body_lines, fixed_count = _greedy_fix_split_body_headings(body_lines, toc_headings)
    print(f"  正文候选: 贪心修复 {fixed_count} 个截断标题, 压缩后 {len(body_lines)} 行")

    start_idx = _find_body_title_start_by_llm(body_lines, toc_headings)
    if start_idx is None:
        print("  未找到可信正文标题起点，保留目录结束后的候选正文")
        start_idx = 0

    body_start = body_lines[start_idx]["line"] if body_lines else candidate_start
    body_lines = body_lines[start_idx:]
    print(f"  正文标题起点: 行{body_start}, body_lines: {len(body_lines)} 行")
    return body_start, body_lines

# ============================================================
# 第四层 B: 正文标题匹配与等级映射
# ============================================================

def _strip_chapter_prefix(text: str) -> str:
    """去掉章节前缀，辅助第四层模糊匹配。

    作用:
        有些正文标题可能是 "概述"，目录标题是 "第1章 概述"。
        去前缀后可以匹配到同一标题。

    输入:
        text: str，标准化后的标题 key 或普通标题文本。

    输出:
        str，去掉 "第X章/Chapter X/Section X/Part X" 后的文本。

    例子:
        _strip_chapter_prefix("第1章 概述") -> "概述"
    """
    # 第1章 / Chapter 1 / Section 1.1 / Part I；兼容 title_key 去空格后的 key。
    text = re.sub(r"^第\s*[一二三四五六七八九十百零0-9]+\s*[章节篇卷部]\s*", "", text)
    text = re.sub(r"^附录\s*[A-Za-z]\s*(?:[.、．:：\-\s]+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:Chapter|Chap|Section|Sec|Subsection|Subsec|Part|Book|Volume|Vol)\s*"
        r"(?:\d+(?:\.\d+)*|[IVXLCDM]+)\s*(?:[.、．:：\-\s]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^Appendix\s*[A-Za-z]\s*(?:[.、．:：\-\s]+)?", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_numbered_prefix(text: str) -> str:
    """去掉 1.2 / Section 1.2 / Chapter 1 这类结构编号前缀。"""
    text = text or ""
    text = re.sub(
        r"^(?:Section|Sec|Subsection|Subsec)\s*\d+(?:\.\d+)*\s*(?:[.、．:：\-\s]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:Chapter|Chap|Part|Book|Volume|Vol)\s*(?:\d+|[IVXLCDM]+)\s*(?:[.、．:：\-\s]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^Appendix\s*[A-Za-z](?:\.\d+(?:\.\d+)*)?\s*(?:[.、．:：\-\s]+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+(?:\.\d+)*(?:[.、．:：\-\s]+)?", "", text)
    return text.strip()


def _chapter_number_key(text: str) -> str:
    """提取第 N 章/篇/部/卷的编号，用于 OCR 模糊匹配章标题。"""
    m = re.match(r"^第([一二三四五六七八九十百零0-9]+)[章节篇卷部]", text or "")
    return m.group(1) if m else ""


def _similar_title_text(a: str, b: str) -> bool:
    """判断两个已标准化标题主体是否足够相似。"""
    if not a or not b:
        return False
    if _contains_title_phrase(a, b) or _contains_title_phrase(b, a):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.72


_NOISE_HEADING_KEYS = {"linux", "tinux", "jinux"}


def _is_noise_heading_text(text: str) -> bool:
    """识别 OCR 页眉/书名噪声标题，如 # LINUX / # TINUX / # JINUX。"""
    s = (text or "").strip()
    m = re.match(r"^#{1,6}\s+(.+)$", s)
    if m:
        s = m.group(1).strip()
    return title_key(s) in _NOISE_HEADING_KEYS


def _is_bare_chapter_title(text: str) -> bool:
    """判断正文候选是否只是一个孤立的章号。

    PDF/OCR 转 Markdown 时，经常把一个章标题拆成两行:
        # 第9章
        # 中断管理

    目录里的真实标题通常是 "第 9 章 中断管理"。第四层映射时需要先识别这种
    "孤立章号"，再和后一个候选标题合并匹配目录。

    输入:
        text: str，候选标题文本。

    输出:
        bool，True 表示类似 "第9章"、"第 10 章"、"第一章"。
    """
    tk = title_key(text)
    return bool(
        re.fullmatch(r"第[一二三四五六七八九十百零0-9]+[章节篇卷部]", tk)  # 第9章、第一章
        or re.fullmatch(r"chapter\s*\d+", tk)                          # Chapter 20
        or re.fullmatch(r"part\s*([ivx]+|\d+)", tk)                   # Part III
        or re.fullmatch(r"section\s*\d+(\.\d+)*", tk)                 # Section 5.1
        or re.fullmatch(r"\d+\.\d+(\.\d+)*", tk)                      # 20.5, 7.2.3
    )


def _contains_title_phrase(container: str, needle: str) -> bool:
    """判断 `needle` 是否作为完整标题短语出现在 `container` 中。

    作用:
        第四层模糊匹配需要处理 "中断管理" vs "第9章 中断管理" 这类情况，
        但不能把代码/API 标识符前缀误当成同一标题，例如:

        - OSStatTaskCPUUsage 不应匹配 OSStatTaskCPUUsageInit()
        - OSStatTaskHook() 可以匹配 A. 56 OSStatTaskHook()

    输入:
        container: str，较长的标准化标题 key。
        needle: str，较短的标准化标题 key。

    输出:
        bool，True 表示 needle 以较完整的短语边界出现在 container 中。
    """
    if not container or not needle:
        return False
    start = container.find(needle)
    while start >= 0:
        end = start + len(needle)
        before = container[start - 1] if start > 0 else ""
        after = container[end] if end < len(container) else ""
        before_ok = not before or not re.match(r"[A-Za-z0-9_]", before)
        after_ok = not after or not re.match(r"[A-Za-z0-9_]", after)
        if before_ok and after_ok:
            return True
        start = container.find(needle, start + 1)
    return False


def _has_numbered_prefix(key: str) -> bool:
    """判断标题 key 是否带有章节/附录编号前缀。

    作用:
        避免把正文里的短词 "# 信号量" 模糊匹配到目录里的
        "13.3 信号量" 或 "A.5 信号量 同步"。
    """
    key = (key or "").strip().lower()
    return bool(
        re.match(r"^\d+(?:\.\d+)*(?:[.、．])?", key)
        or re.match(r"^第[一二两三四五六七八九十百千零〇0-9]+[章节篇卷部]", key)
        or re.match(r"^[a-z]\.\s*\d+", key)
        or re.match(
            r"^(?:chapter|chap|section|sec|subsection|subsec|part|book|volume|vol)"
            r"\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)",
            key,
        )
        or re.match(r"^(?:appendix|附录)[a-z](?:\.\d+(?:\.\d+)*)?", key)
    )


def _numbered_tail_symbol_key(key: str) -> str:
    """取编号后的标题主体，并去掉符号，供编号一致时做 OCR 宽匹配。"""
    tail = _strip_numbered_prefix(_strip_chapter_prefix(key or ""))
    return symbol_stripped_title_key(tail)


def _drop_cjk_connector_yi(key: str) -> str:
    """OCR 常把中文连接横线识别成“一”，编号已一致时可把它当连接符。"""
    return re.sub(r"(?<=[\u4e00-\u9fff])一(?=[\u4e00-\u9fff])", "", key or "")


def _same_number_fuzzy_match(body_key: str, toc_key: str) -> bool:
    """编号路径一致时，对标题尾部做更宽的 OCR/符号容错匹配。"""
    body_number = _toc_number_path(body_key)
    toc_number = _toc_number_path(toc_key)
    if not body_number or body_number != toc_number:
        return False

    body_tail = _numbered_tail_symbol_key(body_key)
    toc_tail = _numbered_tail_symbol_key(toc_key)
    if not body_tail or not toc_tail:
        return False
    if body_tail == toc_tail:
        return True

    body_yi = _drop_cjk_connector_yi(body_tail)
    toc_yi = _drop_cjk_connector_yi(toc_tail)
    if (
        body_yi
        and toc_yi
        and min(len(body_yi), len(toc_yi)) >= 3
        and body_yi == toc_yi
    ):
        return True

    if min(len(body_yi), len(toc_yi)) >= 4:
        return difflib.SequenceMatcher(None, body_yi, toc_yi).ratio() >= 0.86
    return False


def _toc_match_kind(body_key: str, toc_key: str) -> str | None:
    """返回正文标题和目录标题的匹配类型。

    输入:
        body_key: 正文候选标题的标准化 key，例如 "处理器"。
        toc_key: 目录标题的标准化 key，例如 "第3章 处理器"。

    输出:
        "exact" | "stripped" | "phrase" | None。

    例子:
        _toc_match_kind("处理器", "第3章 处理器") -> "stripped"
        _toc_match_kind("osstat", "osstatinit") -> None
    """
    if not body_key or not toc_key:
        return None
    if body_key == toc_key:
        return "exact"

    body_number_path = _toc_number_path(body_key)
    toc_number_path = _toc_number_path(toc_key)
    number_path_conflict = (
        body_number_path
        and toc_number_path
        and body_number_path != toc_number_path
    )
    if number_path_conflict:
        return None

    body_stripped = _strip_chapter_prefix(body_key)
    toc_stripped = _strip_chapter_prefix(toc_key)
    if body_stripped and toc_stripped and body_stripped == toc_stripped:
        return "stripped"

    body_num_stripped = _strip_numbered_prefix(body_stripped)
    toc_num_stripped = _strip_numbered_prefix(toc_stripped)
    if body_num_stripped and toc_num_stripped and body_num_stripped == toc_num_stripped:
        return "number_stripped"

    body_chapter_no = _chapter_number_key(body_key)
    toc_chapter_no = _chapter_number_key(toc_key)
    if body_chapter_no and body_chapter_no == toc_chapter_no:
        if _similar_title_text(body_num_stripped, toc_num_stripped):
            return "chapter_fuzzy"

    body_symbol_stripped = symbol_stripped_title_key(body_key)
    toc_symbol_stripped = symbol_stripped_title_key(toc_key)
    if (
        body_symbol_stripped
        and toc_symbol_stripped
        and body_symbol_stripped == toc_symbol_stripped
    ):
        return "symbol_stripped"

    if _same_number_fuzzy_match(body_key, toc_key):
        return "same_number_fuzzy"

    if len(body_key) <= 4 or len(toc_key) <= 4:
        return None
    if _has_numbered_prefix(body_key) != _has_numbered_prefix(toc_key):
        return None
    # 只允许“正文短标题包含在目录长标题中”，不允许“目录标题藏在更长的封面/篇章列表文本里”。
    if _contains_title_phrase(toc_key, body_key):
        return "phrase"
    return None


def _find_toc_matches(body_key: str, toc_entries: list[dict],
                      used_orders: set[int]) -> list[dict]:
    """在完整目录中查找所有尚未使用的匹配项。

    作用:
        第四层先全局找候选，而不是用游标限制范围。这样第5章、第6章不会因为
        前面某个重复标题误推进游标而被跳过。

    输入:
        body_key: 正文候选标题 key。
        toc_entries: 目录条目列表，每项含 key/text/level/order/ancestors。
        used_orders: 已经映射过的目录 order 集合。

    输出:
        list[dict]，所有可能匹配项，每项额外带 match_kind。
    """
    matches = []
    for entry in toc_entries:
        if entry["order"] in used_orders:
            continue
        kind = _toc_match_kind(body_key, entry["key"])
        if kind:
            item = dict(entry)
            item["match_kind"] = kind
            matches.append(item)
    return matches


def _nearest_active_parent(active_stack: list[dict], target_level: int) -> dict | None:
    """找到当前正文结构中低于目标标题层级的最近父级标题。

    输入:
        active_stack: 已确认正文标题路径，例如 [第11章, 11.2]。
        target_level: 当前候选目录项等级，例如 4。

    输出:
        dict | None，最近父级。例如 target_level=4 时返回 11.2。
    """
    parents = [e for e in active_stack if int(e.get("level", 9)) < target_level]
    if not parents:
        return None
    return max(parents, key=lambda e: int(e.get("level", 0)))


def _entry_has_ancestor(entry: dict, ancestor: dict) -> bool:
    """判断目录项是否位于指定父级目录项之下。"""
    ancestor_order = ancestor.get("order")
    return any(a.get("order") == ancestor_order for a in entry.get("ancestors", []))


def _score_toc_candidate(body_key: str, entry: dict,
                         active_stack: list[dict]) -> tuple[int, bool, str]:
    """给一个全局目录候选打分，并判断是否与当前正文父级冲突。

    输入:
        body_key: 正文候选标题 key。
        entry: 一个目录候选项。
        active_stack: 当前正文已经确认的标题路径。

    输出:
        (score, conflict, reason)
        conflict=True 表示当前正文已经在某章/节内，但该目录候选不属于这个父级。
    """
    kind_score = {
        "exact": 300,
        "stripped": 240,
        "number_stripped": 220,
        "same_number_fuzzy": 215,
        "chapter_fuzzy": 210,
        "symbol_stripped": 205,
        "phrase": 180,
    }
    score = kind_score.get(entry.get("match_kind"), 0)
    target_level = int(entry.get("level") or 6)

    if _has_numbered_prefix(body_key) and _has_numbered_prefix(entry.get("key", "")):
        score += 160

    if target_level <= 2:
        # 如果该标题在目录中重复出现 (count > 1)，即使 level ≤ 2 也需要
        # 走父级消歧，防止把第3章的"思考题"错配到第1章。
        if entry.get("count", 1) <= 1:
            return score, False, "顶层标题无需父级消歧"
        # count > 1: 继续往下走父级消歧

    parent = _nearest_active_parent(active_stack, target_level)
    if not parent:
        return score, False, "当前没有已确认父级"

    if _entry_has_ancestor(entry, parent):
        ancestor_orders = {a.get("order") for a in entry.get("ancestors", [])}
        matched_depth = sum(1 for a in active_stack if a.get("order") in ancestor_orders)
        score += 500 + matched_depth * 80
        return score, False, f"父级路径匹配: {parent.get('text', '')[:30]}"

    score -= 500
    return score, True, f"父级路径冲突: 当前在 {parent.get('text', '')[:30]} 下"


def _choose_toc_match(body_key: str, toc_entries: list[dict],
                      used_orders: set[int],
                      active_stack: list[dict]) -> tuple[dict | None, str]:
    """从全局匹配候选中选择最合理的目录项。

    核心原则:
        1. 不用全局游标，避免被前面的重复标题带偏。
        2. 如果标题在目录中重复，必须看当前正文已经进入的章/节路径。
        3. 如果当前路径和目录候选父级冲突，就把正文候选降为普通文本。

    输入:
        body_key: 正文标题 key。
        toc_entries: 全部目录条目。
        used_orders: 已使用目录项 order。
        active_stack: 当前正文标题路径。

    输出:
        (match, reason)，match 为 None 时表示不应保留为标题。
    """
    matches = _find_toc_matches(body_key, toc_entries, used_orders)
    if not matches:
        return None, "未在目录中出现"

    scored = []
    for entry in matches:
        score, conflict, reason = _score_toc_candidate(body_key, entry, active_stack)
        scored.append((score, conflict, reason, entry))
    scored.sort(key=lambda x: (x[0], -int(x[3].get("order", 0))), reverse=True)

    best_score, best_conflict, best_reason, best = scored[0]
    best_level = int(best.get("level") or 6)
    numbered = _has_numbered_prefix(body_key) and _has_numbered_prefix(best.get("key", ""))

    if best_conflict and not numbered:
        return None, best_reason

    parent = _nearest_active_parent(active_stack, best_level)
    if best_level > 2 and not parent and not numbered:
        if len(matches) > 1:
            return None, "重复标题缺少正文父级路径，无法判断属于哪一章/节"
        if len(body_key) <= 4:
            return None, "短标题缺少正文父级路径，容易误匹配"

    top = [x for x in scored if x[0] == best_score]
    if len(top) > 1 and best_level > 2 and not numbered:
        return None, "多个目录候选同分，无法可靠消歧"

    best["match_reason"] = best_reason
    return best, best_reason


def _entry_parent_order(entry: dict) -> int | None:
    """返回目录项的直接父级 order。"""
    ancestors = entry.get("ancestors") or []
    if not ancestors:
        return None
    try:
        return int(ancestors[-1].get("order"))
    except Exception:
        return None


def _same_toc_parent(a: dict, b: dict) -> bool:
    """判断两个目录项是否有同一个直接父级。"""
    return _entry_parent_order(a) == _entry_parent_order(b)


def _direct_child_toc_entries(entry: dict, toc_entries: list[dict],
                              max_children: int = 8) -> list[dict]:
    """取目录项的直接子标题，用于判断正文锚点后方是否展开本章内容。"""
    order = int(entry.get("order", -1))
    children = []
    for item in toc_entries:
        ancestors = item.get("ancestors") or []
        if not ancestors:
            continue
        try:
            parent_order = int(ancestors[-1].get("order"))
        except Exception:
            continue
        if parent_order == order:
            children.append(item)
            if len(children) >= max_children:
                break
    return children


def _format_line_context(md_lines: list[str], center_line: int,
                         before: int = 8, after: int = 36) -> str:
    """把某个原文行附近的上下文压缩成 LLM 可读片段。"""
    if not md_lines or center_line <= 0:
        return ""
    start = max(1, center_line - before)
    end = min(len(md_lines), center_line + after)
    out = []
    for line_no in range(start, end + 1):
        text = compact_text(md_lines[line_no - 1], 180)
        out.append(f"{line_no}: {text}")
    return "\n".join(out)


def _find_preview_sibling_matches(all_headings: list[dict], idx: int,
                                  entry: dict, toc_entries: list[dict],
                                  used_orders: set[int]) -> list[dict]:
    """查找当前位置后方密集出现的同父级兄弟标题。

    这种结构常见于“本部分包括以下章节”的导读/预告块:
        # 第一章 简介
        一句简介
        # 第二章 关键数据结构
        一句简介
        # 第三章 ...
    """
    siblings = []
    current_line = int(all_headings[idx].get("line") or 0)
    target_level = int(entry.get("level") or 6)
    current_order = int(entry.get("order") or -1)
    blocked_orders = set(used_orders) | {current_order}

    for next_h in all_headings[idx + 1:idx + 8]:
        next_line = int(next_h.get("line") or 0)
        if not next_line or next_line - current_line > LAYER4_PREVIEW_SIBLING_LINE_SPAN:
            break
        if _is_noise_heading_text(next_h.get("raw", "") or next_h.get("text", "")):
            continue

        next_key = title_key(next_h.get("text", ""))
        matches = _find_toc_matches(next_key, toc_entries, blocked_orders)
        best = None
        for m in matches:
            order_delta = int(m.get("order") or -1) - current_order
            if (
                int(m.get("level") or 6) == target_level
                and _same_toc_parent(m, entry)
                and 0 < order_delta <= 60
            ):
                best = m
                break
        if best:
            siblings.append({
                "line": next_line,
                "body_text": next_h.get("text", ""),
                "toc_order": int(best.get("order")),
                "toc_text": best.get("text", ""),
            })
    return siblings


def _find_later_same_toc_candidates(all_headings: list[dict], idx: int,
                                    entry: dict, toc_entries: list[dict],
                                    used_orders: set[int],
                                    max_candidates: int = 3) -> list[dict]:
    """查找后方是否还有同一个目录标题的候选位置。"""
    current_line = int(all_headings[idx].get("line") or 0)
    target_order = int(entry.get("order") or -1)
    candidates = []
    for next_h in all_headings[idx + 1:]:
        next_line = int(next_h.get("line") or 0)
        if not next_line:
            continue
        if next_line - current_line > LAYER4_PREVIEW_LATER_LINE_SPAN:
            break
        if _is_noise_heading_text(next_h.get("raw", "") or next_h.get("text", "")):
            continue
        next_key = title_key(next_h.get("text", ""))
        matches = _find_toc_matches(next_key, toc_entries, used_orders)
        if any(int(m.get("order") or -1) == target_order for m in matches):
            candidates.append({
                "line": next_line,
                "body_text": next_h.get("text", ""),
            })
            if len(candidates) >= max_candidates:
                break
    return candidates


def _choose_deferred_preview_match(body_key: str,
                                   toc_entries: list[dict],
                                   used_orders: set[int],
                                   deferred_orders: dict[int, str]) -> tuple[dict | None, str]:
    """导读块被拒绝后，允许后方省略章号的真实标题重新接回对应目录项。"""
    if not deferred_orders:
        return None, ""
    for order, reject_reason in list(deferred_orders.items()):
        if order in used_orders:
            continue
        entry = next((x for x in toc_entries if int(x.get("order") or -1) == order), None)
        if not entry:
            continue
        kind = _toc_match_kind(body_key, entry.get("key", ""))
        if not kind:
            continue
        item = dict(entry)
        item["match_kind"] = kind
        return item, f"导读块后续真实候选接回: {reject_reason[:80]}"
    return None, ""


def _llm_accepts_preview_like_anchor(current: dict, entry: dict,
                                     sibling_matches: list[dict],
                                     later_candidates: list[dict],
                                     child_entries: list[dict],
                                     md_lines: list[str]) -> tuple[bool, str, int | None]:
    """让 LLM 判断疑似导读/预告标题是否是真正文锚点。

    返回:
        (accept, reason, better_line)。accept=False 表示应降为普通正文。
    """
    sibling_text = "\n".join(
        f"- line {x['line']}: 正文候选《{compact_text(x['body_text'], 80)}》"
        f" -> 目录 order={x['toc_order']}《{compact_text(x['toc_text'], 80)}》"
        for x in sibling_matches
    ) or "无"
    later_text = "\n".join(
        f"- line {x['line']}: 《{compact_text(x['body_text'], 100)}》"
        for x in later_candidates
    ) or "无"
    child_text = "\n".join(
        f"- order={x.get('order')} H{x.get('level')}: {compact_text(x.get('text', ''), 100)}"
        for x in child_entries
    ) or "无"
    context_text = _format_line_context(md_lines, int(current.get("line") or 0))
    expected = (
        '{"is_real_body_anchor": true/false, "better_candidate_line": 行号或null, '
        '"confidence": 0.0, "reason": "一句话"}'
    )
    prompt = f"""禁止分析过程。只返回严格 JSON。

你是第四层正文标题锚点判定器。目录标题已经由前面层级确定，不能修改目录；你只判断“当前正文候选位置”是不是该目录标题的真正正文起点。

重要背景:
- 正文中可能先出现“导读/预告/本部分包括以下章节”的章节列表。
- 这类导读标题虽然文本能匹配目录，但不是正文真实锚点。
- 真正文锚点通常会在后面展开本章内容，并逐步出现本章子标题。
- 不要因为当前标题文本更完整就优先选择；正文展开证据更重要。
- 如果当前位置后面密集列出多个同级兄弟章节，且每个只跟简短介绍，通常是导读/预告，应判为 false。
- 如果标题被 OCR 拆成“# 第一章”+“# 简介”，合并后的后续候选也可能才是真正文锚点。

典型例子:
导读/预告块，不是真正文锚点:
```
# 第一章 简介
本章介绍网络代码中常见的模式。

# 第二章 关键数据结构
本章介绍 sk_buff 和 net_device。

# 第三章 用户空间与内核的接口
本章介绍 procfs、ioctl、Netlink。
```
原因: 一个标题后面没有展开本章内容，而是马上连续出现多个同级章节标题。这是在介绍“本部分包括哪些章”。

真正正文锚点:
```
# 第一章
# 简介
这里开始展开第一章正文内容。

# 基本术语
这里开始解释第一章的子标题。
```
原因: 标题后面进入正文段落，随后出现的是该章子标题，而不是一串同级章节标题。

判断准则:
- 正文标题后面一般接正文段落，或者接该章/该节的下一级子标题。
- 如果标题后面是一群同级标题，每个同级标题后面只有一句简介，那就是导读/预告。
- 如果后面再次出现同一标题或省略章号后的标题，并且后方开始正文展开，后面的候选更可信。

当前目录标题:
order={entry.get('order')} H{entry.get('level')} 《{compact_text(entry.get('text', ''), 120)}》

当前候选:
line={current.get('line')} 《{compact_text(current.get('text', ''), 120)}》

该目录标题的直接子标题:
{child_text}

当前候选后方密集出现的同父级兄弟候选:
{sibling_text}

后方同一目录标题的其他候选:
{later_text}

当前候选附近原文:
{context_text}

返回 JSON:
{expected}"""
    messages = [
        {
            "role": "system",
            "content": "你是正文标题锚点判定器。禁止推理过程，只能输出严格 JSON。/no_think",
        },
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    try:
        result = call_llm_json_with_repair(
            messages,
            max_tokens=1000,
            stage="layer4_preview_anchor_judge",
            expected_format=expected,
            timeout=TIMEOUT,
        )
    except Exception as exc:
        return False, f"疑似导读块，LLM 判断失败，保守降为正文: {exc}", None

    accept = bool(result.get("is_real_body_anchor"))
    confidence = result.get("confidence")
    reason = str(result.get("reason") or "").strip()
    better_line = result.get("better_candidate_line")
    try:
        better_line = int(better_line) if better_line is not None else None
    except Exception:
        better_line = None
    detail = f"LLM导读判断 confidence={confidence}, better_line={better_line}, reason={reason}"
    return accept, detail, better_line


def _update_active_stack(active_stack: list[dict], entry: dict) -> list[dict]:
    """用新匹配到的目录项更新正文标题路径。

    输入:
        active_stack: 当前路径，例如 [第11章, 11.2]。
        entry: 新确认标题，例如 11.2.3。

    输出:
        list[dict]，更新后的路径 [第11章, 11.2, 11.2.3]。
    """
    level = int(entry.get("level") or 6)
    return [e for e in active_stack if int(e.get("level") or 6) < level] + [entry]


def layer4_apply(all_headings: list[dict], toc_headings: list[dict],
                 toc_range: dict, md_text: str = "") -> list[dict]:
    """第四层: 基于目录结构全局映射所有 `#` 候选标题。

    原理:
        目录结构是准绳，正文区只有能匹配目录的 `#` 候选才保留为标题。
        目录区域本身的 `#` 条目全部降为正文，避免目录污染标题树。

    重复标题:
        目录中同名标题可能出现多次，例如多个章节都叫 "小结" 或 "案例分析"。
        第四层不会用固定词表补丁，而是比较当前正文已确认的父级路径和目录项父级路径:
        正文已经在 "第11章 > 11.2" 下时，只接受同样位于该路径下的目录候选。

    输入:
        all_headings: list[dict]，extract_headings 输出，正文中所有 `#` 候选。
        toc_headings: list[dict]，第三层校验后的目录标题，例如:
            [{"text": "第1章 概述", "level": 2}, {"text": "1.1 背景", "level": 3}]
        toc_range: dict，目录范围，例如:
            {"toc_marker_line": 67, "toc_start_line": 67, "toc_end_line": 480, "found": True}

    输出:
        list[dict]，每个候选标题的最终决策:
            {
              "line": 505,
              "raw": "# 第1章 概述",
              "text": "第1章 概述",
              "is_heading": True,
              "level": 2,
              "role": "section",
              "reason": "匹配目录: 第1章 概述"
            }

    例子:
        目录区域内 "# 第1章 概述" -> is_heading=False, role="toc_entry"
        正文区域 "# 第1章 概述" -> is_heading=True, level=2
        正文区域 "# 图 F1.1" 且目录无该项 -> is_heading=False
    """
    # 构建目录条目列表，并为每个目录项记录父级路径。
    # 第四层会全局查找候选，再用父级路径消歧重复标题。
    toc_entries = []
    toc_counts = {}
    toc_stack = []
    for order, h in enumerate(toc_headings):
        key = title_key(h.get("text", ""))
        if key:
            level = int(h.get("level") or 6)
            while toc_stack and int(toc_stack[-1].get("level") or 6) >= level:
                toc_stack.pop()
            entry = {
                "key": key,
                "level": level,
                "text": h["text"],
                "order": order,
                "ancestors": [
                    {
                        "key": p["key"],
                        "level": p["level"],
                        "text": p["text"],
                        "order": p["order"],
                    }
                    for p in toc_stack
                ],
                "count": 1,
            }
            toc_entries.append(entry)
            toc_counts[key] = toc_counts.get(key, 0) + 1
            toc_stack.append(entry)
    for entry in toc_entries:
        entry["count"] = toc_counts.get(entry["key"], 1)

    toc_start = toc_range.get("toc_start_line", 0)
    toc_end = toc_range.get("toc_end_line", 0)
    toc_marker_line = toc_range.get("toc_marker_line", 0)

    decisions = []
    matched = 0
    fuzzy_matched = 0
    unmatched = 0
    used_orders = set()
    active_stack = []
    split_subtitle_lines = set()  # 被合并到上一行的 # 标题，后续降为正文
    preview_rejected_lines: set[int] = set()
    preview_reject_reasons: dict[int, str] = {}
    preview_deferred_orders: dict[int, str] = {}
    preview_forced_lines: dict[int, dict] = {}
    preview_llm_calls = 0

    # ================================================================
    # 步骤 0: 预处理 — 修复正文中被 OCR 截断的标题
    # ================================================================
    _body_start_re = re.compile(
        r"^(This chapter|In this chapter|This section|In this section"
        r"|本章|本节|本文|这一章|这一节)",
        re.IGNORECASE,
    )
    _body_lines = md_text.split("\n") if md_text else []

    for idx, h in enumerate(all_headings):
        if toc_start <= h["line"] <= toc_end:
            continue
        if not _is_bare_chapter_title(h["text"]):
            continue

        merged_text = None
        merged_next_line = None
        merged_plain_line = None

        # 尝试 A: 下一个 # 标题 (如 "# 第9章" + "# 中断管理")
        if idx + 1 < len(all_headings):
            next_h = all_headings[idx + 1]
            nline = next_h["line"]
            if not (toc_start <= nline <= toc_end):
                # 计数两行之间的非空行
                gap_non_empty = 0
                if _body_lines:
                    for li in range(h["line"], nline - 1):
                        if li < len(_body_lines) and _body_lines[li].strip():
                            gap_non_empty += 1
                if gap_non_empty <= 5:
                    merged_text = f"{h['text']} {next_h['text']}"
                    merged_next_line = nline

        # 尝试 B: 纯文本续文 (如 "# 第20章" + "负载均衡集群 LVS 与 HAProxy")
        if merged_text is None:
            raw_next = h.get("next_text", "")
            first_line = ""
            for line_text in raw_next.split("\n")[:3]:
                c = line_text.strip()
                if c:
                    first_line = c
                    break
            if first_line and len(first_line) <= 80 \
                    and not re.match(r"^[#!<\{]", first_line) \
                    and not re.search(r"[。！？；;:.]$", first_line) \
                    and not re.search(r"[…\.]{2,}\s*\d+$", first_line) \
                    and not _body_start_re.match(first_line):
                merged_text = f"{h['text']} {first_line}"
                # 从原始文本中定位续文行号（next_text 被 strip 过，行号不准）
                if _body_lines:
                    for li in range(h["line"], min(h["line"] + 5, len(_body_lines))):
                        if _body_lines[li].strip() == first_line:
                            merged_plain_line = li + 1  # 1-indexed
                            break

        if merged_text:
            h["text"] = merged_text
            h["_original_text"] = h.get("_original_text") or all_headings[idx]["text"]
            if merged_next_line:
                split_subtitle_lines.add(merged_next_line)
                h["_skip_next_line"] = merged_next_line  # Case A: 第二行从输出中移除
            if merged_plain_line:
                h["_merged_plain_line"] = merged_plain_line

    # ================================================================
    # Phase 2: 逐条判定
    #   Tier 1: title_key 精确匹配目录 → 直接得等级
    #   Tier 2: 未命中 → 噪声过滤 → LLM 兜底判断
    # ================================================================
    for idx, h in enumerate(all_headings):
        line = h["line"]
        key = title_key(h["text"])

        d = {"line": line, "raw": h["raw"], "text": h["text"],
             "is_heading": False, "level": None, "role": "paragraph", "reason": "",
             "_merged_plain_line": h.get("_merged_plain_line"),
             "_skip_next_line": h.get("_skip_next_line"),
             "inside_code_block": bool(h.get("inside_code_block")),
             "code_toc_match_kind": h.get("code_toc_match_kind")}

        # 规则N: OCR 页眉/书名噪声，直接丢弃
        if _is_noise_heading_text(h.get("raw", "") or h.get("text", "")):
            d.update(role="noise", reason="OCR 页眉/书名噪声 → 丢弃")
            decisions.append(d); continue

        # 规则0: 目录标记之前 → 降为正文（封面、前言、篇目概览等非正文）
        if line < toc_marker_line:
            d.update(reason="目录前内容 → 降为正文")
            decisions.append(d); continue

        # 规则1: 目录标记
        if line == toc_marker_line:
            d.update(is_heading=True, level=2, role="toc_marker",
                     reason="目录标记 → H2")
            decisions.append(d); continue

        # 规则2: 目录区域内 → 降为正文
        if toc_start <= line <= toc_end:
            d.update(role="toc_entry", reason="目录区域内 → 降为正文")
            decisions.append(d); continue

        # 规则3a: 被合并到上一行的副标题 → 降为正文
        if line in split_subtitle_lines:
            d.update(reason="章节标题拆分后的副标题已合并到上一行 → 降为正文")
            decisions.append(d); continue

        # 规则3b: 同一导读/预告块中已被 LLM 判定为假锚点的兄弟标题
        if line in preview_rejected_lines:
            d.update(
                role="preview_entry",
                reason=preview_reject_reasons.get(line, "同一导读/预告块标题 → 降为正文"),
            )
            unmatched += 1
            decisions.append(d); continue

        # 规则3: 正文区 → 全局匹配目录
        #   尝试0: 原标题 key
        #   尝试1: 合并下一非空行
        #   尝试2: 合并下两非空行
        match, match_reason = _choose_toc_match(key, toc_entries, used_orders, active_stack)
        _merged_lines = None   # 被合并的续文行号列表

        if line in preview_forced_lines and int(preview_forced_lines[line].get("order") or -1) not in used_orders:
            match = dict(preview_forced_lines[line])
            match_reason = "LLM 指定为导读块后的更可信正文锚点"

        if not match:
            raw_next = h.get("next_text", "")
            # 提取 next_text 中前几个非空行（纯文本内容）
            _non_empty = []
            text_lines = raw_next.split("\n")
            for lt in text_lines[:5]:
                c = lt.strip()
                if c and not re.match(r"^[#!<\{]", c) and len(c) <= 80:
                    _non_empty.append(c)
                if len(_non_empty) >= 3:
                    break

            if _non_empty:
                for _take in (1, 2):
                    if len(_non_empty) >= _take:
                        _combined = title_key(h["text"] + " " + " ".join(_non_empty[:_take]))
                        match, match_reason = _choose_toc_match(
                            _combined, toc_entries, used_orders, active_stack
                        )
                        if match:
                            h["text"] = h["text"] + " " + " ".join(_non_empty[:_take])
                            d["text"] = h["text"]
                            # 从原始文本定位续文行号（next_text 被 strip 过，行号不准）
                            if _body_lines:
                                _merged_lines = []
                                for _ct in _non_empty[:_take]:
                                    for li in range(h["line"],
                                                    min(h["line"] + 6, len(_body_lines))):
                                        if _body_lines[li].strip() == _ct:
                                            _merged_lines.append(li + 1)
                                            break
                            break

        if not match:
            deferred_match, deferred_reason = _choose_deferred_preview_match(
                title_key(h["text"]),
                toc_entries,
                used_orders,
                preview_deferred_orders,
            )
            if deferred_match:
                match = deferred_match
                match_reason = deferred_reason

        if match:
            sibling_matches = _find_preview_sibling_matches(
                all_headings,
                idx,
                match,
                toc_entries,
                used_orders,
            )
            later_candidates = _find_later_same_toc_candidates(
                all_headings,
                idx,
                match,
                toc_entries,
                used_orders,
            )
            preview_level = int(match.get("level") or 6)
            preview_structure = _toc_structural_marker(title_key(match.get("text", "")))
            preview_allowed = preview_level <= 3 or preview_structure is not None
            preview_suspicious = bool(
                preview_allowed
                and sibling_matches
                and (len(sibling_matches) >= 2 or later_candidates)
            )
            if preview_suspicious:
                child_entries = _direct_child_toc_entries(match, toc_entries)
                if preview_llm_calls < LAYER4_PREVIEW_LLM_MAX_CALLS:
                    preview_llm_calls += 1
                    accept_anchor, preview_reason, better_line = _llm_accepts_preview_like_anchor(
                        h,
                        match,
                        sibling_matches,
                        later_candidates,
                        child_entries,
                        _body_lines,
                    )
                else:
                    accept_anchor = False
                    better_line = None
                    preview_reason = "第四层导读判断次数达到上限，疑似导读块保守降为正文"

                if not accept_anchor:
                    rejected = [line] + [int(x["line"]) for x in sibling_matches]
                    for rejected_line in rejected:
                        preview_rejected_lines.add(rejected_line)
                        preview_reject_reasons[rejected_line] = (
                            "疑似章节导读/预告假锚点 → 降为正文；" + preview_reason
                        )
                    reject_reason_short = preview_reason[:120]
                    preview_deferred_orders[int(match["order"])] = reject_reason_short
                    for x in sibling_matches:
                        preview_deferred_orders[int(x["toc_order"])] = reject_reason_short
                    if better_line:
                        preview_forced_lines[better_line] = dict(match)
                    d.update(
                        role="preview_entry",
                        reason=preview_reject_reasons[line],
                    )
                    unmatched += 1
                    decisions.append(d); continue

            match_key = match.get("key", key)
            used_orders.add(int(match["order"]))
            preview_deferred_orders.pop(int(match["order"]), None)
            active_stack = _update_active_stack(active_stack, match)
            tag = "合并匹配" if _merged_lines else "全局匹配目录"
            d.update(is_heading=True, level=match["level"], role="section",
                     reason=f"{tag}: " + match["text"][:40] + f" ({match_reason})",
                     toc_order=int(match["order"]),
                     toc_text=match["text"],
                     toc_key=match.get("key"),
                     match_kind=match.get("match_kind"),
                     toc_ancestors=[
                         {
                             "order": int(a.get("order")),
                             "level": int(a.get("level")),
                             "text": a.get("text", ""),
                             "key": a.get("key", ""),
                         }
                         for a in match.get("ancestors", [])
                     ])
            if h.get("inside_code_block"):
                d["_close_bad_fence_before"] = True
                d["reason"] = "代码块中匹配目录标题，输出前关闭坏代码块；" + d["reason"]
            if match.get("text") and match_key != key:
                d["normalized_title"] = display_title(match["text"])
                fuzzy_matched += 1
            else:
                matched += 1
            # 记录被合并的续文行，输出渲染时会跳过它们。
            if _merged_lines:
                for _ml in _merged_lines:
                    if not d.get("_merged_plain_line"):
                        d["_merged_plain_line"] = _ml
                    elif not d.get("_merged_plain_line2"):
                        d["_merged_plain_line2"] = _ml
        else:
            d.update(reason=f"{match_reason} → 降为正文")
            unmatched += 1

        decisions.append(d)

    print(f"  第四层映射: {matched} 精确匹配, {fuzzy_matched} 模糊匹配, {unmatched} 未匹配")
    if preview_llm_calls or preview_rejected_lines:
        print(
            f"  第四层导读过滤: LLM 判断 {preview_llm_calls} 次, "
            f"降为正文 {len(preview_rejected_lines)} 个候选"
        )
    return decisions


# ============================================================
# 应用判定


# ============================================================
# 第五层: 目录树挂载与缺失父级补齐
# ============================================================

def _chinese_num_to_int(text: str) -> int | None:
    """把简单中文数字转成整数，用于目录编号路径对齐。"""
    s = (text or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    digit = {
        "零": 0, "〇": 0,
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    unit = {"十": 10, "百": 100, "千": 1000}
    total = 0
    num = 0
    seen = False
    for ch in s:
        if ch in digit:
            num = digit[ch]
            seen = True
        elif ch in unit:
            seen = True
            if num == 0:
                num = 1
            total += num * unit[ch]
            num = 0
        else:
            return None
    return total + num if seen else None


def _toc_number_path(text: str) -> str:
    """提取目录/正文标题的结构编号路径，如 10、10.1、A、A.1。"""
    key = title_key(text)
    if not key:
        return ""

    m = re.match(r"^第([一二两三四五六七八九十百千零〇0-9]+)[章节篇卷部]", key)
    if m:
        n = _chinese_num_to_int(m.group(1))
        return str(n) if n is not None else m.group(1)

    m = re.match(r"^(?:附录|appendix)([a-z])(?:\.(\d+(?:\.\d+)*))?", key, flags=re.IGNORECASE)
    if m:
        appendix_no = m.group(1).upper()
        return f"{appendix_no}.{m.group(2)}" if m.group(2) else appendix_no

    m = re.match(r"^(?:chapter|chap)\s*(\d+)", key, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.match(r"^(?:section|sec|subsection|subsec)\s*(\d+(?:\.\d+)*)", key, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.match(r"^(?:part|book|volume|vol)\s*(\d+|[ivxlcdm]+)", key, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.match(r"^([a-z])\.(\d+(?:\.\d+)*)", key, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}.{m.group(2)}"

    m = re.match(r"^(\d+(?:\.\d+)*)", key)
    if m:
        return m.group(1)

    return ""


def _build_toc_tree_entries(toc_headings: list[dict]) -> list[dict]:
    """用目录标题列表构建带父子关系的扁平目录树。"""
    entries = []
    stack = []

    for order, h in enumerate(toc_headings):
        text = h.get("text", "")
        level = int(h.get("level") or 2)
        while stack and int(stack[-1].get("level") or 9) >= level:
            stack.pop()

        parent = stack[-1] if stack else None
        entry = {
            "order": order,
            "text": text,
            "key": title_key(text),
            "level": level,
            "number_path": _toc_number_path(text),
            "parent_order": parent["order"] if parent else None,
            "children": [],
            "ancestors": [
                {
                    "order": int(a["order"]),
                    "text": a["text"],
                    "key": a["key"],
                    "level": int(a["level"]),
                    "number_path": a.get("number_path", ""),
                }
                for a in stack
            ],
        }
        entries.append(entry)
        if parent:
            parent["children"].append(order)
        stack.append(entry)

    return entries


def _is_toc_descendant(entry: dict, ancestor_order: int) -> bool:
    """判断目录 entry 是否位于指定目录节点之下。"""
    return any(int(a.get("order", -1)) == ancestor_order for a in entry.get("ancestors", []))


def _body_records_between(body_lines: list[dict], start_line: int,
                          end_line: int, max_records: int = 220) -> list[dict]:
    """收集正文窗口，保留原始行号，过长时保留前部和尾部。"""
    records = [
        rec for rec in body_lines
        if rec.get("line") and start_line <= int(rec["line"]) <= end_line
           and rec.get("text", "").strip()
    ]
    if len(records) <= max_records:
        return records
    head = max_records - 40
    return records[:head] + records[-40:]


def _format_layer5_window(records: list[dict], max_chars_per_line: int = 180) -> str:
    """把正文窗口格式化成带行号的 LLM 输入。"""
    lines = []
    for rec in records:
        text = rec.get("text", "").strip()
        if len(text) > max_chars_per_line:
            text = text[:max_chars_per_line].rstrip()
        lines.append(f"line={rec['line']}: {text}")
    return "\n".join(lines)


def _first_prefix_evidence_line(records: list[dict], number_path: str) -> int | None:
    """在正文窗口中找能提示某目录编号存在的第一行。"""
    path = (number_path or "").strip()
    if not path:
        return None
    path_l = path.lower()

    for rec in records:
        raw = rec.get("text", "").strip()
        if not raw:
            continue
        text = re.sub(r"^#{1,6}\s+", "", raw).strip()
        if len(text) > 100 and not raw.startswith("#"):
            continue
        key = title_key(text)
        if not key:
            continue

        if "." in path_l:
            if key.startswith(path_l):
                return int(rec["line"])
            continue

        if path_l.isdigit():
            if re.match(rf"^{re.escape(path_l)}\.\d+", key):
                return int(rec["line"])
            continue

        if len(path_l) == 1 and path_l.isalpha():
            if re.match(rf"^{re.escape(path_l)}\.\d+", key):
                return int(rec["line"])
            if key.startswith(f"附录{path_l}") or key.startswith(f"appendix{path_l}"):
                return int(rec["line"])

    return None


def _nearest_anchor(order: int, anchors: list[dict], before: bool) -> dict | None:
    """按目录顺序找目标节点前/后的最近正文锚点。"""
    if before:
        candidates = [a for a in anchors if int(a["toc_order"]) < order]
        return max(candidates, key=lambda x: int(x["toc_order"])) if candidates else None
    candidates = [a for a in anchors if int(a["toc_order"]) > order]
    return min(candidates, key=lambda x: int(x["toc_order"])) if candidates else None


def _first_descendant_anchor(entry: dict, anchors: list[dict],
                             entries_by_order: dict[int, dict]) -> dict | None:
    """找某个缺失目录节点下第一个已经在正文中定位到的子节点。"""
    descendants = []
    for anchor in anchors:
        child = entries_by_order.get(int(anchor["toc_order"]))
        if child and _is_toc_descendant(child, int(entry["order"])):
            descendants.append(anchor)
    return min(descendants, key=lambda x: int(x["line"])) if descendants else None


def _looks_like_forward_reference_to_entry(text: str, entry: dict) -> bool:
    """判断一行是否只是上一节对目标章节的前瞻引用。"""
    raw = (text or "").strip()
    if not raw:
        return False
    key = title_key(raw)
    path = (entry.get("number_path") or "").strip()
    if not path:
        return False

    has_target_ref = False
    if path.isdigit():
        has_target_ref = bool(
            re.search(rf"第\s*{re.escape(path)}\s*[章节篇卷部]", raw)
            or re.search(rf"第\s*{re.escape(path)}\s*[章节篇卷部]", key)
        )
    elif len(path) == 1 and path.isalpha():
        has_target_ref = bool(
            re.search(rf"附录\s*{re.escape(path)}\b", raw, flags=re.IGNORECASE)
            or key.startswith(f"附录{path.lower()}")
        )
    if not has_target_ref:
        return False

    return bool(re.search(
        r"将|将在|将会|会在|后面|后续|下面|下一章|下章|稍后|随后|见|参见|请见|讨论|讲述|介绍|显示|说明",
        raw,
    ))


def _is_layer5_noise_boundary(text: str) -> bool:
    """第五层局部窗口里的广告/页眉页脚边界。"""
    raw = (text or "").strip()
    if not raw:
        return True
    return bool(
        _is_noise_heading_text(raw)
        or re.search(r"精品学习资料|下载汇总|考试时间|视频教程|资源索引|电子书", raw)
    )


def _starts_with_continuation_cue(text: str) -> bool:
    """判断段落是否明显承接上一段，标题应回退到上一段前。"""
    raw = re.sub(r"^#{1,6}\s+", "", text or "").strip()
    return bool(re.match(
        r"^(此外|另外|而且|同时|因此|所以|于是|接着|然后|同样|类似地|该|这|正如|在此基础上)",
        raw,
    ))


def _adjust_layer5_insert_line(entry: dict, records: list[dict],
                               insert_line: int, fallback_line: int) -> int:
    """校验第五层插入点，避免把新章标题插进上一节总结正文中。"""
    if not insert_line:
        return fallback_line

    by_line = {int(r["line"]): r for r in records}
    chosen = by_line.get(int(insert_line))
    if not chosen:
        return insert_line

    if not _looks_like_forward_reference_to_entry(chosen.get("text", ""), entry):
        if _starts_with_continuation_cue(chosen.get("text", "")):
            prev = None
            for rec in records:
                if int(rec["line"]) >= int(insert_line):
                    break
                if rec.get("text", "").strip():
                    prev = rec
            if prev and not _is_layer5_noise_boundary(prev.get("text", "")) \
                    and not _looks_like_forward_reference_to_entry(prev.get("text", ""), entry):
                return int(prev["line"])
        return insert_line

    # 如果 LLM 选中的是“将在第 N 章...”这种前瞻句，往后找真正的新段落。
    for rec in records:
        line = int(rec["line"])
        if line <= int(insert_line):
            continue
        if line > int(fallback_line):
            break
        text = rec.get("text", "").strip()
        if not text:
            continue
        if _looks_like_forward_reference_to_entry(text, entry):
            continue
        # 避免选图片说明、广告标题等明显噪声；实在没有时 fallback 到子标题。
        if _is_layer5_noise_boundary(text):
            continue
        return line

    return fallback_line


def _layer5_find_missing_start_by_llm(entry: dict, prev_anchor: dict | None,
                                      next_anchor: dict | None,
                                      records: list[dict]) -> int | None:
    """让 LLM 在局部窗口中判断缺失目录标题应该插入在哪一行前。"""
    if not records:
        return None

    valid_lines = {int(r["line"]) for r in records}
    prev_text = prev_anchor.get("toc_text", "") if prev_anchor else "无"
    next_text = next_anchor.get("toc_text", "") if next_anchor else "无"
    window = _format_layer5_window(records)

    prompt = f"""禁止分析过程。你的回复第一个字符必须是 {{，最后一个字符必须是 }}。
只输出一个 JSON 对象；不要输出解释、推理、Markdown 或代码块。

## 任务
目录树中有一个标题在正文标题里缺失。请在给定正文窗口里判断这个标题对应的正文区域从哪一行开始。

## 缺失目录标题
H{entry.get('level')}: {entry.get('text')}

## 前后已定位标题
前一个标题: {prev_text}
后一个标题: {next_text}

## 正文窗口
{window}

## 判断规则
1. start_line 表示应该在该行之前插入缺失目录标题。
2. 如果窗口开头仍是前一个标题的收尾内容，不要选它。
3. 如果某行只是上一节总结中说“将在第X章/见第X章/下一章会介绍...”，它是前瞻引用，不是第X章开头。
4. 缺失标题后面可以紧跟无 # 的章节引言段，也可以紧跟第一个编号子标题；优先选择真正新章节引言的第一行。
5. 跳过广告、资源下载页、页眉页脚、图片说明等无关内容。
6. 如果存在破损标题行、无 # 的章节引言首段、或编号子标题行，都可以作为 start_line。
7. 如果无法确认该标题的正文区域存在，返回 found=false。
8. 只返回 JSON，不要输出解释。

返回格式:
{{"found": true/false, "start_line": 行号或null, "confidence": 0到1, "reason": "一句话"}}"""

    messages = [
        {"role": "system", "content": "你是文档结构分析师。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
        {"role": "user", "content": prompt},
    ]

    try:
        result = parse_json(call_llm(messages, max_tokens=3000))
        if not result.get("found"):
            return None
        line = int(result.get("start_line") or 0)
        if line not in valid_lines:
            raise ValueError(f"第五层 LLM 返回行号不在窗口内: {result}")
        conf = float(result.get("confidence") or 0)
        if conf < 0.45:
            return None
        return line
    except Exception as e:
        print(f"  第五层 LLM 定位失败: {entry.get('text', '')[:40]}: {e}")
        return None


def layer5_mount_body_to_toc(toc_headings: list[dict], body_lines: list[dict],
                             decisions: list[dict], use_llm: bool = True,
                             max_llm_calls: int = 12) -> dict:
    """第五层: 以目录树为骨架，把正文锚点挂载到目录节点并补齐缺失父级标题。

    第四层从正文标题出发做匹配；第五层反过来从完整目录树出发检查覆盖情况。
    如果目录节点没有正文标题，但它的子节点已经出现，或局部正文存在编号证据，
    就在合适位置插入一个 synthetic heading。
    """
    entries = _build_toc_tree_entries(toc_headings)
    entries_by_order = {int(e["order"]): e for e in entries}

    anchors = []
    for d in decisions:
        if not d.get("is_heading") or d.get("role") != "section":
            continue
        if d.get("toc_order") is None:
            continue
        order = int(d["toc_order"])
        if order not in entries_by_order:
            continue
        anchors.append({
            "toc_order": order,
            "line": int(d["line"]),
            "toc_text": d.get("toc_text") or entries_by_order[order]["text"],
            "text": d.get("text", ""),
        })
    anchors.sort(key=lambda x: (int(x["toc_order"]), int(x["line"])))
    matched_orders = {int(a["toc_order"]) for a in anchors}

    candidate_orders: set[int] = set()
    candidate_reasons: dict[int, str] = {}

    # 1. 已定位子节点的缺失祖先，必须补齐。
    for anchor in anchors:
        entry = entries_by_order.get(int(anchor["toc_order"]))
        if not entry:
            continue
        for ancestor in entry.get("ancestors", []):
            order = int(ancestor["order"])
            if order not in matched_orders:
                candidate_orders.add(order)
                candidate_reasons.setdefault(
                    order,
                    f"子级标题已定位: {entry.get('text', '')[:40]}",
                )

    # 2. 目录节点本身没匹配，但相邻正文窗口里出现了同编号证据，也作为候选。
    for entry in entries:
        order = int(entry["order"])
        if order in matched_orders or order in candidate_orders:
            continue
        number_path = entry.get("number_path", "")
        if not number_path or int(entry.get("level") or 6) > 4:
            continue

        prev_anchor = _nearest_anchor(order, anchors, before=True)
        next_anchor = _nearest_anchor(order, anchors, before=False)
        if not next_anchor:
            continue
        start_line = int(prev_anchor["line"]) + 1 if prev_anchor else int(body_lines[0]["line"])
        end_line = int(next_anchor["line"]) - 1
        records = _body_records_between(body_lines, start_line, end_line)
        if _first_prefix_evidence_line(records, number_path) is not None:
            candidate_orders.add(order)
            candidate_reasons.setdefault(order, f"正文窗口出现编号证据: {number_path}")

    insertions = []
    synthetic_orders: set[int] = set()
    llm_calls = 0

    for order in sorted(candidate_orders):
        if order in matched_orders or order in synthetic_orders:
            continue
        entry = entries_by_order.get(order)
        if not entry:
            continue

        prev_anchor = _nearest_anchor(order, anchors, before=True)
        next_anchor = _nearest_anchor(order, anchors, before=False)
        first_child_anchor = _first_descendant_anchor(entry, anchors, entries_by_order)
        end_anchor = first_child_anchor or next_anchor
        if not end_anchor:
            continue

        start_line = int(prev_anchor["line"]) + 1 if prev_anchor else int(body_lines[0]["line"])
        end_line = int(end_anchor["line"]) - 1
        records = _body_records_between(body_lines, start_line, end_line)

        default_line = int(end_anchor["line"])
        evidence_line = _first_prefix_evidence_line(records, entry.get("number_path", ""))
        evidence_rec = next(
            (r for r in records if int(r.get("line", 0) or 0) == int(evidence_line or 0)),
            None,
        )
        if evidence_line is not None and not _looks_like_forward_reference_to_entry(
            evidence_rec.get("text", "") if evidence_rec else "",
            entry,
        ):
            default_line = min(default_line, evidence_line)

        insert_line = None
        source = "rule"
        if use_llm and records and llm_calls < max_llm_calls:
            llm_calls += 1
            insert_line = _layer5_find_missing_start_by_llm(entry, prev_anchor, end_anchor, records)
            if insert_line is not None:
                source = "llm"

        if insert_line is None:
            insert_line = default_line

        insert_line = _adjust_layer5_insert_line(entry, records, int(insert_line), int(default_line))

        # 不允许把缺失父级插到第一个已定位子节点之后。
        if int(insert_line) > int(end_anchor["line"]):
            insert_line = int(end_anchor["line"])

        insertions.append({
            "insert_before_line": int(insert_line),
            "toc_order": order,
            "level": int(entry.get("level") or 2),
            "text": entry.get("text", ""),
            "number_path": entry.get("number_path", ""),
            "role": "synthetic_heading",
            "source": source,
            "reason": candidate_reasons.get(order, "目录树覆盖补齐"),
        })
        synthetic_orders.add(order)

    covered_orders = matched_orders | synthetic_orders
    unresolved = [
        {
            "toc_order": int(e["order"]),
            "level": int(e.get("level") or 2),
            "text": e.get("text", ""),
            "number_path": e.get("number_path", ""),
        }
        for e in entries
        if int(e["order"]) not in covered_orders
    ]

    print(
        f"  第五层挂载: 正文锚点 {len(matched_orders)} 个, "
        f"补齐 {len(insertions)} 个目录标题, 未定位 {len(unresolved)} 个"
    )

    return {
        "synthetic_headings": sorted(
            insertions,
            key=lambda x: (int(x["insert_before_line"]), int(x["toc_order"])),
        ),
        "matched_orders": sorted(matched_orders),
        "synthetic_orders": sorted(synthetic_orders),
        "unresolved": unresolved,
        "llm_calls": llm_calls,
    }


def _layer5_collect_covered_anchors(entries: list[dict], decisions: list[dict],
                                    layer5_plan: dict) -> list[dict]:
    """收集第四层真实标题和第五层已补标题，作为后机制锚点。"""
    entries_by_order = {int(e["order"]): e for e in entries}
    anchors = []

    for d in decisions:
        if not d.get("is_heading") or d.get("role") != "section":
            continue
        if d.get("toc_order") is None:
            continue
        order = int(d["toc_order"])
        if order not in entries_by_order:
            continue
        anchors.append({
            "toc_order": order,
            "line": int(d["line"]),
            "text": d.get("toc_text") or entries_by_order[order].get("text", ""),
            "level": int(entries_by_order[order].get("level") or d.get("level") or 2),
            "source": "body",
        })

    for item in layer5_plan.get("synthetic_headings", []):
        if item.get("toc_order") is None or item.get("insert_before_line") is None:
            continue
        order = int(item["toc_order"])
        if order not in entries_by_order:
            continue
        anchors.append({
            "toc_order": order,
            "line": int(item["insert_before_line"]),
            "text": item.get("text") or entries_by_order[order].get("text", ""),
            "level": int(entries_by_order[order].get("level") or item.get("level") or 2),
            "source": item.get("source") or "synthetic",
        })

    best_by_order = {}
    for anchor in sorted(anchors, key=lambda x: (int(x["toc_order"]), int(x["line"]))):
        best_by_order.setdefault(int(anchor["toc_order"]), anchor)
    return [best_by_order[k] for k in sorted(best_by_order)]


def _layer5_entry_is_descendant(entry: dict, ancestor_order: int) -> bool:
    """判断目录 entry 是否位于 ancestor_order 下面。"""
    return any(int(a.get("order", -1)) == int(ancestor_order)
               for a in entry.get("ancestors", []))


def _layer5_post_find_bad_anchor_orders(entries: list[dict],
                                        anchors: list[dict]) -> dict[int, str]:
    """找出已覆盖但物理顺序明显错误的假锚点。"""
    entries_by_order = {int(e["order"]): e for e in entries}
    anchors_by_order = {int(a["toc_order"]): a for a in anchors}
    bad: dict[int, str] = {}

    max_line = -1
    max_anchor = None
    for anchor in sorted(anchors, key=lambda x: int(x["toc_order"])):
        order = int(anchor["toc_order"])
        line = int(anchor["line"])
        if line < max_line:
            prev_order = int(max_anchor["toc_order"]) if max_anchor else -1
            current_entry = entries_by_order.get(order)
            prev_entry = entries_by_order.get(prev_order)
            prev_marker = (
                _toc_structural_marker(title_key(prev_entry.get("text", "")))
                if prev_entry else None
            )
            if max_anchor and (
                (current_entry and _layer5_entry_is_descendant(current_entry, prev_order))
                or (prev_marker is not None and prev_marker[0] == "part")
            ):
                bad.setdefault(
                    prev_order,
                    (
                        "父级/篇章标题锚点晚于后续章节标题: "
                        f"parent line {max_line} > next line {line} ({anchor.get('text', '')})"
                    ),
                )
                max_line = line
                max_anchor = anchor
            else:
                prev_text = max_anchor.get("text", "") if max_anchor else ""
                bad.setdefault(
                    order,
                    f"目录顺序靠后但原文行号倒退: line {line} < previous line {max_line} ({prev_text})",
                )
            continue
        max_line = line
        max_anchor = anchor

    for anchor in anchors:
        order = int(anchor["toc_order"])
        if order in bad:
            continue
        entry = entries_by_order.get(order)
        if not entry:
            continue

        descendant_anchors = []
        for child_order, child_entry in entries_by_order.items():
            if child_order not in anchors_by_order:
                continue
            if _layer5_entry_is_descendant(child_entry, order):
                descendant_anchors.append(anchors_by_order[child_order])
        if not descendant_anchors:
            continue

        first_descendant = min(descendant_anchors, key=lambda x: int(x["line"]))
        current_level = int(entry.get("level") or 6)
        boundary_anchor = None
        for candidate_order in sorted(anchors_by_order):
            if candidate_order <= order:
                continue
            candidate_entry = entries_by_order.get(candidate_order)
            if not candidate_entry:
                continue
            if _layer5_entry_is_descendant(candidate_entry, order):
                continue
            if int(candidate_entry.get("level") or 6) <= current_level:
                boundary_anchor = anchors_by_order[candidate_order]
                break

        if boundary_anchor and int(boundary_anchor["line"]) < int(first_descendant["line"]):
            bad.setdefault(
                order,
                (
                    "疑似前言/概览中的提前标题: "
                    f"下一个同级/上级标题 line {boundary_anchor['line']} "
                    f"早于第一个子标题 line {first_descendant['line']}"
                ),
            )

    return bad


def _layer5_post_remove_bad_anchors(decisions: list[dict], layer5_plan: dict,
                                    bad_orders: dict[int, str]) -> None:
    """从第四层/第五层结果中剔除假锚点，让后机制重新补齐这些 toc_order。"""
    if not bad_orders:
        return

    for d in decisions:
        if not d.get("is_heading") or d.get("role") != "section":
            continue
        if d.get("toc_order") is None:
            continue
        order = int(d["toc_order"])
        if order not in bad_orders:
            continue
        old_reason = d.get("reason", "")
        d["_layer5_post_removed_toc_order"] = order
        d["_layer5_post_removed_toc_text"] = d.get("toc_text", "")
        d.update(
            is_heading=False,
            level=None,
            role="paragraph",
            reason=f"第五层后机制剔除顺序错乱假锚点: {bad_orders[order]}；{old_reason}",
        )
        for key in ("toc_order", "toc_text", "toc_key", "toc_ancestors",
                    "match_kind", "normalized_title"):
            d.pop(key, None)

    kept = []
    removed = []
    for item in layer5_plan.get("synthetic_headings", []):
        order = int(item.get("toc_order", -1))
        if order in bad_orders:
            removed.append({
                **item,
                "remove_reason": bad_orders[order],
            })
            continue
        kept.append(item)
    if removed:
        layer5_plan["synthetic_headings"] = kept
        layer5_plan.setdefault("post_removed_synthetic", []).extend(removed)


def _layer5_anchor_unreliable_message(entries_by_order: dict[int, dict],
                                      anchors: list[dict],
                                      bad_orders: dict[int, str]) -> str | None:
    """当正文锚点大面积错乱时返回熔断原因，否则返回 None。"""
    if not bad_orders:
        return None

    all_anchor_orders = {int(a["toc_order"]) for a in anchors}
    body_anchor_orders = {
        int(a["toc_order"]) for a in anchors
        if a.get("source") == "body"
    }
    bad_all_orders = set(int(o) for o in bad_orders if int(o) in all_anchor_orders)
    bad_body_orders = bad_all_orders & body_anchor_orders

    all_ratio = len(bad_all_orders) / max(1, len(all_anchor_orders))
    body_ratio = len(bad_body_orders) / max(1, len(body_anchor_orders))
    too_many_bad_body = (
        len(bad_body_orders) >= LAYER5_POST_MIN_BAD_ANCHORS_TO_SKIP
        and body_ratio >= LAYER5_POST_MAX_BAD_ANCHOR_RATIO
    )
    too_many_bad_all = (
        len(bad_all_orders) >= LAYER5_POST_MIN_BAD_ANCHORS_TO_SKIP * 2
        and all_ratio >= LAYER5_POST_MAX_BAD_ANCHOR_RATIO
    )
    if not (too_many_bad_body or too_many_bad_all):
        return None

    sample = "、".join(
        f"{order}:{entries_by_order.get(order, {}).get('text', '')[:24]}"
        for order in sorted(bad_orders)[:8]
    )
    return (
        "第五层后机制检测到正文锚点大面积错乱，停止 LLM 硬补: "
        f"真实正文锚点 {len(body_anchor_orders)} 个，其中错乱 {len(bad_body_orders)} 个 "
        f"({body_ratio:.0%}); 全部锚点 {len(all_anchor_orders)} 个，其中错乱 "
        f"{len(bad_all_orders)} 个 ({all_ratio:.0%}); 示例: {sample}"
    )


def _layer5_large_fill_unreliable_message(entries: list[dict],
                                          anchors: list[dict],
                                          groups: list[list[int]]) -> str | None:
    """缺失标题过多且真实正文锚点不足时，拒绝让 LLM 大规模硬补。"""
    missing_count = sum(len(g) for g in groups)
    if missing_count <= LAYER5_POST_MAX_LARGE_FILL_TITLES:
        return None

    body_anchor_count = len({
        int(a["toc_order"]) for a in anchors
        if a.get("source") == "body"
    })
    if missing_count <= body_anchor_count:
        return None

    toc_count = len(entries)
    covered_count = len({int(a["toc_order"]) for a in anchors})
    largest_group = max((len(g) for g in groups), default=0)
    return (
        "第五层后机制需要补齐的标题过多，正文锚点不足以可靠定位，停止 LLM 硬补: "
        f"目录 {toc_count} 个，当前覆盖 {covered_count} 个，真实正文锚点 "
        f"{body_anchor_count} 个，缺失 {missing_count} 个，最大连续缺失组 "
        f"{largest_group} 个"
    )


def _layer5_post_promote_existing_matches(group_orders: list[int],
                                          entries_by_order: dict[int, dict],
                                          decisions: list[dict],
                                          start_line: int,
                                          end_line: int) -> list[dict]:
    """在候选窗口里重新拾取已存在但被第四层漏掉的真实标题。"""
    promotions = []
    used_lines = set()
    min_line = start_line

    for order in group_orders:
        entry = entries_by_order.get(int(order))
        if not entry:
            continue
        entry_key = entry.get("key") or title_key(entry.get("text", ""))
        found = None
        found_kind = None
        for d in decisions:
            line = int(d.get("line") or 0)
            if line < min_line or line > end_line or line in used_lines:
                continue
            if d.get("is_heading"):
                continue
            if d.get("_layer5_post_removed_toc_order") is not None:
                continue
            body_key = title_key(d.get("text", ""))
            kind = _toc_match_kind(body_key, entry_key)
            if not kind:
                continue
            found = d
            found_kind = kind
            break

        if not found:
            continue

        found.update(
            is_heading=True,
            level=int(entry.get("level") or 2),
            role="section",
            toc_order=int(order),
            toc_text=entry.get("text", ""),
            toc_key=entry_key,
            match_kind=f"layer5_post_{found_kind}",
            reason=(
                "第五层后机制重新拾取窗口内真实标题: "
                f"{entry.get('text', '')[:40]} ({found_kind})"
            ),
            toc_ancestors=[
                {
                    "order": int(a.get("order")),
                    "level": int(a.get("level")),
                    "text": a.get("text", ""),
                    "key": a.get("key", ""),
                }
                for a in entry.get("ancestors", [])
            ],
        )
        if display_title(found.get("text", "")) != display_title(entry.get("text", "")):
            found["normalized_title"] = display_title(entry.get("text", ""))

        used_lines.add(int(found["line"]))
        min_line = int(found["line"]) + 1
        promotions.append({
            "toc_order": int(order),
            "line": int(found["line"]),
            "text": entry.get("text", ""),
            "source_text": found.get("text", ""),
            "match_kind": found_kind,
        })

    return promotions


def _layer5_missing_order_groups(entries: list[dict],
                                 covered_orders: set[int]) -> list[list[int]]:
    """把目录中仍未覆盖的标题按连续 toc_order 分组。"""
    groups = []
    current = []
    for entry in entries:
        order = int(entry["order"])
        if order in covered_orders:
            if current:
                groups.append(current)
                current = []
            continue
        current.append(order)
    if current:
        groups.append(current)
    return groups


def _layer5_post_records_between(body_lines: list[dict], start_line: int,
                                 end_line: int) -> list[dict]:
    """收集后机制 LLM 判断用的非空正文行。"""
    return [
        rec for rec in body_lines
        if rec.get("line") and start_line <= int(rec["line"]) <= end_line
           and rec.get("text", "").strip()
    ]


def _layer5_format_anchor(anchor: dict | None, label: str) -> str:
    """把第五层补齐流程的相邻锚点格式化为 LLM 提示词片段。"""
    if not anchor:
        return f"{label}: 无"
    return (
        f"{label}: toc_order={anchor.get('toc_order')}, "
        f"line={anchor.get('line')}, level={anchor.get('level')}, "
        f"title={anchor.get('text', '')}"
    )


def _layer5_post_call_llm_for_group(group_orders: list[int], entries_by_order: dict[int, dict],
                                    records: list[dict], prev_anchor: dict,
                                    next_anchor: dict, valid_start: int,
                                    valid_end: int) -> list[dict]:
    """让 LLM 为一组连续缺失标题选择插入行。"""
    missing_text = "\n".join(
        f"- toc_order={order}, level={int(entries_by_order[order].get('level') or 2)}, "
        f"number_path={entries_by_order[order].get('number_path', '')}, "
        f"title={entries_by_order[order].get('text', '')}"
        for order in group_orders
    )
    range_text = "\n".join(
        f"line {int(r['line'])}: {r.get('text', '')}"
        for r in records
    )
    expected = """{
  "insertions": [
    {"toc_order": 12, "insert_before_line": 3456, "confidence": 0.86, "reason": "一句话说明"}
  ]
}"""
    prompt = f"""你需要在正文范围内为缺失的目录标题选择插入位置。

目标:
让最终正文标题与目录标题严格一一对应。你只判断插入位置，不生成标题文本。

规则:
1. 必须为每一个缺失标题返回一条 insertion，不多不少。
2. toc_order 只能来自“缺失标题列表”。
3. insert_before_line 表示把该标题插入到这一原文行之前。
4. insert_before_line 必须在 {valid_start} 到 {valid_end} 之间。
5. 如果标题应该紧贴后锚点前，返回后锚点行号。
6. 如果标题应该放到正文最后，返回 END_OF_BODY 对应行号。
7. 多个缺失标题的插入顺序必须符合目录顺序，行号必须非递减。
8. 标题文字和等级完全由程序使用目录提供，你不要改写标题。
9. 只返回 JSON，不要解释，不要 Markdown。

前锚点:
{_layer5_format_anchor(prev_anchor, "prev_anchor")}

后锚点:
{_layer5_format_anchor(next_anchor, "next_anchor")}

缺失标题列表:
{missing_text}

候选正文范围:
{range_text}

返回格式:
{expected}"""

    messages = [
        {"role": "system", "content": "你是文档结构分析师。禁止分析过程，只能输出严格 JSON 对象。/no_think"},
        {"role": "user", "content": prompt},
    ]
    result = call_llm_json_with_repair(
        messages,
        max_tokens=max(1200, min(12000, 260 * len(group_orders) + 800)),
        stage="layer5_post_missing_headings",
        expected_format=expected,
        timeout=TIMEOUT,
    )
    raw_insertions = result.get("insertions", [])
    if not isinstance(raw_insertions, list):
        raise SkipMarkdownFile(
            "layer5_post_invalid_llm_result",
            "layer5_post_missing_headings",
            f"LLM 返回 insertions 不是列表: {result}",
        )
    return raw_insertions


def _layer5_post_validate_insertions(raw_insertions: list[dict], group_orders: list[int],
                                     entries_by_order: dict[int, dict],
                                     valid_start: int, valid_end: int) -> list[dict]:
    """校验 LLM 返回的插入位置，确保严格覆盖当前缺失组。"""
    expected_orders = [int(o) for o in group_orders]
    seen_orders = []
    cleaned_by_order: dict[int, dict] = {}
    clamped_count = 0
    ignored_extra_count = 0
    ignored_duplicate_count = 0
    for item in raw_insertions:
        if not isinstance(item, dict):
            continue
        try:
            order = int(item.get("toc_order"))
            insert_line = int(item.get("insert_before_line"))
        except Exception:
            continue
        if order not in expected_orders:
            ignored_extra_count += 1
            continue
        if not (valid_start <= insert_line <= valid_end):
            old_line = insert_line
            insert_line = min(max(insert_line, valid_start), valid_end)
            item["reason"] = (
                f"{item.get('reason', '')}；程序修正越界行号 "
                f"{old_line}->{insert_line}"
            )
            item["confidence"] = min(float(item.get("confidence") or 0), 0.5)
            clamped_count += 1
        if order in seen_orders:
            ignored_duplicate_count += 1
            continue
        seen_orders.append(order)
        cleaned_by_order[order] = {
            "toc_order": order,
            "insert_before_line": insert_line,
            "confidence": float(item.get("confidence") or 0),
            "reason": str(item.get("reason") or "LLM 判断缺失标题插入位置"),
        }

    if set(seen_orders) != set(expected_orders):
        raise SkipMarkdownFile(
            "layer5_post_invalid_llm_result",
            "layer5_post_missing_headings",
            f"LLM 未严格返回全部缺失标题: expected={expected_orders}, got={seen_orders}",
        )

    if clamped_count:
        print(f"  第五层后机制: 修正 {clamped_count} 个越界插入行号")
    if ignored_extra_count or ignored_duplicate_count:
        print(
            f"  第五层后机制: 忽略 LLM 额外/重复插入 "
            f"{ignored_extra_count}/{ignored_duplicate_count} 个"
        )

    cleaned = [cleaned_by_order[order] for order in expected_orders]
    returned_order_changed = seen_orders != expected_orders
    if returned_order_changed:
        print(
            "  第五层后机制: LLM 返回顺序不等于目录顺序，"
            "已按 toc_order 重排"
        )

    prev_line = valid_start
    fixed_count = 0
    for item in cleaned:
        if int(item["insert_before_line"]) < prev_line:
            old_line = int(item["insert_before_line"])
            item["insert_before_line"] = prev_line
            item["reason"] = (
                f"{item.get('reason', '')}；程序修正倒序行号 {old_line}->{prev_line}"
            )
            item["confidence"] = min(float(item.get("confidence") or 0), 0.5)
            fixed_count += 1
        prev_line = int(item["insert_before_line"])
    if fixed_count:
        print(f"  第五层后机制: 修正 {fixed_count} 个倒序插入行号")
    return cleaned


def layer5_post_fill_missing_headings(toc_headings: list[dict], body_lines: list[dict],
                                      decisions: list[dict], layer5_plan: dict,
                                      max_range_tokens: int = LAYER5_POST_MAX_RANGE_TOKENS,
                                      use_llm: bool = True) -> dict:
    """第五层后机制: 用 LLM 补齐第五层后仍缺失的全部目录标题。"""
    if not body_lines:
        return layer5_plan

    entries = _build_toc_tree_entries(toc_headings)
    entries_by_order = {int(e["order"]): e for e in entries}
    anchors = _layer5_collect_covered_anchors(entries, decisions, layer5_plan)
    bad_orders = _layer5_post_find_bad_anchor_orders(entries, anchors)
    unreliable_message = _layer5_anchor_unreliable_message(
        entries_by_order,
        anchors,
        bad_orders,
    )
    if unreliable_message:
        raise SkipMarkdownFile(
            "layer5_post_unreliable_anchors",
            "layer5_post_missing_headings",
            unreliable_message,
        )
    if bad_orders:
        _layer5_post_remove_bad_anchors(decisions, layer5_plan, bad_orders)
        sample = "、".join(
            f"{order}:{entries_by_order.get(order, {}).get('text', '')[:24]}"
            for order in sorted(bad_orders)[:8]
        )
        print(f"  第五层后机制: 剔除 {len(bad_orders)} 个顺序错乱假锚点 ({sample})")
        anchors = _layer5_collect_covered_anchors(entries, decisions, layer5_plan)
    covered_orders = {int(a["toc_order"]) for a in anchors}
    groups = _layer5_missing_order_groups(entries, covered_orders)
    large_fill_message = _layer5_large_fill_unreliable_message(
        entries,
        anchors,
        groups,
    )
    if large_fill_message:
        raise SkipMarkdownFile(
            "layer5_post_large_fill_unreliable",
            "layer5_post_missing_headings",
            large_fill_message,
        )

    if not groups:
        layer5_plan["post_insertions"] = []
        layer5_plan["post_unresolved"] = []
        layer5_plan["strict_aligned"] = True
        layer5_plan["post_removed_bad_orders"] = sorted(bad_orders)
        print("  第五层后机制: 目录标题已全部覆盖，无需补齐")
        return layer5_plan

    first_line = int(body_lines[0]["line"])
    last_line = int(body_lines[-1]["line"])
    post_insertions = []
    post_promotions = []
    llm_calls = 0

    for group in groups:
        prev_anchor = next(
            (a for a in reversed(anchors) if int(a["toc_order"]) < int(group[0])),
            None,
        )
        next_anchor = next(
            (a for a in anchors if int(a["toc_order"]) > int(group[-1])),
            None,
        )
        if prev_anchor is None:
            prev_anchor = {
                "toc_order": -1,
                "line": first_line - 1,
                "level": 1,
                "text": "START_OF_BODY",
                "source": "virtual",
            }
        if next_anchor is None:
            next_anchor = {
                "toc_order": len(entries),
                "line": last_line + 1,
                "level": 1,
                "text": "END_OF_BODY",
                "source": "virtual",
            }

        start_line = max(first_line, int(prev_anchor["line"]) + 1)
        end_line = min(last_line, int(next_anchor["line"]) - 1)
        valid_start = start_line
        valid_end = int(next_anchor["line"])
        if valid_end < valid_start:
            valid_start = valid_end

        promoted = _layer5_post_promote_existing_matches(
            group,
            entries_by_order,
            decisions,
            start_line,
            end_line,
        )
        if promoted:
            post_promotions.extend(promoted)
            promoted_orders = {int(x["toc_order"]) for x in promoted}
            covered_orders.update(promoted_orders)
            group = [order for order in group if int(order) not in promoted_orders]
            if not group:
                continue
            anchors = _layer5_collect_covered_anchors(entries, decisions, layer5_plan)
            prev_anchor = next(
                (a for a in reversed(anchors) if int(a["toc_order"]) < int(group[0])),
                None,
            )
            next_anchor = next(
                (a for a in anchors if int(a["toc_order"]) > int(group[-1])),
                None,
            )
            if prev_anchor is None:
                prev_anchor = {
                    "toc_order": -1,
                    "line": first_line - 1,
                    "level": 1,
                    "text": "START_OF_BODY",
                    "source": "virtual",
                }
            if next_anchor is None:
                next_anchor = {
                    "toc_order": len(entries),
                    "line": last_line + 1,
                    "level": 1,
                    "text": "END_OF_BODY",
                    "source": "virtual",
                }
            start_line = max(first_line, int(prev_anchor["line"]) + 1)
            end_line = min(last_line, int(next_anchor["line"]) - 1)
            valid_start = start_line
            valid_end = int(next_anchor["line"])
            if valid_end < valid_start:
                valid_start = valid_end

        records = _layer5_post_records_between(body_lines, start_line, end_line)
        range_text = "\n".join(r.get("text", "") for r in records)
        range_tokens = count_tokens(range_text)
        group_limit = (
            max(max_range_tokens, LAYER5_POST_SINGLE_TITLE_MAX_RANGE_TOKENS)
            if len(group) == 1
            else max_range_tokens
        )
        if range_tokens > group_limit:
            titles = "、".join(entries_by_order[o].get("text", "") for o in group[:8])
            raise SkipMarkdownFile(
                "layer5_post_range_too_large",
                "layer5_post_missing_headings",
                (
                    f"缺失标题组候选范围过大: toc_order={group[0]}-{group[-1]}, "
                    f"range_tokens≈{range_tokens}, limit={group_limit}, titles={titles}"
                ),
            )

        if not records or not use_llm:
            raw_insertions = [
                {
                    "toc_order": order,
                    "insert_before_line": valid_end,
                    "confidence": 0.0,
                    "reason": "候选范围为空或未启用 LLM，按目录顺序插入到后锚点前",
                }
                for order in group
            ]
        else:
            llm_calls += 1
            raw_insertions = _layer5_post_call_llm_for_group(
                group,
                entries_by_order,
                records,
                prev_anchor,
                next_anchor,
                valid_start,
                valid_end,
            )

        cleaned = _layer5_post_validate_insertions(
            raw_insertions,
            group,
            entries_by_order,
            valid_start,
            valid_end,
        )
        for item in cleaned:
            entry = entries_by_order[int(item["toc_order"])]
            post_insertions.append({
                "insert_before_line": int(item["insert_before_line"]),
                "toc_order": int(item["toc_order"]),
                "level": int(entry.get("level") or 2),
                "text": entry.get("text", ""),
                "number_path": entry.get("number_path", ""),
                "role": "synthetic_heading",
                "source": "layer5_post_llm" if records and use_llm else "layer5_post_rule",
                "confidence": float(item.get("confidence") or 0),
                "reason": item.get("reason") or "第五层后机制补齐缺失标题",
            })

    if post_insertions:
        merged = list(layer5_plan.get("synthetic_headings", [])) + post_insertions
        layer5_plan["synthetic_headings"] = sorted(
            merged,
            key=lambda x: (
                int(x.get("insert_before_line") or 0),
                int(x.get("toc_order") or 0),
            ),
        )

    final_anchors = _layer5_collect_covered_anchors(entries, decisions, layer5_plan)
    final_body_orders = {int(a["toc_order"]) for a in final_anchors}
    all_synthetic_orders = {
        int(x["toc_order"])
        for x in layer5_plan.get("synthetic_headings", [])
        if x.get("toc_order") is not None
    }
    final_covered = final_body_orders | all_synthetic_orders
    unresolved = [
        {
            "toc_order": int(e["order"]),
            "level": int(e.get("level") or 2),
            "text": e.get("text", ""),
            "number_path": e.get("number_path", ""),
        }
        for e in entries
        if int(e["order"]) not in final_covered
    ]

    layer5_plan["post_insertions"] = post_insertions
    layer5_plan["post_promotions"] = post_promotions
    layer5_plan["post_unresolved"] = unresolved
    layer5_plan["strict_aligned"] = not unresolved
    layer5_plan["post_llm_calls"] = llm_calls
    layer5_plan["post_removed_bad_orders"] = sorted(bad_orders)
    layer5_plan["synthetic_orders"] = sorted(all_synthetic_orders)
    layer5_plan["unresolved"] = unresolved

    print(
        f"  第五层后机制: 缺失组 {len(groups)} 个, "
        f"规则拾取 {len(post_promotions)} 个标题, "
        f"LLM 补齐 {len(post_insertions)} 个标题, 剩余 {len(unresolved)} 个"
    )
    if unresolved:
        raise SkipMarkdownFile(
            "layer5_post_unresolved",
            "layer5_post_missing_headings",
            f"第五层后机制后仍有 {len(unresolved)} 个目录标题未覆盖",
        )
    return layer5_plan


# ============================================================
# 输出层: 目录与正文渲染
# ============================================================

def _render_toc_and_body(toc_headings: list[dict], body_lines: list[dict],
                         decisions: list[dict], layer5_plan: dict | None = None) -> str:
    """输出只包含目录和正文的 Markdown。"""
    result_lines = ["## 目录", ""]
    for h in toc_headings:
        if title_key(h.get("text", "")) == title_key("目录"):
            continue
        indent = "  " * (int(h.get("level", 2) or 2) - 2)
        result_lines.append(f"{indent}- {display_title(h.get('text', ''))}")
    result_lines.append("")

    decision_by = {d["line"]: d for d in decisions}
    rescued_heading_lines = {
        int(d["line"])
        for d in decisions
        if d.get("is_heading") and d.get("_close_bad_fence_before")
    }
    skip_lines = set()
    for d in decisions:
        for key in ("_merged_plain_line", "_merged_plain_line2", "_skip_next_line"):
            if d.get(key):
                skip_lines.add(d[key])

    synthetic_by_line: dict[int, list[dict]] = {}
    if layer5_plan:
        for item in layer5_plan.get("synthetic_headings", []):
            line = int(item.get("insert_before_line") or 0)
            if line > 0:
                synthetic_by_line.setdefault(line, []).append(item)
        for items in synthetic_by_line.values():
            items.sort(key=lambda x: int(x.get("toc_order") or 0))

    def _opens_orphan_fence_before_rescued_heading(start_index: int) -> bool:
        """判断当前 fence 是否是坏代码块残留，而不是真正代码块开头。"""
        for future in body_lines[start_index + 1:start_index + 120]:
            future_line = int(future.get("line") or 0)
            future_text = future.get("text", "").strip()
            if future_line in rescued_heading_lines:
                return True
            if future_text.startswith("```") or future_text.startswith("~~~"):
                return False
        return False

    pat = re.compile(r"^(#{1,6})\s+(.+)")
    render_in_fence = False
    for rec_index, rec in enumerate(body_lines):
        line_no = rec["line"]
        line_text = rec.get("text", "")
        stripped = line_text.strip()
        d = decision_by.get(line_no)
        m = pat.match(stripped)
        confirmed_heading = bool(m and d and d.get("is_heading") and d.get("level"))
        pending_synthetic = synthetic_by_line.get(line_no, [])

        if render_in_fence and (
            pending_synthetic or (confirmed_heading and d.get("_close_bad_fence_before"))
        ):
            result_lines.append("```")
            render_in_fence = False

        for item in pending_synthetic:
            level = int(item.get("level") or 2)
            result_lines.append(f"{'#' * level} {display_title(item.get('text', ''))}")
            result_lines.append("")

        if line_no in skip_lines:
            continue

        if not line_text.strip():
            result_lines.append("")
            continue

        if re.fullmatch(r"`{1,2}|~{1,2}", stripped):
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            if (
                not render_in_fence
                and _opens_orphan_fence_before_rescued_heading(rec_index)
            ):
                continue
            result_lines.append(line_text)
            render_in_fence = not render_in_fence
            continue

        if d and d.get("role") == "noise":
            continue
        if _is_noise_heading_text(line_text):
            continue

        if m and d and d["is_heading"] and d["level"]:
            title = display_title(d.get("normalized_title") or d.get("text") or m.group(2))
            result_lines.append(f"{'#' * int(d['level'])} {title}")
        elif m and d and not d["is_heading"]:
            result_lines.append(m.group(2))
        elif m and not d:
            result_lines.append(m.group(2))
        else:
            result_lines.append(line_text)

    last_seen_line = int(body_lines[-1]["line"]) if body_lines else 0
    tail_synthetic = [
        (line, items)
        for line, items in sorted(synthetic_by_line.items())
        if int(line) > last_seen_line
    ]
    if tail_synthetic and render_in_fence:
        result_lines.append("```")
        render_in_fence = False
    for _, items in tail_synthetic:
        for item in items:
            level = int(item.get("level") or 2)
            result_lines.append(f"{'#' * level} {display_title(item.get('text', ''))}")
            result_lines.append("")

    return neutralize_setext_headings("\n".join(result_lines))


def _validate_final_heading_alignment(md_text: str, toc_headings: list[dict],
                                      file_name: str) -> None:
    """最终输出前校验正文标题与目录标题严格一一对应。"""
    visible = []
    pat = re.compile(r"^(#{2,6})\s+(.+)$")
    for line_no, line in enumerate(md_text.splitlines(), 1):
        m = pat.match(line)
        if not m:
            continue
        visible.append({
            "line": line_no,
            "level": len(m.group(1)),
            "text": display_title(m.group(2)),
        })

    dir_key = title_key("目录")
    if visible and title_key(visible[0]["text"]) == dir_key:
        visible = visible[1:]

    expected = [
        {
            "level": int(h.get("level") or 2),
            "text": display_title(h.get("text", "")),
        }
        for h in toc_headings
        if title_key(h.get("text", "")) != dir_key
    ]

    if len(visible) != len(expected):
        raise SkipMarkdownFile(
            "final_heading_alignment_failed",
            "final_validation",
            (
                f"{file_name} 最终标题数量不一致: "
                f"目录 {len(expected)} 个, 正文 {len(visible)} 个"
            ),
        )

    for idx, (want, got) in enumerate(zip(expected, visible), 1):
        if int(want["level"]) == int(got["level"]) \
                and title_key(want["text"]) == title_key(got["text"]):
            continue
        raise SkipMarkdownFile(
            "final_heading_alignment_failed",
            "final_validation",
            (
                f"{file_name} 第 {idx} 个标题不一致: "
                f"目录 H{want['level']} {want['text']}；"
                f"正文 line {got['line']} H{got['level']} {got['text']}"
            ),
        )


# ============================================================
# 输出后处理
# ============================================================

def neutralize_setext_headings(md_text: str) -> str:
    """消除 Markdown 隐式标题。

    转换后的标题统一使用 # 形式。OCR 图片文本里经常出现:
        VCC5V
        -
    Markdown 会把它渲染成 H2。这里转义这种下划线行，避免产生伪标题。

    输入:
        md_text: str，已经按 `#` 标准化后的 Markdown。

    输出:
        str，转义 setext 下划线后的 Markdown。

    例子:
        输入:
            VCC5V
            -
        输出:
            VCC5V
            \\-
    """
    lines = md_text.split("\n")
    in_fence = False
    result = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            result.append(line)
            continue

        if (
            not in_fence
            and result
            and result[-1].strip()
            and not result[-1].lstrip().startswith("#")
            and re.fullmatch(r"[-=]+", stripped or "")
        ):
            indent = line[:len(line) - len(line.lstrip())]
            result.append(indent + "\\" + line.lstrip())
        else:
            result.append(line)

    return "\n".join(result)


# ============================================================
# 缓存
# ============================================================


def _cache_key(input_path: str, stage: str) -> Path:
    """生成缓存文件路径。

    作用:
        缓存固定放在输入 md 所在目录的 `.cache_batch3` 下，避免从不同工作目录运行时
        产生多套缓存。

    输入:
        input_path: str，输入 Markdown 文件路径。
        stage: str，缓存阶段名，例如 "layer1_toc" 或 "l2b.s220.l120..."。

    输出:
        Path，缓存 JSON 路径。

    例子:
        _cache_key("H:/book/a.md", "layer1_toc")
        -> H:/book/.cache_batch3/a.<hash>.layer1_toc.json
    """
    resolved_input = Path(input_path).resolve()
    digest_source = f"{CACHE_VERSION}|{resolved_input}"
    digest = hashlib.sha1(digest_source.encode()).hexdigest()[:12]
    stem = resolved_input.stem[:80]
    return resolved_input.parent / ".cache_batch3" / f"{stem}.{digest}.{stage}.json"


def _load_cache(path: Path) -> dict | None:
    """读取 JSON 缓存。

    输入:
        path: Path，缓存文件路径。

    输出:
        dict | None，读取成功返回 dict；不存在或解析失败返回 None。

    例子:
        cached = _load_cache(Path(".cache_batch3/x.json"))
    """
    try:
        if path.exists():
            return json.loads(path.read_text("utf-8"))
    except Exception: pass
    return None


def _save_cache(path: Path, data: dict) -> None:
    """保存 JSON 缓存。

    输入:
        path: Path，缓存文件路径。
        data: dict，可 JSON 序列化的数据。

    输出:
        None。函数会自动创建父目录。

    例子:
        _save_cache(Path(".cache_batch3/x.json"), {"found": True})
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ============================================================
# 主入口
# ============================================================


def process_file(
    input_path: str,
    output_path: str = "",
    json_path: str = "",
    use_cache: bool = True,
    toc_single_max_lines: int = TOC_PARSE_SINGLE_MAX_LINES,
    toc_chunk_lines: int = TOC_PARSE_CHUNK_LINES,
    toc_chunk_max_tokens: int = TOC_PARSE_CHUNK_MAX_TOKENS,
) -> dict:
    """处理单个 Markdown 文件。

    运行顺序:
        0. 预处理: 提取原文 `#` 候选。
        1. 第一层: 定位目录范围，并用 tail 扫描补全目录尾部。
        2. 第二层: 解析目录文本，得到目录标题列表。
        3. 第三层: 按完整语义块重建目录 parent tree，并计算标题等级。
        4. 第四层: 准备正文候选，并把目录标题映射到正文标题位置。
        5. 第五层: 用目录树补齐可定位的缺失父级标题。
        6. 输出层: 只渲染目录和正文，并消除 setext 隐式标题。
    """
    in_path = Path(input_path)
    output_path = output_path or str(Path("standardized_md3") / in_path.name)
    json_path = json_path or str(Path("heading_json3") / f"{in_path.stem}.headings.json")

    print(f"\n{'='*60}")
    print(f"处理: {in_path.name}")
    print(f"{'='*60}")

    if not in_path.exists():
        print(f"  文件不存在，跳过: {input_path}")
        return {
            "input": input_path,
            "error": "missing_file",
            "message": "输入文件不存在，可能在批量运行期间被移动或重命名",
        }

    # 0. 读取 & 提取
    md_text = in_path.read_text("utf-8")
    headings = extract_headings(md_text)
    print(f"  提取: {len(headings)} 个 # 候选")

    # 1. 第一层: 定位目录
    cache = _cache_key(input_path, "layer1_toc")
    if use_cache and (cached := _load_cache(cache)):
        toc_range = cached
        print(f"  [缓存] 第一层: 目录行{toc_range.get('toc_start_line')}-{toc_range.get('toc_end_line')}")
    else:
        toc_range = layer1_detect_toc(headings, in_path.name, md_text)

        if use_cache:
            _save_cache(cache, toc_range)

    # 第一层 B: 向后扫描目录尾部。即使第一层来自缓存，也允许新规则修正旧范围。
    # 如果第一层已经覆盖到接近整本书，先交给第二层范围确认，避免 tail 扫描继续扩大问题。
    if toc_range.get("found") and not toc_range.get("layer1_overwide"):
        extended = _scan_toc_tail(md_text, toc_range, in_path.name)
        if extended.get("toc_end_line") != toc_range.get("toc_end_line"):
            print(f"  tail扫描扩展: {toc_range['toc_end_line']} → {extended['toc_end_line']}")
            toc_range = extended
            if use_cache:
                _save_cache(cache, toc_range)

    if not toc_range.get("found"):
        print("  文本没有目录区域，跳过第二层目录解析")
        return {
            "input": input_path,
            "error": "no_toc_region",
            "message": "文本没有目录区域",
        }

    # 1B/2A. 第二层范围确认: 第一层给宽范围，第二层在正式解析前确认真实目录尾部。
    confirmed_toc_range = _confirm_layer2_toc_range_by_llm(md_text, toc_range, in_path.name)
    if confirmed_toc_range.get("toc_end_line") != toc_range.get("toc_end_line"):
        toc_range = confirmed_toc_range
        if use_cache:
            _save_cache(_cache_key(input_path, "layer1_toc"), toc_range)

    validate_toc_range_or_raise(md_text, toc_range, in_path.name)

    # 2. 第二层: 解析目录
    layer2_stage = (
        f"layer2_toc_headings.r{toc_range.get('toc_start_line')}-{toc_range.get('toc_end_line')}"
        f".s{toc_single_max_lines}"
        f".l{toc_chunk_lines}.t{toc_chunk_max_tokens}"
    )
    cache = _cache_key(input_path, layer2_stage)
    if use_cache and (cached := _load_cache(cache)):
        toc_headings = cached.get("headings", [])
        print(f"  [缓存] 第二层: {len(toc_headings)} 个目录标题")
        _cached_toc_headings = json.loads(json.dumps(toc_headings, ensure_ascii=False))
        _cleaned_toc_lines, _, _ = _clean_toc_lines(md_text, toc_range)
        toc_headings = _repair_layer2_missing_prefix_by_llm(
            toc_headings,
            _cleaned_toc_lines,
            in_path.name,
            toc_chunk_max_tokens,
        )
        if use_cache and toc_headings != _cached_toc_headings:
            _save_cache(cache, {"headings": toc_headings})
    else:
        toc_headings = layer2_parse_toc(
            md_text,
            toc_range,
            in_path.name,
            single_max_lines=toc_single_max_lines,
            chunk_lines=toc_chunk_lines,
            chunk_max_tokens=toc_chunk_max_tokens,
            cache_input_path=input_path,
            use_cache=use_cache,
        )
        if use_cache:
            _save_cache(cache, {"headings": toc_headings})

    toc_headings = _filter_layer2_toc_roles_by_llm(toc_headings, in_path.name)
    toc_headings = _drop_toc_tail_contamination(toc_headings)
    toc_headings = _drop_pretoc_frontmatter_headings(toc_headings, md_text, toc_range)

    # 3. 第三层: 完整语义块 parent tree 重建。
    layer3_digest = hashlib.sha1(
        json.dumps(
            [
                {
                    "text": display_title(h.get("text", "")),
                    "level": int(h.get("level") or 2),
                }
                for h in toc_headings
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    layer3_cache = _cache_key(input_path, f"layer3_parent_tree.{layer3_digest}")
    if use_cache and (cached := _load_cache(layer3_cache)):
        toc_headings = cached.get("headings", [])
        print(f"  [缓存] 第三层 parent tree: {len(toc_headings)} 个目录标题")
    else:
        toc_headings = layer3_validate(toc_headings, in_path.name)
        if use_cache:
            _save_cache(layer3_cache, {"headings": toc_headings})

    # 3B/4A. 第四层准备: 清洗正文候选，并让 LLM 找真正正文标题起点
    body_start, body_lines = _prepare_layer4_body_lines(md_text, toc_range, toc_headings)

    # 4B. 第四层: 正文标题匹配与等级映射。
    body_headings = _extract_headings_from_body_lines(body_lines, toc_headings)
    decisions = layer4_apply(body_headings, toc_headings, toc_range, md_text)

    # 5. 第五层: 以目录树为骨架补齐正文缺失的父级标题。
    layer5_plan = layer5_mount_body_to_toc(toc_headings, body_lines, decisions)
    layer5_plan = layer5_post_fill_missing_headings(
        toc_headings,
        body_lines,
        decisions,
        layer5_plan,
    )
    n_heading = sum(1 for d in decisions if d["is_heading"])
    n_plain = sum(1 for d in decisions if not d["is_heading"])
    n_synthetic = len(layer5_plan.get("synthetic_headings", []))
    print(f"  最终: {n_heading + n_synthetic} 个标题 "
          f"({n_heading} 正文锚点 + {n_synthetic} 目录补齐), {n_plain} 个非标题")

    # 6. 输出: 仅保留目录与正文。目录项是普通文本，正文小标题保留 Markdown 标题格式。
    fixed = _render_toc_and_body(toc_headings, body_lines, decisions, layer5_plan)
    _validate_final_heading_alignment(fixed, toc_headings, in_path.name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(fixed, "utf-8")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(
        json.dumps({"file_name": in_path.name, "toc_range": toc_range,
                    "toc_headings": toc_headings, "decisions": decisions,
                    "layer5": layer5_plan,
                    "body_start": body_start, "body_lines": body_lines},
                   ensure_ascii=False, indent=2), "utf-8")
    print(f"  输出: {output_path}")

    return {"input": input_path, "output": output_path, "json": json_path,
            "candidates": len(headings), "toc_found": toc_range.get("found"),
            "toc_headings_count": len(toc_headings),
            "body_start": body_start,
            "final_headings": n_heading + n_synthetic,
            "final_non_headings": n_plain}


def find_md_files(paths: list[str]) -> list[str]:
    """从文件或目录参数中查找 Markdown 文件。

    输入:
        paths: list[str]，命令行传入的路径列表。元素可以是单个 .md 文件，也可以是目录。

    输出:
        list[str]，所有找到的 .md 文件路径。目录会递归查找。

    例子:
        find_md_files(["book.md"]) -> ["book.md"]
        find_md_files(["data"]) -> ["data/a.md", "data/sub/b.md", ...]
    """
    results = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".md":
            results.append(str(p))
        elif p.is_dir():
            results.extend(str(f) for f in sorted(p.rglob("*.md")))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Markdown 标题标准化 (目录驱动长文本架构)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out-dir", default="standardized_md3")
    ap.add_argument("--json-dir", default="heading_json3")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base-url", "--url", dest="base_url", default=None,
                    help="OpenAI-compatible chat completions 地址；可传根地址，如 https://api.deepseek.com")
    ap.add_argument("--model", default=None,
                    help=f"本次运行使用的模型名；默认 {MODEL}")
    ap.add_argument("--api-key", default=None,
                    help="本次运行使用的 API key；不传则读取 --api-key-env 指定的环境变量")
    ap.add_argument("--api-key-env", default=None,
                    help=f"API key 环境变量名；默认 {API_KEY_ENV}")
    ap.add_argument("--vllm-extra-body", action=argparse.BooleanOptionalAction, default=None,
                    help="是否发送 vLLM/Qwen 的 enable_thinking 扩展字段；默认本地服务器发送，指定 --base-url 时不发送")
    ap.add_argument("--toc-single-max-lines", type=int, default=TOC_PARSE_SINGLE_MAX_LINES,
                    help="第二层目录清洗后行数不超过该值时单批发送")
    ap.add_argument("--toc-chunk-lines", type=int, default=TOC_PARSE_CHUNK_LINES,
                    help="第二层目录分块时每批最多发送多少条清洗后目录行")
    ap.add_argument("--toc-chunk-max-tokens", type=int, default=TOC_PARSE_CHUNK_MAX_TOKENS,
                    help="第二层每批 LLM 输出 token 上限")
    args = ap.parse_args()

    configure_llm(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        use_vllm_extras=args.vllm_extra_body,
    )
    key_source = "命令行 --api-key" if args.api_key else f"环境变量 {API_KEY_ENV}"
    if not args.api_key and not os.getenv(API_KEY_ENV):
        key_source = "默认 EMPTY"
    print(
        f"LLM 配置: model={MODEL}, url={BASE_URL}, "
        f"api_key={key_source}, vllm_extra_body={LLM_USE_VLLM_EXTRAS}"
    )

    files = find_md_files(args.paths)
    if not files: print("没有 .md 文件"); sys.exit(1)

    for f in files:
        out = str(Path(args.out_dir) / Path(f).name)
        js = str(Path(args.json_dir) / f"{Path(f).stem}.headings.json")
        if args.resume and Path(out).exists() and Path(js).exists():
            print(f"跳过: {Path(f).name}"); continue
        try:
            result = process_file(
                f,
                out,
                js,
                use_cache=not args.no_cache,
                toc_single_max_lines=args.toc_single_max_lines,
                toc_chunk_lines=args.toc_chunk_lines,
                toc_chunk_max_tokens=args.toc_chunk_max_tokens,
            )
            if result.get("error"):
                Path(js).parent.mkdir(parents=True, exist_ok=True)
                Path(js).write_text(json.dumps({
                    "file_name": Path(f).name,
                    "input": f,
                    **result,
                }, ensure_ascii=False, indent=2), "utf-8")
        except SkipMarkdownFile as e:
            print(f"  跳过当前文件: {Path(f).name} ({e.error}/{e.stage}) {e.message}")
            Path(js).parent.mkdir(parents=True, exist_ok=True)
            Path(js).write_text(json.dumps({
                "file_name": Path(f).name,
                "input": f,
                "error": e.error,
                "stage": e.stage,
                "message": e.message,
            }, ensure_ascii=False, indent=2), "utf-8")
            continue
        except Exception as e:
            print(f"  跳过当前文件: {Path(f).name} (processing_failed) {e}")
            Path(js).parent.mkdir(parents=True, exist_ok=True)
            Path(js).write_text(json.dumps({
                "file_name": Path(f).name,
                "input": f,
                "error": "processing_failed",
                "stage": "unknown",
                "message": str(e),
            }, ensure_ascii=False, indent=2), "utf-8")
            continue
