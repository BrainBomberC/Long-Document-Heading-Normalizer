"""
Markdown 标题标准化 - 两层架构 (精简版)

架构:
  第一层 (LLM):  提取 # 候选 → 按需分批 → LLM 结合上下文语义判定
  第二层 (规则): 合并 LLM 结果 → 全局规则重建标题等级 → 输出标准化 MD

用法:
  python tools.py data --recursive
  python tools.py paper.md --cache-dir .cache
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ============================================================
# 常量
# ============================================================

BASE_URL = "http://brain-X99:8000/v1/chat/completions"
MODEL = "qwen3-30b"
API_KEY_ENV = "LOCAL_LLM_API_KEY"

CONTEXT_LENGTH = 16384
OUTPUT_TOKENS = 2000
TOKEN_BUDGET = CONTEXT_LENGTH - OUTPUT_TOKENS - 700  # ≈ 13684   700是预留安全范围给output_tokens的

SYSTEM_PROMPT = """你是一个文档标题结构分析师。审查 Markdown 中所有 `#` 候选行，判断真正的正文标题和级别。

## 宏观规则

### H1 (#) — 极其严格
只有这些才是 H1: 论文题目、学位论文题目、书籍书名。其他任何情况都不设 H1。

### H2 (##)
摘要/Abstract、目录(本身)、第X章、Introduction/Methods/Results/Discussion、参考文献/References、
致谢/Acknowledgements、附录/Appendix、一级数字编号(1./2./3.)

### H3 (###)
1.1/2.3 等二级数字编号、主章节下的分节

### H4 (####)
1.1.1/2.3.4 等三级数字编号

### 非标题
- 目录区条目(只是索引，不是正文标题)
- 封面信息(作者、单位、期刊名)
- 图片说明、表格说明
- 中文学位论文括号编号: (1)/(2)/（一）是段内枚举

### 其他
- 同一标题在目录和正文各出现一次 → 目录的设 is_heading=false, role="toc_entry"
- 同级别标题给相同的 level

## 输出格式
直接返回 JSON，以 { 开头、} 结尾。不要写分析过程，不要用 "首先/接下来/现在" 开头。
返回格式: {"decisions": [{"line": 行号, "is_heading": true/false, "level": 1-6或null, "role": "...", "reason": "..."}]}
每个候选对应一个 decision，不要遗漏。"""

USER_PROMPT = """文件名: {file_name}
以下是 `#` 候选标题行，含前后正文和前后 # 标题:
{heading_details}
请逐个判断，返回 JSON。"""

BATCH_USER_PROMPT = """文件名: {file_name}
## 全局标题大纲
{global_outline}
## 当前批次候选
{heading_details}
请逐个判断**当前批次**候选，返回 JSON。"""

REPAIR_PROMPT = """上次输出非法。重试: 只返回合法 JSON，包含这些 line: {line_numbers}"""


# ============================================================
# 工具函数
# ============================================================


def count_tokens(text: str) -> int:
    """粗略 token 估算: 中文 1.2, 英文 0.25"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return int(cjk * 1.2 + (len(text) - cjk) / 4) + 1


def parse_json(text: str) -> dict:
    """从 LLM 返回文本提取 JSON。

    按优先级尝试:
    1. 直接解析全文 (去掉 think 标签后)
    2. 提取 ```json ... ``` 代码块
    3. 用正则找第一个 { ... } 对象 (尝试多种边界)
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 策略1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2: 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 策略3: 找第一个完整 JSON 对象 { ... }
    # 从第一个 { 开始，用计数器匹配配对的 }
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break  # 这个区间不行，跳出试别的
        # 策略4: 直接尝试找到最后一个 }
        end = text.rfind("}")
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"无法解析 JSON，原始文本前500字符:\n{text[:500]}")


def clean_title(text: str) -> str:
    """清理标题: 去尾部点线页码，合并空白"""
    t = re.sub(r"[.…·]{2,}\s*\d*\s*$", "", text.strip())
    t = re.sub(r"\s+\d{1,4}\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ============================================================
# 第一层: 提取候选
# ============================================================


def extract_headings(md_text: str, context_chars: int = 200) -> list[dict]:
    """提取所有 # 行及其上下文。

    Returns:
        [{line, raw, text, prev_text, next_text, prev_headings, next_headings}, ...]
    """
    lines = md_text.split("\n")
    pat = re.compile(r"^(#{1,6})\s+(.+)")

    # 收集所有 # 行
    all_h = [(i + 1, l.strip(), m.group(2).strip())
             for i, l in enumerate(lines)
             if (m := pat.match(l.strip()))]

    results = []
    for idx, (line_num, raw, text) in enumerate(all_h):
        pos = line_num - 1

        # 前后正文 (各取约 20 行窗口，截断到 context_chars)
        prev_text = "\n".join(lines[max(0, pos - 20):pos])
        if len(prev_text) > context_chars:
            prev_text = prev_text[-context_chars:]
        next_text = "\n".join(lines[pos + 1 : min(len(lines), pos + 21)])
        if len(next_text) > context_chars:
            next_text = next_text[:context_chars]

        # 前后相邻 # 标题 (各最多 2 条)
        prev_h = [f"行{all_h[j][0]}: {all_h[j][2]}" for j in range(idx - 1, max(idx - 3, -1), -1)][::-1]
        next_h = [f"行{all_h[j][0]}: {all_h[j][2]}" for j in range(idx + 1, min(idx + 3, len(all_h)))]

        results.append({
            "line": line_num, "raw": raw, "text": text,
            "prev_text": prev_text.strip(), "next_text": next_text.strip(),
            "prev_headings": prev_h, "next_headings": next_h,
        })

    return results


def format_heading(h: dict) -> str:
    """单个候选的 prompt 文本块"""
    parts = [f"--- 行 {h['line']} ---", f"候选行: {h['raw']}",
             f"前文: {h['prev_text'][:300]}", f"后文: {h['next_text'][:300]}"]
    if h["prev_headings"]:
        parts.append(f"前面标题: {' | '.join(h['prev_headings'])}")
    if h["next_headings"]:
        parts.append(f"后面标题: {' | '.join(h['next_headings'])}")
    return "\n".join(parts)


def build_outline(headings: list[dict]) -> str:
    """全局标题大纲 (分批时附带)"""
    return "\n".join(f"行{h['line']}: {h['text']}" for h in headings)


# ============================================================
# 第一层: 分批
# ============================================================


def batch_headings(headings: list[dict], file_name: str) -> list[list[dict]]:
    """判断并拆分批次，基于 token 预算 (≈13684)"""
    # 不分批时 prompt
    details = "\n\n".join(format_heading(h) for h in headings)
    if count_tokens(SYSTEM_PROMPT) + count_tokens(USER_PROMPT.format(
            file_name=file_name, heading_details=details)) <= TOKEN_BUDGET:
        return [headings]

    # 需分批 建立batches
    batches, cur_batch, cur_details = [], [], ""
    outline = build_outline(headings)

    for h in headings:
        test_details = cur_details + format_heading(h) + "\n\n"
        test_prompt = BATCH_USER_PROMPT.format(
            file_name=file_name, global_outline=outline, heading_details=test_details)
        if cur_batch and count_tokens(SYSTEM_PROMPT) + count_tokens(test_prompt) > TOKEN_BUDGET:
            batches.append(cur_batch)
            cur_batch, cur_details = [h], format_heading(h) + "\n\n"
        elif not cur_batch and count_tokens(SYSTEM_PROMPT) + count_tokens(test_prompt) > TOKEN_BUDGET:  # 单个标题超token预算
            print(f"  ⚠ 警告: 行{h['line']} 单个标题超批次token预算，强制放入")                        
            cur_batch.append(h)                                                                        
            cur_details = test_details          
        else:
            cur_batch.append(h)
            cur_details = test_details

    if cur_batch:
        batches.append(cur_batch)
    return batches


def build_messages(headings_batch: list[dict], file_name: str, outline: str = "") -> list[dict]:
    """构建 system + user messages"""
    details = "\n\n".join(format_heading(h) for h in headings_batch)
    if outline:
        user = BATCH_USER_PROMPT.format(file_name=file_name, global_outline=outline, heading_details=details)
    else:
        user = USER_PROMPT.format(file_name=file_name, heading_details=details)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think\n" + user},   # 双重保险禁止思考
    ]


# ============================================================
# LLM 调用
# ============================================================


def call_llm(messages: list[dict], max_tokens: int = OUTPUT_TOKENS,
             timeout: int = 240, retries: int = 3) -> str:
    """调用 OpenAI 兼容 API，失败重试"""
    api_key = os.getenv(API_KEY_ENV) or "EMPTY"
    payload = {"model": MODEL, "messages": messages, "temperature": 0.0,
               "top_p": 0.1, "max_tokens": max_tokens,
               "enable_thinking": False,
               "chat_template_kwargs": {"enable_thinking": False}}

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
            if attempt < retries:
                time.sleep(1.0 + attempt)
            last_error = e
    raise RuntimeError(f"LLM 调用失败: {last_error}")


# ============================================================
# 第一层: 分类
# ============================================================


def layer1_classify(headings: list[dict], file_name: str,
                    cache_dir: str = "") -> list[dict]:
    """第一层: LLM 判定每个 # 候选的语义级别"""
    batches = batch_headings(headings, file_name)
    outline = build_outline(headings) if len(batches) > 1 else ""
    all_decisions = []

    for idx, batch in enumerate(batches):
        print(f"  批次 {idx+1}/{len(batches)}: {len(batch)} 候选 行{batch[0]['line']}-{batch[-1]['line']}")

        # 缓存
        cached = _load_cache(cache_dir, file_name, idx, batch) if cache_dir else None
        if cached:
            all_decisions.extend(cached)
            print("    缓存命中")
            continue

        messages = build_messages(batch, file_name, outline)
        last_content = ""
        decisions = None
        for attempt in range(3):  # 最多3次: 原始 + 2次修复
            try:
                content = call_llm(messages)
                last_content = content
                result = parse_json(content)
                decisions = result["decisions"]
                expected = {h["line"] for h in batch}
                got = {d["line"] for d in decisions}
                if got != expected:
                    raise ValueError(f"行号不匹配: 缺{expected-got} 多{got-expected}")
                break  # 成功
            except Exception as e:
                print(f"    第{attempt+1}次尝试失败: {e}")
                if attempt < 2 and last_content:
                    messages += [
                        {"role": "assistant", "content": last_content[:1200]},
                        {"role": "user", "content": REPAIR_PROMPT.format(
                            line_numbers=[h["line"] for h in batch])}]
                else:
                    # 最后一次也失败 → 打印响应并抛出
                    print(f"    LLM 原始响应前600字符:\n{last_content[:600]}")
                    raise RuntimeError(
                        f"批次{idx+1} LLM响应3次均无法解析。"
                        f"请检查 LLM 是否支持 JSON 输出。"
                        f"可尝试用 --json-mode 参数启用 response_format")

        all_decisions.extend(decisions)
        if cache_dir:
            _save_cache(cache_dir, file_name, idx, batch, decisions)

    return all_decisions


# ============================================================
# 第二层: 规则重建标题等级
# ============================================================


def layer2_rebuild(headings: list[dict], decisions: list[dict]) -> list[dict]:
    """第二层: 规则引擎全局重建标题等级。"""
    decision_by = {d["line"]: d for d in decisions}
    is_thesis = _is_thesis(headings)
    toc_ranges = _toc_ranges(headings)

    MAJOR = {"abstract", "introduction", "results", "discussion", "methods", "method",
             "materials and methods", "references", "reference", "acknowledgements",
             "acknowledgments", "appendix", "author contributions", "conclusion", "conclusions"}

    def _in_toc(line):
        return any(r["start"] <= line <= r["end"] for r in toc_ranges)

    def _is_toc_marker(t):
        return clean_title(t).lower() in {"目录", "contents", "table of contents"}

    def _depth(t):
        m = re.match(r"^(\d+(?:\.\d+)*)", clean_title(t))
        return m.group(1).count(".") + 1 if m else 0

    def _is_ch_chapter(t):
        return bool(re.match(r"^第[一二三四五六七八九十百零0-9]+[章节篇卷部]", clean_title(t)))

    def _is_paren(t):
        return bool(re.match(r"^[（(]\s*[一二三四五六七八九十百零0-9]+\s*[）)]", clean_title(t)))

    def _key(t):
        return clean_title(t).lower()

    def _hdr(line, raw, text, **kw):
        return {"line": line, "raw": raw, "text": text,
                "is_heading": False, "level": None, "role": "paragraph", "reason": "", **kw}

    final = []
    for h in headings:
        line, raw, text = h["line"], h["raw"], h["text"]
        d = decision_by.get(line, {})
        llm_h, llm_lv, llm_role = d.get("is_heading"), d.get("level"), d.get("role", "")
        r = _hdr(line, raw, text)

        # 1. 目录区
        if _in_toc(line):
            if _is_toc_marker(text):
                r.update(is_heading=True, level=2, role="toc_marker", reason="目录标记")
            else:
                r.update(role="toc_entry", reason="目录条目")
            final.append(r); continue

        # 2. 文档主标题 (LLM 判定)
        if llm_role == "document_title":
            r.update(is_heading=True, level=1, role="document_title", reason="主标题")
            final.append(r); continue

        # 3-12. 规则判定
        if _is_ch_chapter(text):
            r.update(is_heading=True, level=2, role="chapter", reason="第X章")

        elif (dv := _depth(text)) > 0:
            r.update(is_heading=True, level=min(dv + 1, 6),
                     role={1: "section", 2: "subsection"}.get(dv, "subsubsection"),
                     reason=f"编号 depth={dv}")

        elif _key(text) in {"摘要", "abstract"}:
            r.update(is_heading=True, level=2, role="abstract", reason="摘要")

        elif _key(text) in {"参考文献", "references", "reference", "references and notes"}:
            r.update(is_heading=True, level=2, role="references", reason="参考文献")

        elif _key(text) in {"致谢", "acknowledgements", "acknowledgments"}:
            r.update(is_heading=True, level=2, role="acknowledgements", reason="致谢")

        elif _key(text) in {"附录", "appendix"}:
            r.update(is_heading=True, level=2, role="appendix", reason="附录")

        elif _key(text) in MAJOR:
            r.update(is_heading=True, level=2, role="section", reason="英文主章节")

        elif is_thesis and _is_paren(text):
            r.update(role="paragraph", reason="学位论文括号编号")

        elif llm_h is True and isinstance(llm_lv, int) and 1 <= llm_lv <= 6:
            r.update(is_heading=True, level=llm_lv, role=llm_role or "section",
                     reason=f"LLM判定: {d.get('reason','')}")

        elif llm_h is False:
            r.update(role=llm_role or "paragraph", reason=f"LLM判定非标题: {d.get('reason','')}")

        else:
            r.update(role="unknown", reason="LLM未明确判定")

        final.append(r)

    _fix_gaps(final)
    return final


# ============================================================
# 第二层: 辅助规则函数
# ============================================================


def _is_thesis(headings: list[dict]) -> bool:
    """是否中文学位论文"""
    titles = {h["text"].strip() for h in headings}
    ch = sum(1 for h in headings if re.match(r"^第[一二三四五六七八九十百零0-9]+[章节]", h["text"].strip()))
    return "目录" in titles and "摘要" in titles and ch >= 2 and bool({"参考文献", "致谢"} & titles)


def _toc_ranges(headings: list[dict]) -> list[dict]:
    """推测目录行号范围。基于标题重复检测 (目录条目 vs 正文标题)"""
    ranges = []
    for i, h in enumerate(headings):
        if clean_title(h["text"]).lower() not in {"目录", "contents", "table of contents"}:
            continue
        seen, end = {}, None
        for j in range(i + 1, len(headings)):
            key = clean_title(headings[j]["text"]).lower()
            if not key:
                continue
            if key in seen:
                end = headings[j - 1]["line"]; break
            seen[key] = headings[j]["line"]
        if end is None and seen:
            end = list(seen.values())[-1]
        if end and end > h["line"]:
            ranges.append({"start": h["line"], "end": end})
    return ranges


def _fix_gaps(final: list[dict]) -> None:
    """修复标题级别跳跃: H2→H4 改为 H2→H3"""
    h_only = [(i, r) for i, r in enumerate(final) if r["is_heading"]]
    for i in range(1, len(h_only)):
        prev_lv = h_only[i - 1][1]["level"]
        curr = h_only[i][1]
        if curr["level"] > prev_lv + 1:
            curr["level"] = prev_lv + 1
            curr["reason"] += "; 跳跃修复"


# ============================================================
# 应用判定
# ============================================================


def apply_decisions(md_text: str, decisions: list[dict]) -> str:
    """根据判定重写 # 标题: heading → 正确级别, 非标题 → 去 #"""
    lines = md_text.split("\n")
    decision_by = {d["line"]: d for d in decisions}
    pat = re.compile(r"^(#{1,6})\s+(.+)")

    result = []
    for i, line_text in enumerate(lines):
        d = decision_by.get(i + 1)
        m = pat.match(line_text.strip())
        if m and d:
            if d["is_heading"] and d["level"]:
                result.append(f"{'#' * d['level']} {m.group(2)}")
            else:
                result.append(m.group(2))
        else:
            result.append(line_text)
    return "\n".join(result)


# ============================================================
# 缓存
# ============================================================


def _cache_path(cache_dir: str, file_name: str, idx: int, batch: list[dict]) -> Path:
    stem = Path(file_name).stem[:80]
    digest = hashlib.sha1(str(Path(file_name).resolve()).encode()).hexdigest()[:10]
    return Path(cache_dir) / f"{stem}.{digest}.batch{idx:03d}.{batch[0]['line']}-{batch[-1]['line']}.json"


def _load_cache(cache_dir: str, file_name: str, idx: int, batch: list[dict]) -> list[dict] | None:
    path = _cache_path(cache_dir, file_name, idx, batch)
    try:
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            decisions = data.get("decisions", [])
            if len(decisions) == len(batch):
                return decisions
    except Exception:
        pass
    return None


def _save_cache(cache_dir: str, file_name: str, idx: int, batch: list[dict], decisions: list[dict]) -> None:
    path = _cache_path(cache_dir, file_name, idx, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"decisions": decisions}, ensure_ascii=False, indent=2), "utf-8")


# ============================================================
# 主入口
# ============================================================


def process_file(input_path: str, output_path: str = "", json_path: str = "",
                 cache_dir: str = "") -> dict:
    """单文件完整流水线"""
    in_path = Path(input_path)
    output_path = output_path or str(Path("standardized_md2") / in_path.name)
    json_path = json_path or str(Path("heading_json2") / f"{in_path.stem}.headings.json")

    print(f"处理: {in_path.name}")
    md_text = in_path.read_text("utf-8")
    headings = extract_headings(md_text)
    print(f"  {len(headings)} 个 # 候选")

    decisions = layer1_classify(headings, in_path.name, cache_dir=cache_dir)
    n_llm = sum(1 for d in decisions if d.get("is_heading"))
    print(f"  LLM: {n_llm} 标题, {len(decisions)-n_llm} 非标题")

    final = layer2_rebuild(headings, decisions)
    n_final = sum(1 for d in final if d["is_heading"])
    print(f"  规则: {n_final} 标题, {len(final)-n_final} 非标题")

    fixed = apply_decisions(md_text, final)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(fixed, "utf-8")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(
        json.dumps({"file_name": in_path.name, "decisions": final}, ensure_ascii=False, indent=2),
        "utf-8")
    print(f"  → {output_path}")

    return {"input": input_path, "output": output_path, "json": json_path,
            "candidates": len(headings), "headings_llm": n_llm, "headings_final": n_final}


def find_md_files(paths: list[str]) -> list[str]:
    """查找 .md 文件"""
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
    ap = argparse.ArgumentParser(description="Markdown 标题标准化 (两层架构)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out-dir", default="standardized_md2")
    ap.add_argument("--json-dir", default="heading_json2")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    files = find_md_files(args.paths)
    if not files:
        print("没有 .md 文件"); sys.exit(1)

    for f in files:
        out = str(Path(args.out_dir) / Path(f).name)
        js = str(Path(args.json_dir) / f"{Path(f).stem}.headings.json")
        if args.resume and Path(out).exists() and Path(js).exists():
            print(f"跳过: {Path(f).name}"); continue
        process_file(f, out, js, cache_dir=args.cache_dir)
