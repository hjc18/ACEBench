import json
import re
from typing import Dict, Any, List, Optional, Tuple
import os, sys

FUNC_CALL_RE = re.compile(
    r"""
    ^\s*
    \[
        \s*
        (?P<name>[A-Za-z_]\w*)
        \s*\(
            (?P<args>.*)
        \)
        \s*
    \]
    \s*$
    """,
    re.VERBOSE | re.DOTALL,
)

IDENT_BRACKET_RE = re.compile(
    r"""
    ^\s*
    \[
        \s*
        (?P<name>[A-Za-z_]\w*)
        \s*
    \]
    \s*$
    """,
    re.VERBOSE | re.DOTALL,
)

BARE_CALL_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z_]\w*)
    \s*\(
        (?P<args>.*)
    \)
    \s*$
    """,
    re.VERBOSE | re.DOTALL,
)

IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def _strip_ws(s: str) -> str:
    return s.strip()


def _remove_single_think_block(text: str) -> Tuple[str, Optional[Tuple[int, int]], Dict[str, Any]]:
    open_cnt = len(re.findall(r"<think>", text))
    close_cnt = len(re.findall(r"</think>", text))
    details = {"open_count": open_cnt, "close_count": close_cnt}

    if open_cnt == 0 and close_cnt == 0:
        return text, None, {"status": "none", "details": details, "content": None}

    if open_cnt == 1 and close_cnt == 1:
        m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if not m:
            return text, None, {"status": "incomplete", "details": details, "content": None}
        content = m.group(1)
        status = "empty" if _strip_ws(content) == "" else "normal"
        span = (m.start(), m.end())
        outside = text[:span[0]] + text[span[1]:]
        return outside, span, {"status": status, "details": details, "content": content}

    status = "incomplete" if open_cnt != close_cnt else "multiple"
    return text, None, {"status": status, "details": details, "content": None}


def _split_top_level_commas(s: str, keep_empty: bool = False) -> List[str]:
    """
    Split by commas, but only at top level: ignores commas inside (), [], {} and inside quotes.
    This accepts argument values like:
      vaccineTypes=['Pfizer', 'Moderna']
      foo={"a": [1,2], "b": "x,y"}
    """
    parts = []
    buf = []
    depth_paren = depth_brack = depth_brace = 0
    in_single = in_double = False
    esc = False

    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue

        if ch == "\\":
            buf.append(ch)
            esc = True
            continue

        if in_single:
            buf.append(ch)
            if ch == "'":
                in_single = False
            continue
        if in_double:
            buf.append(ch)
            if ch == '"':
                in_double = False
            continue

        if ch == "'":
            in_single = True
            buf.append(ch)
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            continue

        if ch == "(":
            depth_paren += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            buf.append(ch)
            continue
        if ch == "[":
            depth_brack += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth_brack = max(0, depth_brack - 1)
            buf.append(ch)
            continue
        if ch == "{":
            depth_brace += 1
            buf.append(ch)
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            buf.append(ch)
            continue

        if ch == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            part = "".join(buf).strip()
            if part or keep_empty:
                parts.append(part)
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail or keep_empty:
        parts.append(tail)
    return parts


def _balanced_delims(s: str) -> bool:
    """
    Quick sanity: ensure (), [], {} are balanced outside quotes.
    """
    depth_paren = depth_brack = depth_brace = 0
    in_single = in_double = False
    esc = False

    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue

        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if ch == '"':
                in_double = False
            continue

        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1

        if depth_paren < 0 or depth_brack < 0 or depth_brace < 0:
            return False

    return depth_paren == 0 and depth_brack == 0 and depth_brace == 0 and not in_single and not in_double and not esc


def check_llm_tool_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loose checker:
      1) <think> formatting: exactly one pair; status: normal/none/empty/multiple/incomplete
      2) Function call outside think: exactly one bracketed call (no extra text):
           [FuncName(key=VALUE, ...)]
         - VALUE can be ANY text (strings, numbers, lists, dicts, identifiers, etc.)
         - We do NOT parse values; we just validate 'key=...' pairs separated by top-level commas.
         status: normal/none/multiple/incomplete

    Returns diagnostics + extracted name + raw args + key->raw_value strings.
    """
    text = obj.get("result", "")
    if not isinstance(text, str):
        raise TypeError("obj['result'] must be a string")

    report: Dict[str, Any] = {
        "id": obj.get("id"),
        "think": {"status": None, "details": {}, "content": None},
        "call": {
            "status": None,
            "details": {},
            "name": None,
            "args_raw": None,
            "args_kv_raw": None,  # dict[str, str]
        },
    }

    outside, think_span, think_info = _remove_single_think_block(text)
    report["think"] = think_info

    outside_stripped = outside.strip()

    call_info = _check_function_call(outside_stripped)
    report["call"] = call_info
    return report


def _consume_identifier(text: str, i: int) -> Tuple[Optional[str], int]:
    if i >= len(text):
        return None, i
    ch = text[i]
    if not (ch.isalpha() or ch == "_"):
        return None, i
    j = i + 1
    while j < len(text):
        ch = text[j]
        if not (ch.isalnum() or ch == "_"):
            break
        j += 1
    return text[i:j], j


def _find_matching_bracket(text: str, start: int) -> Optional[int]:
    """
    Find the matching ']' for text[start] == '[' while ignoring delimiters inside quotes.
    Returns the closing index or None if no matching bracket exists.
    """
    if start >= len(text) or text[start] != "[":
        return None

    depth = 1
    i = start + 1
    in_sq = in_dq = esc = False

    while i < len(text):
        ch = text[i]
        if esc:
            esc = False
            i += 1
            continue
        if ch == "\\":
            esc = True
            i += 1
            continue
        if in_sq:
            if ch == "'":
                in_sq = False
            i += 1
            continue
        if in_dq:
            if ch == '"':
                in_dq = False
            i += 1
            continue
        if ch == "'":
            in_sq = True
            i += 1
            continue
        if ch == '"':
            in_dq = True
            i += 1
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
            if depth < 0:
                return None
        i += 1

    return None


def _scan_bracket_ident_blocks(text: str) -> Dict[str, Any]:
    """
    Scan bracket blocks that start like [Identifier ...] outside quotes.
    Returns:
      - blocks: list[{start, end, raw, has_open_paren}]
      - call_start_count: count of '[Identifier(' starts (balanced or not)
    """
    blocks: List[Dict[str, Any]] = []
    call_start_count = 0
    i = 0
    in_sq = in_dq = esc = False

    while i < len(text):
        ch = text[i]
        if esc:
            esc = False
            i += 1
            continue
        if ch == "\\":
            esc = True
            i += 1
            continue
        if in_sq:
            if ch == "'":
                in_sq = False
            i += 1
            continue
        if in_dq:
            if ch == '"':
                in_dq = False
            i += 1
            continue
        if ch == "'":
            in_sq = True
            i += 1
            continue
        if ch == '"':
            in_dq = True
            i += 1
            continue
        if ch != "[":
            i += 1
            continue

        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        _, j_after_ident = _consume_identifier(text, j)
        if j_after_ident == j:
            i += 1
            continue

        k = j_after_ident
        while k < len(text) and text[k].isspace():
            k += 1
        has_open_paren = k < len(text) and text[k] == "("
        if has_open_paren:
            call_start_count += 1

        end = _find_matching_bracket(text, i)
        if end is not None:
            blocks.append({
                "start": i,
                "end": end,
                "raw": text[i:end + 1],
                "has_open_paren": has_open_paren,
            })
            i = end + 1
            continue

        i += 1

    return {"blocks": blocks, "call_start_count": call_start_count}


def _parse_kv_args(args_raw: str) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
    """
    Parse key=value pairs from a raw argument string.
    Returns (kv_dict, error_details) — one of the two is always None.
    """
    stripped = args_raw.strip()
    if stripped == "":
        return {}, None

    if not _balanced_delims(stripped):
        return None, {"reason": "Unbalanced delimiters/quotes in arglist."}

    parts = _split_top_level_commas(stripped, keep_empty=True)
    kv: Dict[str, str] = {}

    for part in parts:
        if part == "":
            return None, {"reason": "Empty argument segment (possible trailing/consecutive comma)."}
        if "=" not in part:
            return None, {"reason": "Arg missing '='.", "bad_part": part}

        key, val = part.split("=", 1)
        key, val = key.strip(), val.strip()

        if not IDENT_RE.fullmatch(key):
            return None, {"reason": "Invalid key identifier.", "bad_part": part}
        if val == "":
            return None, {"reason": "Empty value.", "bad_part": part}
        if key in kv:
            return None, {"reason": "Duplicate key.", "bad_key": key}

        kv[key] = val

    return kv, None


def _check_function_call(text: str) -> Dict[str, Any]:
    """
    Validate that *text* is exactly one [FuncName(key=VALUE, ...)] call.

    Returns a call-report dict with keys: status, details, name, args_raw, args_kv_raw.
    Status is one of: normal, none, multiple, incomplete.
    """
    result: Dict[str, Any] = {
        "status": None,
        "details": {},
        "name": None,
        "args_raw": None,
        "args_kv_raw": None,
    }

    stripped = text.strip()
    if stripped == "":
        result["status"] = "none"
        return result

    scan = _scan_bracket_ident_blocks(stripped)
    blocks = scan["blocks"]
    call_like_blocks = [b for b in blocks if b["has_open_paren"]]
    unclosed_call_starts = max(0, scan["call_start_count"] - len(call_like_blocks))

    parsed_call_blocks: List[Tuple[Dict[str, Any], Any]] = []
    malformed_call_blocks: List[Dict[str, Any]] = []
    for block in call_like_blocks:
        bm = FUNC_CALL_RE.fullmatch(block["raw"])
        if bm is None:
            malformed_call_blocks.append(block)
            continue
        parsed_call_blocks.append((block, bm))

    whole_parsed_call_blocks = [
        (block, bm)
        for block, bm in parsed_call_blocks
        if block["start"] == 0 and block["end"] == len(stripped) - 1
    ]

    if len(whole_parsed_call_blocks) == 1 and len(parsed_call_blocks) == 1 and unclosed_call_starts == 0:
        _, m = whole_parsed_call_blocks[0]
        result["name"] = m.group("name")
        result["args_raw"] = m.group("args")

        kv, err = _parse_kv_args(result["args_raw"])
        if err is not None:
            result["status"] = "incomplete"
            result["details"] = err
            return result

        result["args_kv_raw"] = kv
        result["status"] = "normal"
        return result

    if len(parsed_call_blocks) > 1:
        result["status"] = "multiple"
        result["details"]["found_blocks"] = [b["raw"] for b, _ in parsed_call_blocks]
        return result

    if len(parsed_call_blocks) == 1:
        block, _ = parsed_call_blocks[0]
        result["status"] = "incomplete"
        result["details"] = {
            "reason": "Extra non-whitespace text outside [Name(...)] block.",
            "found_block": block["raw"],
        }
        prefix = stripped[:block["start"]].strip()
        suffix = stripped[block["end"] + 1:].strip()
        if prefix:
            result["details"]["prefix_text"] = prefix
        if suffix:
            result["details"]["suffix_text"] = suffix
        if unclosed_call_starts > 0:
            result["details"]["unclosed_call_starts"] = unclosed_call_starts
        return result

    if unclosed_call_starts > 0:
        result["status"] = "incomplete"
        result["details"] = {
            "reason": "Unclosed [Name(...)] block.",
            "unclosed_call_starts": unclosed_call_starts,
        }
        return result

    if malformed_call_blocks:
        result["status"] = "incomplete"
        result["details"] = {
            "reason": "Malformed [Name(...)] block.",
            "found_block": malformed_call_blocks[0]["raw"],
        }
        return result

    ident_bracket_match = IDENT_BRACKET_RE.fullmatch(stripped)
    if ident_bracket_match is not None:
        result["status"] = "incomplete"
        result["details"] = {
            "reason": "Bracketed function name is missing '(...)'.",
            "name": ident_bracket_match.group("name"),
        }
        return result

    bare_call_match = BARE_CALL_RE.fullmatch(stripped)
    if bare_call_match is not None:
        result["status"] = "incomplete"
        result["name"] = bare_call_match.group("name")
        result["args_raw"] = bare_call_match.group("args")
        result["details"] = {"reason": "Missing surrounding [ ... ] around function call."}
        return result

    if blocks:
        result["status"] = "incomplete"
        result["details"] = {
            "reason": "Bracketed identifier block is not a valid function-call format.",
            "found_block": blocks[0]["raw"],
        }
        return result

    result["status"] = "none"
    return result

if __name__ == "__main__":
    # Example usage

    # obj = {"id": "normal_atom_enum_43", "result": "<think>\nOkay, let's see. The user wants to analyze audience activity on Instagram, specifically in the Eastern Standard Time (EST) timezone. The conversation history shows that the user first asked about analyzing Instagram activity, and the system asked for the timezone. The user then provided EST.\n\nNow, I need to figure out which API to use. Looking at the available APIs, there's one called AudienceActivityAnalyzer_getActiveTimes. Its description says it provides analysis of active times on social media platforms. The parameters required are platform and timezone, with platform being mandatory. The user specified Instagram and EST, so that fits.\n\nSo, the API call should include platform='Instagram' and timezone='Eastern Standard Time (EST)'. The parameters are correctly formatted, and since the user provided both required details, I can construct the API request accordingly. No other APIs seem relevant here because the others are for contracts, education resources, or family mediation, which don't relate to social media activity analysis. Therefore, the correct API call is AudienceActivityAnalyzer_getActiveTimes with the given parameters.\n</think>\n\n[AudienceActivityAnalyzer_getActiveTimes(platform='Instagram', timezone='Eastern Standard Time (EST)')]"}
    # result = check_llm_tool_output(obj)
    # print(result)
    # quit()
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "syntax_check_samples"
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if not file.startswith("data_") or not file.endswith("_result.json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                examples = [json.loads(line) for line in f]
            diagnosis = []
            for ex in examples:
                report = check_llm_tool_output(ex)
                diagnosis.append(report)
            # stats
            status_counts = {
                "think": {},
                "call": {},
            }

            for r in diagnosis:
                tstatus = r["think"]["status"]
                cstatus = r["call"]["status"]
                status_counts["think"][tstatus] = status_counts["think"].get(tstatus, 0) + 1
                status_counts["call"][cstatus] = status_counts["call"].get(cstatus, 0) + 1
            diagnosis_file = os.path.join(root, f"diagnosis_{file}")
            with open(diagnosis_file, "w", encoding="utf-8") as f:
                json.dump({"status_counts": status_counts, "reports": diagnosis}, f, ensure_ascii=False, indent=2)
