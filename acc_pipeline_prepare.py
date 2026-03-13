#!/usr/bin/env python3
# Yacine Ould Rouis - BSC
import sys
import re
import json
from collections import defaultdict, deque

# Accept:
# [11:45:20.123] ...
# [11:45:20] ...
# [123.456] ...
TIME_RE  = re.compile(r'^\[(?:(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d+(?:\.\d+)?))\]')
RANK_RE  = re.compile(r'^\[rank=(\d+|[^\]]+)\]\s*')
EVENT_RE = re.compile(r'^(create|alloc|upload|download|delete|attach)\s+CUDA data\s+(.*)$')
KV_RE    = re.compile(r'(\w+)=([^\s]+)')
SD_RE    = re.compile(r'\$sd\d+')

CLAUSE_RE = re.compile(
    r'\b('
    r'copyin|create|copyout|copy|present|deviceptr|attach|no_create|'
    r'present_or_copyin|present_or_create|present_or_copy|present_or_copyout'
    r')\s*\(',
    re.IGNORECASE
)

source_cache = {}
directive_cache = {}


def parse_time_prefix(line):
    m = TIME_RE.match(line)
    if not m:
        return None, line

    # format: [HH:MM:SS(.mmm)]
    if m.group(1) is not None:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3))
        frac = m.group(4) or "0"
        t = hh * 3600 + mm * 60 + ss + float("0." + frac)
    else:
        # format: [seconds_since_start]
        t = float(m.group(5))

    return t, line[m.end():]

def parse_rank_prefix(line):
    m = RANK_RE.match(line)
    if not m:
        return None, line

    rank_txt = m.group(1)

    try:
        rank = int(rank_txt)
    except Exception:
        rank = rank_txt

    return rank, line[m.end():]

def parse_kv(text):
    out = {}
    for k, v in KV_RE.findall(text):
        out[k] = v
    return out


def short_file(path):
    parts = path.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def site_label(file_, line, fn):
    return f"{short_file(file_)}:{line} ({fn})"


def site_key_of(file_, line, fn):
    return f"{file_}|{line}|{fn}"


def is_metadata_name(name):
    if not name:
        return True
    return (
        name == "descriptor"
        or name == ".attach."
        or SD_RE.search(name) is not None
        or name.startswith("_")
    )


def split_hierarchy(name):
    if not name or name.startswith("site:"):
        return [name]
    clean = re.sub(r'\(.*$', '', name)
    parts = clean.split("%")
    out = []
    for i in range(1, len(parts) + 1):
        out.append("%".join(parts[:i]).strip())
    return out


def load_source_lines(path):
    if path in source_cache:
        return source_cache[path]
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        lines = None
    source_cache[path] = lines
    return lines


def split_variables(var_string):
    """
    Split a variable list like:
        a, b(i,j), c % x, foo(bar(1,2), k)
    at top-level commas only.
    """
    variables = []
    cur = []
    depth = 0

    for ch in var_string:
        if ch == ',' and depth == 0:
            s = ''.join(cur).strip()
            if s:
                variables.append(s)
            cur = []
            continue

        cur.append(ch)

        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)

    s = ''.join(cur).strip()
    if s:
        variables.append(s)

    return variables


def _normalize_acc_line_piece(s):
    """
    Remove OpenACC prefix and trailing continuation marker while keeping content.
    """
    s = s.rstrip("\n").strip()

    # Remove leading !$acc, case-insensitive
    s = re.sub(r'^\s*!\$acc\b', '', s, flags=re.IGNORECASE).strip()

    # Remove leading continuation '&'
    if s.startswith("&"):
        s = s[1:].lstrip()

    # Remove trailing continuation '&'
    if s.endswith("&"):
        s = s[:-1].rstrip()

    return s


def _is_acc_line(s):
    return re.match(r'^\s*!\$acc\b', s, flags=re.IGNORECASE) is not None


def _extract_acc_directive_block(lines, start_idx):
    """
    Collect one logical OpenACC directive, including continuation lines such as:

        !$acc enter data create(a, b, &
        !$acc                  c, d) copyin(x, y)

    or variants where a continuation line may begin with '&'.

    Returns:
        (directive_text, last_idx)
    """
    pieces = []
    i = start_idx
    n = len(lines)

    while i < n:
        raw = lines[i].rstrip("\n")
        stripped = raw.strip()

        if i == start_idx:
            if not _is_acc_line(stripped):
                break
        else:
            if not _is_acc_line(stripped):
                break

        pieces.append(_normalize_acc_line_piece(raw))

        # Decide if the directive continues.
        current_has_cont = stripped.rstrip().endswith("&")

        next_is_acc = False
        if i + 1 < n:
            next_is_acc = _is_acc_line(lines[i + 1])

        if current_has_cont and next_is_acc:
            i += 1
            continue

        # Even without trailing &, some codes still keep the same directive on
        # immediately following !$acc continuation-looking lines. We stop unless
        # the parenthesis balance is still open.
        text_so_far = " ".join(pieces)
        paren_balance = 0
        for ch in text_so_far:
            if ch == '(':
                paren_balance += 1
            elif ch == ')':
                paren_balance -= 1

        if paren_balance > 0 and next_is_acc:
            i += 1
            continue

        break

    directive = " ".join(p for p in pieces if p)
    directive = re.sub(r'\s+', ' ', directive).strip()
    return directive, i


def extract_variables_from_directive(directive):
    out = []
    pos = 0

    while True:
        m = CLAUSE_RE.search(directive, pos)
        if not m:
            break

        i = m.end()
        depth = 1
        j = i

        while j < len(directive) and depth > 0:
            if directive[j] == '(':
                depth += 1
            elif directive[j] == ')':
                depth -= 1
            j += 1

        if depth == 0:
            clause_content = directive[i:j - 1]
            out.extend(split_variables(clause_content))

        pos = j

    # Deduplicate while preserving order
    dedup = []
    seen = set()
    for x in out:
        k = re.sub(r'\s+', ' ', x).strip()
        if k and k not in seen:
            dedup.append(k)
            seen.add(k)

    return dedup


def read_acc_directive_near(file_path, line_num, back=12, forward=20):
    """
    Look around line_num and return:
      {
        "vars": [...],
        "directive_text": "...",
        "directive_line": <1-based start line>
      }
    for the closest OpenACC directive block.
    """
    key = (file_path, line_num)
    if key in directive_cache:
        return directive_cache[key]

    lines = load_source_lines(file_path)
    if not lines or not line_num:
        out = {"vars": [], "directive_text": "", "directive_line": None}
        directive_cache[key] = out
        return out

    idx_target = max(0, min(len(lines) - 1, line_num - 1))
    idx0 = max(0, idx_target - back)
    idx1 = min(len(lines) - 1, idx_target + forward)

    candidates = []

    i = idx0
    while i <= idx1:
        if _is_acc_line(lines[i]):
            directive, end_i = _extract_acc_directive_block(lines, i)
            vars_ = extract_variables_from_directive(directive)

            # Distance from requested line to directive span
            if i <= idx_target <= end_i:
                dist = 0
            elif idx_target < i:
                dist = i - idx_target
            else:
                dist = idx_target - end_i

            candidates.append({
                "vars": vars_,
                "directive_text": directive,
                "directive_line": i + 1,
                "start_idx": i,
                "end_idx": end_i,
                "dist": dist,
            })
            i = end_i + 1
        else:
            i += 1

    if not candidates:
        out = {"vars": [], "directive_text": "", "directive_line": None}
        directive_cache[key] = out
        return out

    candidates.sort(key=lambda c: (c["dist"], abs(c["directive_line"] - line_num)))
    best = {
        "vars": candidates[0]["vars"],
        "directive_text": candidates[0]["directive_text"],
        "directive_line": candidates[0]["directive_line"],
    }
    directive_cache[key] = best
    return best


def compress_var_list_for_name(vars_):
    """
    When ambiguous, store the whole list in 'name', exactly as requested.
    """
    cleaned = [re.sub(r'\s+', ' ', v).strip() for v in vars_ if v and not is_metadata_name(v)]
    if not cleaned:
        return None
    return ", ".join(cleaned)


def choose_name_from_source_or_hints(ev, create_ev, hint_ev):
    src = read_acc_directive_near(ev["file"], ev["line"])
    src_vars = src["vars"]
    directive_text = src["directive_text"]
    directive_line = src["directive_line"]

    if src_vars:
        non_meta = [v for v in src_vars if not is_metadata_name(v)]

        if len(non_meta) == 1:
            return non_meta[0], "source_single", directive_text, directive_line

        if hint_ev and hint_ev.get("variable"):
            hint_var = hint_ev["variable"]
            if hint_var in src_vars:
                return hint_var, "source_hint_match", directive_text, directive_line

        if create_ev and create_ev.get("variable"):
            create_var = create_ev["variable"]
            if create_var in src_vars:
                return create_var, "source_create_match", directive_text, directive_line

        if len(non_meta) > 1:
            # IMPORTANT: ambiguous => put whole source list in 'name'
            return compress_var_list_for_name(non_meta), "source_list", directive_text, directive_line

    if hint_ev and hint_ev.get("variable") and not is_metadata_name(hint_ev["variable"]):
        return hint_ev["variable"], "upload_match", directive_text, directive_line

    if create_ev and create_ev.get("variable") and not is_metadata_name(create_ev["variable"]):
        return create_ev["variable"], "create_variable", directive_text, directive_line

    if hint_ev and hint_ev.get("variable"):
        return hint_ev["variable"], "upload_match_metadata", directive_text, directive_line

    if create_ev and create_ev.get("variable"):
        return create_ev["variable"], "create_variable_metadata", directive_text, directive_line

    return f"site:{ev['site_label']}", "site_fallback", directive_text, directive_line


def parse_log(path):
    events = []
    first_time = None
    synthetic = 0
    has_timestamps = False
    same_time_count = defaultdict(int)

    with open(path, "r", errors="ignore") as f:
        for idx, raw in enumerate(f):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue

            t_abs, body = parse_time_prefix(raw)
            rank, body = parse_rank_prefix(body)
            if t_abs is not None:
                has_timestamps = True

            m = EVENT_RE.match(body)
            if not m:
                continue

            action = m.group(1)
            kv = parse_kv(m.group(2))

            file_ = kv.get("file", "")
            fn = kv.get("function", "")
            line = int(kv["line"]) if "line" in kv and kv["line"].isdigit() else None
            devaddr = kv.get("devaddr")
            variable = kv.get("variable")
            bytes_ = int(kv["bytes"]) if "bytes" in kv and kv["bytes"].isdigit() else None

            if t_abs is None:
                t_eff = float(synthetic)
                synthetic += 1
            else:
                k = same_time_count[t_abs]
                same_time_count[t_abs] += 1
                t_eff = float(t_abs) + 1e-9 * k

            if first_time is None:
                first_time = t_eff

            events.append({
                "idx": idx,
                "raw": raw,
                "action": action,
                "rank": rank,
                "file": file_,
                "function": fn,
                "line": line,
                "site_key": site_key_of(file_, line, fn),
                "site_label": site_label(file_, line, fn),
                "devaddr": devaddr,
                "variable": variable,
                "bytes": bytes_,
                "t_abs": t_eff,
                "t_rel": t_eff - first_time,
            })

    return events, has_timestamps


def _site_matches(ev, file_, fn):
    return ev["file"] == file_ and ev["function"] == fn


def _line_distance(ev, line):
    if ev.get("line") is None or line is None:
        return 10**9
    return abs(ev["line"] - line)


def _pop_best_create(pending_create, ev):
    """
    More permissive than exact site_key:
    prefer same file/function and nearest line.
    """
    exact_q = pending_create[ev["site_key"]]
    if exact_q:
        return exact_q.popleft()

    best_bucket = None
    best_score = None

    for key, q in pending_create.items():
        if not q:
            continue
        cand = q[0]
        if not _site_matches(cand, ev["file"], ev["function"]):
            continue

        score = _line_distance(cand, ev["line"])
        if best_score is None or score < best_score:
            best_score = score
            best_bucket = key

    if best_bucket is not None and best_score is not None and best_score <= 8:
        return pending_create[best_bucket].popleft()

    return None


def _take_best_hint(pending_upload, ev, create_ev):
    """
    Look for the best upload/attach hint using:
      1) exact site_key
      2) same file+function, near line
      3) byte compatibility
      4) non-metadata preference
    """
    candidates = []

    def add_candidates_from_queue(q, penalty_site):
        for item in q:
            score = 0
            score += penalty_site
            score += min(_line_distance(item, ev["line"]), 50)

            # Prefer exact byte matches
            if ev.get("bytes") is not None and item.get("bytes") == ev["bytes"]:
                score -= 20

            if create_ev and create_ev.get("bytes") is not None and item.get("bytes") == create_ev["bytes"]:
                score -= 20

            # Prefer named, non-metadata variables
            if item.get("variable"):
                score -= 5
                if not is_metadata_name(item["variable"]):
                    score -= 10
            else:
                score += 20

            candidates.append((score, item))

    exact_q = pending_upload[ev["site_key"]]
    if exact_q:
        add_candidates_from_queue(exact_q, penalty_site=0)

    for key, q in pending_upload.items():
        if key == ev["site_key"] or not q:
            continue
        first = q[0]
        if _site_matches(first, ev["file"], ev["function"]):
            add_candidates_from_queue(q, penalty_site=8)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best_score, best = candidates[0]

    if best_score > 60:
        return None
    return best


def prepare_tracks(events):
    pending_create = defaultdict(deque)
    pending_upload = defaultdict(deque)
    live_by_addr = {}
    tracks = []

    for ev in events:
        act = ev["action"]
        site_key = ev["site_key"]

        if act == "create":
            pending_create[site_key].append(ev)
            continue

        if act in ("upload", "attach"):
            pending_upload[site_key].append(ev)
            continue

        if act == "alloc" and ev["devaddr"]:
            create_ev = _pop_best_create(pending_create, ev)
            hint_ev = _take_best_hint(pending_upload, ev, create_ev)

            label, name_source, directive_text, directive_line = choose_name_from_source_or_hints(
                ev, create_ev, hint_ev
            )

            reused_addr_while_live = ev["devaddr"] in live_by_addr

            track = {
                "id": len(tracks),
                "name": label,
                "name_source": name_source,
                "file": ev["file"],
                "function": ev["function"],
                "line": ev["line"],
                "site_label": ev["site_label"],
                "site_key": site_key,
                "addr": ev["devaddr"],
                "size": ev["bytes"] or 0,
                "start": ev["t_rel"],
                "end": None,
                "rank": ev.get("rank"),
                "directive_text": directive_text,
                "directive_line": directive_line,
                "anomaly": {
                    "overlap": False,
                    "leaked": False,
                    "reused_addr_while_live": reused_addr_while_live,
                },
            }

            # Address overlap check by range
            try:
                b0 = int(track["addr"], 16)
                b1 = b0 + track["size"]
            except Exception:
                b0 = None
                b1 = None

            if b0 is not None:
                for other in live_by_addr.values():
                    try:
                        a0 = int(other["addr"], 16)
                        a1 = a0 + other["size"]
                    except Exception:
                        continue

                    if not (b1 <= a0 or a1 <= b0):
                        track["anomaly"]["overlap"] = True
                        other["anomaly"]["overlap"] = True

            # If the same address is allocated again while still live,
            # close the previous one as anomalous-reused and replace it.
            if reused_addr_while_live:
                prev = live_by_addr[ev["devaddr"]]
                prev["end"] = ev["t_rel"]

            live_by_addr[track["addr"]] = track
            tracks.append(track)
            continue

        if act == "delete" and ev["devaddr"]:
            tr = live_by_addr.get(ev["devaddr"])
            if tr is not None:
                tr["end"] = ev["t_rel"]
                del live_by_addr[ev["devaddr"]]

    end_t = max((ev["t_rel"] for ev in events), default=1.0)

    for tr in live_by_addr.values():
        tr["end"] = None
        tr["anomaly"]["leaked"] = True

    names = {tr["name"] for tr in tracks}
    hierarchy_warnings = []

    for tr in tracks:
        parts = split_hierarchy(tr["name"])
        if len(parts) > 1:
            for i in range(1, len(parts)):
                parent = parts[i - 1]
                if parent not in names:
                    hierarchy_warnings.append({
                        "child": tr["name"],
                        "missing_parent": parent,
                        "site_label": tr["site_label"],
                    })
                    break

    return tracks, hierarchy_warnings, end_t


def group_tracks(tracks):
    groups = defaultdict(list)
    for tr in tracks:
        groups[tr["name"]].append(tr)

    out = []
    for name, rows in groups.items():
        rows = sorted(rows, key=lambda x: (x["start"], x["id"]))
        out.append({
            "name": name,
            "rows": rows,
            "first_start": rows[0]["start"],
            "file": rows[0]["file"],
            "function": rows[0]["function"],
            "line": rows[0]["line"],
            "site_label": rows[0]["site_label"],
        })

    out.sort(key=lambda x: (x["first_start"], x["name"]))
    return out


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: python3 acc_pipeline_prepare2.py out [timeline.json]", file=sys.stderr)
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) == 3 else "timeline.json"

    events, has_timestamps = parse_log(infile)
    tracks, hierarchy_warnings, end_t = prepare_tracks(events)
    groups = group_tracks(tracks)

    payload = {
        "meta": {
            "source_file": infile,
            "has_timestamps": has_timestamps,
            "num_events": len(events),
            "num_tracks": len(tracks),
            "end_t": end_t,
        },
        "groups": groups,
        "hierarchy_warnings": hierarchy_warnings,
    }

    with open(outfile, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {outfile}")
    print(f"events={len(events)} tracks={len(tracks)} timestamps={has_timestamps}")


if __name__ == "__main__":
    main()
