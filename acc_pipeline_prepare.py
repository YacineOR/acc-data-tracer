#!/usr/bin/env python3
# Yacine Ould Rouis - BSC
import sys
import re
import json
from collections import defaultdict

# Accept:
# [11:45:20.123] ...
# [11:45:20] ...
# [123.456] ...
TIME_RE = re.compile(r'^\[(?:(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d+(?:\.\d+)?))\]')
RANK_RE = re.compile(r'^\[rank=(\d+|[^\]]+)\]\s*')
EVENT_RE = re.compile(r'^(create|alloc|upload|download|delete|attach)\s+CUDA data\s+(.*)$')
KV_RE = re.compile(r'(\w+)=([^\s]+)')
SD_RE = re.compile(r'\$sd\d+')

CLAUSE_RE = re.compile(
    r'\b('
    r'copyin|create|copyout|copy|present|deviceptr|attach|no_create|'
    r'present_or_copyin|present_or_create|present_or_copy|present_or_copyout'
    r')\s*\(',
    re.IGNORECASE,
)

source_cache = {}
directive_cache = {}


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def parse_time_prefix(line):
    m = TIME_RE.match(line)
    if not m:
        return None, line

    if m.group(1) is not None:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3))
        frac = m.group(4) or "0"
        t = hh * 3600 + mm * 60 + ss + float("0." + frac)
    else:
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


def site_key_of(rank, file_, line, fn):
    return f"{rank}|{file_}|{line}|{fn}"


def burst_key_of(ev):
    return (ev["rank"], ev["file"], ev["line"], ev["function"])


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
    s = s.rstrip("\n").strip()
    s = re.sub(r'^\s*!\$acc\b', '', s, flags=re.IGNORECASE).strip()
    if s.startswith("&"):
        s = s[1:].lstrip()
    if s.endswith("&"):
        s = s[:-1].rstrip()
    return s


def _is_acc_line(s):
    return re.match(r'^\s*!\$acc\b', s, flags=re.IGNORECASE) is not None


def _extract_acc_directive_block(lines, start_idx):
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

        current_has_cont = stripped.rstrip().endswith("&")
        next_is_acc = i + 1 < n and _is_acc_line(lines[i + 1])

        if current_has_cont and next_is_acc:
            i += 1
            continue

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

    dedup = []
    seen = set()
    for x in out:
        k = re.sub(r'\s+', ' ', x).strip()
        if k and k not in seen:
            dedup.append(k)
            seen.add(k)
    return dedup


def read_acc_directive_near(file_path, line_num, back=12, forward=20):
    key = (file_path, line_num)
    if key in directive_cache:
        return directive_cache[key]

    lines = load_source_lines(file_path)
    if not lines or not line_num:
        out = {"vars": [], "directive_text": "", "directive_line": None, "directive_end_line": None}
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
                "directive_end_line": end_i + 1,
                "dist": dist,
            })
            i = end_i + 1
        else:
            i += 1

    if not candidates:
        out = {"vars": [], "directive_text": "", "directive_line": None, "directive_end_line": None}
        directive_cache[key] = out
        return out

    candidates.sort(key=lambda c: (c["dist"], abs(c["directive_line"] - line_num)))
    best = {
        "vars": candidates[0]["vars"],
        "directive_text": candidates[0]["directive_text"],
        "directive_line": candidates[0]["directive_line"],
        "directive_end_line": candidates[0]["directive_end_line"],
    }
    directive_cache[key] = best
    return best


def compress_var_list_for_name(vars_):
    cleaned = [re.sub(r'\s+', ' ', v).strip() for v in vars_ if v and not is_metadata_name(v)]
    if not cleaned:
        return None
    return ", ".join(cleaned)


def choose_name_from_source_or_hints(ev, create_ev=None, hint_ev=None):
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


def directive_kind_of_text(directive_text):
    txt = (directive_text or "").lower()
    if "enter data" in txt:
        return "enter"
    if "exit data" in txt:
        return "exit"
    return "other"


def classify_event_scope(ev):
    src = read_acc_directive_near(ev["file"], ev["line"], back=0, forward=0)
    directive_text = src.get("directive_text", "")
    directive_line = src.get("directive_line")
    directive_end_line = src.get("directive_end_line")
    kind = directive_kind_of_text(directive_text)

    # Strict mode: the event line must point directly into the exact OpenACC
    # directive block itself. For a continued directive, that means any line
    # inside the block, not just the first line.
    points_directly = (
        directive_line is not None
        and directive_end_line is not None
        and directive_line <= ev["line"] <= directive_end_line
    )

    if ev["action"] in ("create", "alloc", "upload", "attach"):
        return points_directly and kind == "enter", kind, directive_text, directive_line
    if ev["action"] in ("delete", "download"):
        return points_directly and kind == "exit", kind, directive_text, directive_line
    return False, kind, directive_text, directive_line


# -----------------------------------------------------------------------------
# Event parsing
# -----------------------------------------------------------------------------

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

            ev = {
                "idx": idx,
                "raw": raw,
                "action": action,
                "rank": rank,
                "file": file_,
                "function": fn,
                "line": line,
                "site_key": site_key_of(rank, file_, line, fn),
                "site_label": site_label(file_, line, fn),
                "burst_key": (rank, file_, line, fn),
                "devaddr": devaddr,
                "variable": variable,
                "bytes": bytes_,
                "t_abs": t_eff,
                "t_rel": t_eff - first_time,
            }

            in_scope, directive_kind, directive_text, directive_line = classify_event_scope(ev)
            ev["directive_scope_included"] = in_scope
            ev["directive_kind"] = directive_kind
            ev["directive_text_near"] = directive_text
            ev["directive_line_near"] = directive_line
            ev["directive_end_line_near"] = read_acc_directive_near(ev["file"], ev["line"], back=0, forward=0).get("directive_end_line")

            if in_scope:
                events.append(ev)

    return events, has_timestamps


# -----------------------------------------------------------------------------
# Tracking logic
# -----------------------------------------------------------------------------

def is_enter_like(ev):
    return ev["action"] in ("create", "alloc", "upload", "attach")


def is_exit_like(ev):
    return ev["action"] in ("delete", "download")


def build_bursts(events):
    bursts = []
    cur = []
    cur_key = None

    for ev in events:
        if is_enter_like(ev):
            key = burst_key_of(ev)
            if cur and key != cur_key:
                bursts.append(cur)
                cur = []
            cur.append(ev)
            cur_key = key
        else:
            if cur:
                bursts.append(cur)
                cur = []
                cur_key = None
    if cur:
        bursts.append(cur)
    return bursts


def choose_best_create_for_alloc(unmatched_creates, alloc_ev):
    best_idx = None
    best_slack = None
    alloc_size = alloc_ev.get("bytes")
    if alloc_size is None:
        return None

    for i, create_ev in enumerate(unmatched_creates):
        create_size = create_ev.get("bytes")
        if create_size is None:
            continue
        if alloc_size < create_size:
            continue
        slack = alloc_size - create_size
        if best_idx is None or slack < best_slack:
            best_idx = i
            best_slack = slack

    if best_idx is None:
        return None
    return unmatched_creates.pop(best_idx)


def choose_best_hint(events_in_burst, create_ev, alloc_ev):
    candidates = []
    for ev in events_in_burst:
        if ev["action"] not in ("upload", "attach"):
            continue
        if ev["idx"] < create_ev["idx"]:
            continue
        if ev["idx"] > alloc_ev["idx"]:
            continue
        score = 0
        if ev.get("bytes") == create_ev.get("bytes"):
            score -= 20
        if ev.get("variable") and not is_metadata_name(ev["variable"]):
            score -= 10
        elif ev.get("variable"):
            score -= 2
        score += abs(ev["idx"] - create_ev["idx"])
        candidates.append((score, ev))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]["idx"]))
    return candidates[0][1]


def make_track_base(track_id, name, name_source, directive_text, directive_line, create_ev, size, addr, kind, confidence):
    return {
        "id": track_id,
        "name": name,
        "name_source": name_source,
        "file": create_ev["file"],
        "function": create_ev["function"],
        "line": create_ev["line"],
        "site_label": create_ev["site_label"],
        "site_key": create_ev["site_key"],
        "rank": create_ev["rank"],
        "create_size": create_ev.get("bytes"),
        "size": size,
        "addr": addr,
        "start": create_ev["t_rel"],
        "end": None,
        "directive_text": directive_text,
        "directive_line": directive_line,
        "track_kind": kind,
        "inferred_confidence": confidence,
        "warnings": [],
        "anomaly": {
            "leaked": False,
            "overlap": False,
        },
    }


def overlap(a0, a1, b0, b1):
    return not (a1 <= b0 or b1 <= a0)


def reuse_key_of(rank, file_, fn, line, create_size):
    return (rank, file_, fn, line, create_size)


def infer_reuse_candidates(closed_first_by_key, reserved_reuse_addrs, live_exact_by_addr, create_ev):
    key = reuse_key_of(create_ev["rank"], create_ev["file"], create_ev["function"], create_ev["line"], create_ev.get("bytes"))
    prior = closed_first_by_key.get(key, [])
    viable = []
    for tr in prior:
        addr = tr.get("addr")
        if not isinstance(addr, str):
            continue
        if addr in live_exact_by_addr:
            continue
        if addr in reserved_reuse_addrs:
            continue
        viable.append(tr)
    return viable


def _reuse_delete_match_priority(tr):
    addr = tr.get("addr")
    if isinstance(addr, list):
        cand_len = len(addr)
    else:
        cand_len = 1
    return (cand_len, tr.get("start", 0.0), tr.get("id", 10**18))


def reconcile_ambiguous_delete(delete_ev, active_reuse_ambiguous, reserved_reuse_addrs):
    daddr = delete_ev.get("devaddr")
    dbytes = delete_ev.get("bytes")
    drank = delete_ev.get("rank")

    matches = []
    for tr in active_reuse_ambiguous:
        if tr["rank"] != drank:
            continue
        addr = tr.get("addr")
        if not isinstance(addr, list):
            continue
        if daddr not in addr:
            continue
        if tr.get("size") is not None and dbytes is not None and tr["size"] != dbytes:
            continue
        matches.append(tr)

    if not matches:
        return False

    matches.sort(key=_reuse_delete_match_priority)
    winner = matches[0]
    winner["addr"] = daddr
    winner["end"] = delete_ev["t_rel"]
    winner["inferred_confidence"] = "high"
    reserved_reuse_addrs.add(daddr)
    if winner in active_reuse_ambiguous:
        active_reuse_ambiguous.remove(winner)

    for other in list(active_reuse_ambiguous):
        addr = other.get("addr")
        if not isinstance(addr, list):
            continue
        if daddr in addr:
            other["addr"] = [x for x in addr if x != daddr]
            if len(other["addr"]) == 0:
                other["inferred_confidence"] = "low"
                other["warnings"].append(
                    f"candidate address {daddr} was assigned to another reuse track and no alternatives remain"
                )

    return True


def finalize_singleton_candidates(active_reuse_ambiguous, reserved_reuse_addrs):
    changed = True
    while changed:
        changed = False
        for tr in list(active_reuse_ambiguous):
            addr = tr.get("addr")
            if not isinstance(addr, list):
                continue

            filtered = [a for a in addr if a not in reserved_reuse_addrs]
            if len(filtered) != len(addr):
                tr["addr"] = filtered
                addr = filtered
                changed = True

            if isinstance(addr, list) and len(addr) == 0:
                tr["inferred_confidence"] = "low"
                continue

            if isinstance(addr, list) and len(addr) == 1:
                tr["addr"] = addr[0]
                tr["inferred_confidence"] = "high"
                reserved_reuse_addrs.add(addr[0])
                if tr in active_reuse_ambiguous:
                    active_reuse_ambiguous.remove(tr)
                changed = True


def prepare_tracks(events):
    tracks = []
    hierarchy_warnings = []
    burst_warnings = []

    live_exact_by_addr = {}
    # Global reuse pool per rank. Used only as a fallback when strict same-directive
    # reuse does not find anything. Global matching is based on same rank and same
    # create_size, so it is intentionally broader and lower-confidence.
    free_pool_by_rank = defaultdict(list)
    active_reuse_ambiguous = []
    reserved_reuse_addrs = set()
    closed_first_by_key = defaultdict(list)
    next_track_id = 0

    bursts = build_bursts(events)
    burst_iter = iter(bursts)
    current_burst = next(burst_iter, None)

    def register_live_exact_track(track):
        daddr = track["addr"]
        if not isinstance(daddr, str):
            return

        try:
            b0 = int(daddr, 16)
            b1 = b0 + (track["size"] or 0)
            for other in live_exact_by_addr.values():
                oaddr = other.get("addr")
                if not isinstance(oaddr, str):
                    continue
                a0 = int(oaddr, 16)
                a1 = a0 + (other.get("size") or 0)
                if overlap(a0, a1, b0, b1):
                    track["warnings"].append(
                        f"exact live allocation at address {daddr} overlaps another exact live allocation; this should not happen and suggests parser drift"
                    )
                    other["warnings"].append(
                        f"exact live allocation at address {oaddr} overlaps another exact live allocation; this should not happen and suggests parser drift"
                    )
        except Exception:
            pass

        if daddr in live_exact_by_addr:
            track["warnings"].append(
                f"address {daddr} was seen as a new live exact allocation while still considered live; this should not happen and suggests parser drift or inconsistent NV_ACC_NOTIFY ordering"
            )
        live_exact_by_addr[daddr] = track

    def reserve_reuse_track(track):
        addr = track["addr"]
        if isinstance(addr, str):
            if addr in reserved_reuse_addrs:
                track["warnings"].append(
                    f"reuse address {addr} was inferred twice while still reserved by another reuse track; this should not happen and suggests parser drift"
                )
                track["inferred_confidence"] = "low"
            reserved_reuse_addrs.add(addr)
            return

        if isinstance(addr, list):
            filtered = [a for a in addr if a not in reserved_reuse_addrs and a not in live_exact_by_addr]
            track["addr"] = filtered
            if len(filtered) == 1:
                track["addr"] = filtered[0]
                reserved_reuse_addrs.add(filtered[0])
                if track["inferred_confidence"] == "medium":
                    track["inferred_confidence"] = "high"
            elif len(filtered) == 0:
                track["warnings"].append(
                    "all reuse candidates were already reserved or live; dropping this reuse inference would be safer"
                )
                track["inferred_confidence"] = "low"

    def process_burst(burst):
        nonlocal next_track_id

        creates = [ev for ev in burst if ev["action"] == "create"]
        allocs = [ev for ev in burst if ev["action"] == "alloc"]
        unmatched_creates = list(creates)
        explicit_pairs = []
        burst_invalid_size = False

        for alloc_ev in allocs:
            chosen = choose_best_create_for_alloc(unmatched_creates, alloc_ev)
            if chosen is None:
                larger_remaining = [
                    c.get("bytes")
                    for c in unmatched_creates
                    if c.get("bytes") is not None
                    and alloc_ev.get("bytes") is not None
                    and c["bytes"] > alloc_ev["bytes"]
                ]
                if larger_remaining:
                    burst_invalid_size = True
                continue
            explicit_pairs.append((chosen, alloc_ev))

        if burst_invalid_size:
            burst_warnings.append({
                "rank": burst[0]["rank"],
                "file": burst[0]["file"],
                "function": burst[0]["function"],
                "line": burst[0]["line"],
                "create_sizes": [ev.get("bytes") for ev in creates],
                "alloc_sizes": [ev.get("bytes") for ev in allocs],
                "warning": "alloc_size < create_size would be required for some local burst matches; matching was left conservative",
            })

        for create_ev, alloc_ev in explicit_pairs:
            hint_ev = choose_best_hint(burst, create_ev, alloc_ev)
            name, name_source, directive_text, directive_line = choose_name_from_source_or_hints(
                alloc_ev, create_ev=create_ev, hint_ev=hint_ev
            )
            track = make_track_base(
                next_track_id,
                name,
                name_source,
                directive_text,
                directive_line,
                create_ev,
                alloc_ev.get("bytes"),
                alloc_ev.get("devaddr"),
                "first_alloc",
                "high",
            )
            next_track_id += 1
            tracks.append(track)
            register_live_exact_track(track)

        for create_ev in unmatched_creates:
            # Two-stage reuse search:
            #   1) strict local pool: same rank/file/function/line/create_size
            #   2) global fallback: same rank and same create_size anywhere in the run
            candidates = []
            seen_candidate_addr = set()
            reuse_mode = None

            # --- STRICT LOCAL (HIGH CONFIDENCE) ---
            strict_candidates = infer_reuse_candidates(
                closed_first_by_key, reserved_reuse_addrs, live_exact_by_addr, create_ev
            )
            for tr in strict_candidates:
                addr = tr.get("addr")
                if isinstance(addr, str) and addr not in seen_candidate_addr:
                    candidates.append(tr)
                    seen_candidate_addr.add(addr)

            if candidates:
                reuse_mode = "local"
            else:
                # --- GLOBAL FALLBACK (LOWER CONFIDENCE) ---
                for tr in free_pool_by_rank.get(create_ev.get("rank"), []):
                    addr = tr.get("addr")
                    if not isinstance(addr, str):
                        continue
                    if addr in seen_candidate_addr:
                        continue
                    if addr in live_exact_by_addr:
                        continue
                    if addr in reserved_reuse_addrs:
                        continue

                    # IMPORTANT: match on CREATE SIZE ONLY (not alloc size)
                    if tr.get("create_size") is None or create_ev.get("bytes") is None:
                        continue
                    if tr.get("create_size") != create_ev.get("bytes"):
                        continue

                    candidates.append(tr)
                    seen_candidate_addr.add(addr)

                if candidates:
                    reuse_mode = "global"

            if not candidates:
                continue

            candidate_addrs = []
            candidate_sizes = []
            seen_addr = set()
            for tr in candidates:
                addr = tr.get("addr")
                if not isinstance(addr, str) or addr in seen_addr:
                    continue
                seen_addr.add(addr)
                candidate_addrs.append(addr)
                candidate_sizes.append(tr.get("size"))

            if not candidate_addrs:
                continue

            # Size handling: use exact alloc size only if unique, otherwise leave undefined
            unique_sizes = sorted({s for s in candidate_sizes if s is not None})
            if len(unique_sizes) == 1:
                size = unique_sizes[0]
            else:
                size = None
            name, name_source, directive_text, directive_line = choose_name_from_source_or_hints(
                create_ev, create_ev=create_ev, hint_ev=None
            )

            track = make_track_base(
                next_track_id,
                name,
                name_source,
                directive_text,
                directive_line,
                create_ev,
                size,
                candidate_addrs[0] if len(candidate_addrs) == 1 else candidate_addrs,
                "reuse",
                "high" if len(candidate_addrs) == 1 else "medium",
            )
            track["reuse_from_closed_history"] = [tr.get("addr") for tr in candidates if isinstance(tr.get("addr"), str)]
            track["reuse_mode"] = reuse_mode
            if reuse_mode == "local":
                if track["inferred_confidence"] == "medium" and len(candidate_addrs) == 1:
                    track["inferred_confidence"] = "high"
            elif reuse_mode == "global":
                if track["inferred_confidence"] == "high":
                    track["inferred_confidence"] = "medium"
                else:
                    track["inferred_confidence"] = "low"
            next_track_id += 1

            if len(unique_sizes) > 1:
                track["warnings"].append(
                    f"candidate explicit history for create_size={create_ev.get('bytes')} had multiple alloc sizes {unique_sizes}; exported size uses a conservative representative"
                )
                if track["inferred_confidence"] == "high":
                    track["inferred_confidence"] = "medium"

            reserve_reuse_track(track)
            if track["inferred_confidence"] == "low" and (track.get("addr") == [] or track.get("addr") is None):
                continue

            tracks.append(track)
            if isinstance(track["addr"], list):
                active_reuse_ambiguous.append(track)

        finalize_singleton_candidates(active_reuse_ambiguous, reserved_reuse_addrs)

    for ev in events:
        while current_burst is not None and current_burst[-1]["idx"] < ev["idx"]:
            process_burst(current_burst)
            current_burst = next(burst_iter, None)

        if ev["action"] != "delete":
            continue

        daddr = ev.get("devaddr")
        if not daddr:
            continue

        if daddr in live_exact_by_addr:
            tr = live_exact_by_addr[daddr]
            tr["end"] = ev["t_rel"]
            del live_exact_by_addr[daddr]
            if tr["track_kind"] == "first_alloc":
                key_strict = reuse_key_of(
                    tr["rank"], tr["file"], tr["function"], tr["line"], tr.get("create_size")
                )
                closed_first_by_key[key_strict].append(tr)
            if isinstance(tr.get("addr"), str):
                free_pool_by_rank[tr.get("rank")].append(tr)
            continue

        if daddr in reserved_reuse_addrs:
            reserved_reuse_addrs.remove(daddr)
            for tr in tracks:
                if tr["track_kind"] == "reuse" and tr["end"] is None and tr.get("addr") == daddr:
                    tr["end"] = ev["t_rel"]
                    if isinstance(tr.get("addr"), str):
                        free_pool_by_rank[tr.get("rank")].append(tr)
                    break
            finalize_singleton_candidates(active_reuse_ambiguous, reserved_reuse_addrs)
            continue

        if reconcile_ambiguous_delete(ev, active_reuse_ambiguous, reserved_reuse_addrs):
            # winner has just been closed; make it available for later broader reuse
            for tr in tracks:
                if tr["track_kind"] == "reuse" and tr["end"] == ev["t_rel"] and tr.get("addr") == daddr:
                    if isinstance(tr.get("addr"), str):
                        free_pool_by_rank[tr.get("rank")].append(tr)
                    break
            finalize_singleton_candidates(active_reuse_ambiguous, reserved_reuse_addrs)
            continue

        burst_warnings.append({
            "rank": ev["rank"],
            "file": ev["file"],
            "function": ev["function"],
            "line": ev["line"],
            "delete_addr": daddr,
            "delete_size": ev.get("bytes"),
            "warning": "delete event did not match any currently live exact allocation and did not resolve any ambiguous reuse candidate; this suggests parser drift or unsupported log structure",
        })

    while current_burst is not None:
        process_burst(current_burst)
        current_burst = next(burst_iter, None)

    finalize_singleton_candidates(active_reuse_ambiguous, reserved_reuse_addrs)

    names = {tr["name"] for tr in tracks}
    for tr in tracks:
        if tr["end"] is None:
            tr["anomaly"]["leaked"] = True
            if tr["track_kind"] == "reuse" and isinstance(tr.get("addr"), str):
                reserved_reuse_addrs.discard(tr["addr"])
                if tr["inferred_confidence"] == "high":
                    tr["warnings"].append("reuse track inferred from prior closed history but not later confirmed by delete")
            elif tr["track_kind"] == "reuse" and isinstance(tr.get("addr"), list):
                if len(tr["addr"]) == 1:
                    tr["addr"] = tr["addr"][0]
                    tr["inferred_confidence"] = "high"
                else:
                    tr["inferred_confidence"] = "medium"

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

    tracks.sort(key=lambda x: (x["start"], x["id"]))
    end_t = max((ev["t_rel"] for ev in events), default=1.0)
    return tracks, hierarchy_warnings, burst_warnings, end_t


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) not in (2, 3):
        print("usage: python3 acc_pipeline_prepare.py in.log [timeline.json]", file=sys.stderr)
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) == 3 else "timeline.json"

    events, has_timestamps = parse_log(infile)
    tracks, hierarchy_warnings, burst_warnings, end_t = prepare_tracks(events)

    payload = {
        "meta": {
            "source_file": infile,
            "has_timestamps": has_timestamps,
            "num_events": len(events),
            "num_tracks": len(tracks),
            "end_t": end_t,
        },
        "groups": [],
        "tracks": tracks,
        "hierarchy_warnings": hierarchy_warnings,
        "burst_warnings": burst_warnings,
    }

    with open(outfile, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {outfile}")
    print(f"events={len(events)} tracks={len(tracks)} timestamps={has_timestamps}")


if __name__ == "__main__":
    main()
