#!/usr/bin/env python3
"""
judge_and_score.py
End-to-end: LLM judge + robust time-aware metrics in one pass.

- Extracts per-interval trends via:
  (A) LLM judge (default: gpt-4o-mini) with structured JSON output
  (B) Rule-based extractor (lexicon + negation/hedges/ranges) as backup
  Fusion: Prefer LLM unless UNKNOWN/low confidence; fall back to rules.

- Computes:
  * TEDS (Temporal Edit Distance Score; time-gap weighted)
  * Trend-F1 (↑/↓ events), SignAcc, Coverage, Chronology
  * TAC composite = 0.5*TEDS + 0.2*TrendF1 + 0.2*SignAcc + 0.1*Coverage

- Saves:
  * judge_trends.jsonl (intermediate, reusable cache)
  * results.csv and results.jsonl (metrics per case + macro)

Usage:
  python judge_and_score.py --gt gt.jsonl --pred preds.jsonl --out_dir out_eval
  # Options:
  #   --model gpt-4o-mini
  #   --temperature 0.0
  #   --max_retries 2
  #   --reuse_judge
  #   --no_time_weights
  #   --strict_stable_skip
  #   --no_tac
  #   --single_metric TEDS|TAC|TrendF1|SignAcc|Coverage|Chronology
  #   --single_only
  #   --dry_run (no API calls; writes placeholder judge results)

Inputs:
- GT and Pred: JSONL (one object/line) or a single JSON array.
- Match by patient_id (fallback: id).
"""

import argparse, csv, json, os, re, time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from statistics import median

# ------------- OpenAI client (pip install "openai>=1.51.0") -------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allow reuse-only/dry_run without SDK

# --------------------------------- Lexicon ------------------------------------
INCREASE_TERMS = [
    "increase","increased","increasing","progress","progressed","progression",
    "worse","worsened","larger","greater","extension","expansion","expanded",
    "intensified","marked","incremental","spread","peaked","more","greater than"
]
DECREASE_TERMS = [
    "decrease","decreased","decreasing","regress","regressed","regression",
    "reduce","reduced","improve","improved","improvement","resolve","resolved",
    "resolution","smaller","less","diminished","partial resolution","partially resolved"
]
STABLE_TERMS = [
    "stable","unchanged","similar","no change","no notable change",
    "no significant change","comparable","without interval change"
]
NEG_PAT = re.compile(r'\b(no|not|without|no further|no interval|absence of)\b', re.I)
HEDGE_POS = re.compile(r'\b(mild|slight|subtle|minimal|small)\b', re.I)
HEDGE_STRONG = re.compile(r'\b(marked|significant|pronounced|considerable)\b', re.I)

V_RE = re.compile(r'\bV\s?(\d+)\b', re.I)
PAIR_PATTERNS = [
    r'(V\s?\d+)[^\.]{0,60}(?:compared to|relative to|versus|vs\.?|vs)\s*(V\s?\d+)',
    r'(V\s?\d+)\s*[–-]\s*(V\s?\d+)',   # V3–V4
    r'(since|from)\s*(V\s?\d+)\b'      # since V1 ...
]

# ------------------------------ Basic IO utils --------------------------------
def load_any_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text: return []
    if text[0] == "[":
        data = json.loads(text)
        if isinstance(data, list): return data
        raise ValueError("Expected JSON array.")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def index_by_key(objs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        for k in ("patient_id", "id", "case_id", "study_id"):
            v = o.get(k)
            if v is None:
                continue
            key = str(v).strip()
            if key and key not in idx:
                idx[key] = o
    return idx


def normalize_step_text(x) -> str:
    """Coerce any step (str/dict/obj) into a plain string."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # common shapes: {"text": "..."} or {"content": "..."}
        for k in ("text", "content", "message", "step"):
            if k in x and isinstance(x[k], (str, int, float)):
                return str(x[k])
        # fallback: compact JSON
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    # list/tuple/number/etc.
    try:
        return json.dumps(x, ensure_ascii=False) if isinstance(x, (list, tuple)) else str(x)
    except Exception:
        return str(x)

def normalize_steps(steps) -> List[str]:
    return [normalize_step_text(s) for s in (steps or [])]


# ------------------------- Timepoints & intervals ------------------------------
DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')

def parse_timepoints(tp_list: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    visits = []
    for i, tp in enumerate(tp_list):
        label = tp.get("label", f"V{i+1}")
        mv = V_RE.search(label)
        vnum = mv.group(1) if mv else str(i+1)
        md = DATE_RE.search(label)
        d = md.group(1) if md else "-"
        visits.append(f"V{vnum} ({d})")
    intervals = [f"V{i+1}->V{i+2}" for i in range(max(0,len(visits)-1))]
    return visits, intervals

def parse_dates_from_gt(gt: Dict[str, Any]) -> List[Optional[datetime]]:
    dates=[]
    for tp in gt.get("timepoints", []):
        label = tp.get("label","")
        m = DATE_RE.search(label)
        if m:
            try: dates.append(datetime.strptime(m.group(1), "%Y-%m-%d"))
            except: dates.append(None)
        else:
            dates.append(None)
    return dates

def interval_weights_from_dates(dates: List[Optional[datetime]]) -> List[float]:
    if not dates or len(dates)<2: return []
    deltas=[]
    for i in range(len(dates)-1):
        if dates[i] and dates[i+1]:
            dd = abs((dates[i+1]-dates[i]).days)
            deltas.append(max(1,dd))
        else:
            deltas.append(1)
    if not deltas: return []
    med = median(deltas) if any(deltas) else 1
    if med<=0: med=1
    return [d/med for d in deltas]

# ---------------------- Rule-based arrow (↑/↓/→) with conf ---------------------
def detect_arrow_with_conf(text: str) -> Tuple[Optional[str], float, Optional[str]]:
    t = (text or "").lower()
    pos = any(term in t for term in INCREASE_TERMS)
    neg = any(term in t for term in DECREASE_TERMS)
    stab = any(term in t for term in STABLE_TERMS)
    has_negation = bool(NEG_PAT.search(t))

    arrow = None
    if neg and not pos: arrow = "↓"
    elif pos and not neg: arrow = "↑"
    elif stab or (has_negation and (pos or neg)): arrow = "→"

    conf = 0.5
    if HEDGE_STRONG.search(t): conf = 0.9
    elif HEDGE_POS.search(t): conf = 0.6
    if "partial" in t or "partially" in t: conf -= 0.1
    conf = max(0.1, min(conf, 0.95))

    if has_negation and arrow in ("↑","↓"):
        arrow = "→"; conf = max(conf,0.7)

    magnitude = None
    if HEDGE_STRONG.search(t): magnitude = "MARKED"
    elif HEDGE_POS.search(t): magnitude = "SLIGHT"

    return arrow, conf, magnitude


def extract_steps_from_pred(pred: Any) -> List[str]:
    # pred may be None, dict, or odd types
    if not isinstance(pred, dict):
        return []
    # parsed may be a dict or a JSON string
    parsed = pred.get("parsed")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        s = parsed.get("steps")
        if isinstance(s, list):
            return s
    # fallback to top-level steps
    s = pred.get("steps")
    return s if isinstance(s, list) else []


def extract_visits(text: str) -> List[int]:
    text = normalize_step_text(text)
    return [int(v) for v in V_RE.findall(text or "")]


def _extract_visit_num(tok: Optional[str]) -> Optional[int]:
    """Return visit number from tokens like 'V3' / 'V 3' / 'v3'; else None."""
    if not tok:
        return None
    m = re.search(r'V\s*(\d+)', str(tok), flags=re.I)
    return int(m.group(1)) if m else None


def rule_based_intervals(steps: List[str], T: int, overall_default: Optional[str]=None) -> Tuple[List[Optional[str]], List[float], List[Optional[str]], float]:
    """
    Returns per-interval arrow, confidence, magnitude, chronology score.
    Safe against malformed pairs like 'Vx' or missing digits.
    """
    steps = normalize_steps(steps) 

    pred = [None]*max(0,T-1)
    confs = [0.0]*max(0,T-1)
    mags  = [None]*max(0,T-1)

    first_visits_order=[]
    steps = steps or []

    for s in steps:
        s = s or ""
        vs = extract_visits(s)
        if vs:
            first_visits_order.append(vs[0])

        arrow, conf, mag = detect_arrow_with_conf(s)
        found = False

        # Pattern 1: "(V#) ... compared to/versus/vs (V#)"
        for m in re.finditer(PAIR_PATTERNS[0], s, flags=re.I):
            vL = _extract_visit_num(m.group(1))
            vR = _extract_visit_num(m.group(2))
            if vL is None or vR is None:
                continue
            a, b = sorted([vL, vR])
            for vv in range(a+1, b+1):
                idx = vv - 2  # interval (vv-1)->vv
                if 0 <= idx < len(pred) and arrow:
                    if conf > confs[idx]:
                        pred[idx] = arrow; confs[idx] = conf; mags[idx] = mag
            found = True

        # Pattern 2: "V# – V#" (dash range)
        for m in re.finditer(PAIR_PATTERNS[1], s, flags=re.I):
            vL = _extract_visit_num(m.group(1))
            vR = _extract_visit_num(m.group(2))
            if vL is None or vR is None:
                continue
            a, b = sorted([vL, vR])
            for vv in range(a+1, b+1):
                idx = vv - 2
                if 0 <= idx < len(pred) and arrow:
                    if conf > confs[idx]:
                        pred[idx] = arrow; confs[idx] = conf; mags[idx] = mag
            found = True

        # Pattern 3: "(since|from) V#"
        for m in re.finditer(PAIR_PATTERNS[2], s, flags=re.I):
            vstart = _extract_visit_num(m.group(2))
            if vstart is None:
                continue
            for vv in range(vstart+1, T+1):
                idx = vv - 2
                if 0 <= idx < len(pred) and arrow:
                    if conf > confs[idx]:
                        pred[idx] = arrow; confs[idx] = conf; mags[idx] = mag
            found = True

        # Fallbacks when no explicit pair/range matched
        if not found and vs and arrow:
            # Map each visit mention to its incoming interval
            for v in vs:
                idx = v - 2
                if 0 <= idx < len(pred):
                    if conf > confs[idx]:
                        pred[idx] = arrow; confs[idx] = conf; mags[idx] = mag

        if not found and len(vs) == 1 and "stable" in s.lower():
            v = vs[0]; idx = v - 2
            if 0 <= idx < len(pred) and conf > confs[idx]:
                pred[idx] = "→"; confs[idx] = conf; mags[idx] = mags[idx] or "NONE"

    # Backfill with overall default if provided
    if overall_default:
        for i in range(len(pred)):
            if pred[i] is None:
                pred[i] = overall_default
                confs[i] = max(confs[i], 0.55)

    chronology = 1.0 if first_visits_order == sorted(first_visits_order) else 0.0
    return pred, confs, mags, chronology



# -------------------------- LLM Judge (structured JSON) ------------------------
SYSTEM_PROMPT = """You are a meticulous radiology temporal-reasoning judge.
Given longitudinal MRI visits and intervals, the ground-truth question/answer/steps, and the model's steps,
return per-interval trends for the PRIMARY TARGET mentioned in the question.

Emit arrays of length exactly T-1 (intervals V_k->V_{k+1}).
Tokens:
- GT:     "UP" | "DOWN" | "STABLE"
- PRED:   "UP" | "DOWN" | "STABLE" | "UNKNOWN"
Also provide per-interval magnitude ("NONE"|"SLIGHT"|"MODERATE"|"MARKED") and confidence (0..1) for PRED,
a coverage list of which intervals are explicitly mentioned in model steps,
and a boolean for chronology order in the model narrative.

Rules:
- Consider ONLY the target pathology named in the question (e.g., edema/enhancement, mass effect, gliosis).
- Use negation correctly ("no further increase" => STABLE).
- Prefer explicit interval statements; apply overall trend across intervals only if needed and not contradicted.
- If interval evidence is insufficient/contradictory in model steps, use "UNKNOWN" for that PRED interval.
- Output STRICT JSON with the schema keys exactly."""

def build_user_payload(case: Dict[str, Any],
                       rb_hint: Dict[str, Any]) -> str:
    gt_q = " ".join(case.get("question", [])) if isinstance(case.get("question"), list) else (case.get("question") or "")
    visits, intervals = parse_timepoints(case.get("timepoints", []))
    gt_answer = case.get("answer", "")
    model_steps = case.get("_model_steps", [])
    gt_steps    = normalize_steps(case.get("steps", []))
    payload = {
        "instruction": "Extract per-interval trends for the primary target in the question.",
        "visits": visits,
        "intervals": intervals,
        "target_question": gt_q,
        "ground_truth": {"answer": gt_answer, "steps": gt_steps},
        "model_output": {"steps": model_steps},
        "rule_based_hint": rb_hint,  # helps the judge; may be overridden if contradictory
        "output_schema": {
            "type": "object",
            "required": ["intervals","gt_trends","pred_trends","pred_magnitude","pred_confidence","coverage","chronology_ok"],
            "properties": {
              "intervals":{"type":"array","items":{"type":"string"}},
              "gt_trends":{"type":"array","items":{"type":"string","enum":["UP","DOWN","STABLE"]}},
              "pred_trends":{"type":"array","items":{"type":"string","enum":["UP","DOWN","STABLE","UNKNOWN"]}},
              "pred_magnitude":{"type":"array","items":{"type":"string","enum":["NONE","SLIGHT","MODERATE","MARKED"]}},
              "pred_confidence":{"type":"array","items":{"type":"number"}},
              "coverage":{"type":"array","items":{"type":"string"}},
              "chronology_ok":{"type":"boolean"}
            }
        }
    }
    return json.dumps(payload, ensure_ascii=False)

def _align_arrays(data: Dict[str, Any], intervals: List[str]) -> Dict[str, Any]:
    """Pad/trim and normalize so arrays always match len(intervals)."""
    n = len(intervals)
    data["intervals"] = intervals[:]  # force
    # gt trends
    gt = list(data.get("gt_trends") or [])
    if len(gt) < n: gt += ["STABLE"]*(n-len(gt))
    data["gt_trends"] = [str(x).upper() if str(x).upper() in {"UP","DOWN","STABLE"} else "STABLE" for x in gt[:n]]
    # preds
    pt = list(data.get("pred_trends") or [])
    if len(pt) < n: pt += ["UNKNOWN"]*(n-len(pt))
    data["pred_trends"] = [str(x).upper() if str(x).upper() in {"UP","DOWN","STABLE","UNKNOWN"} else "UNKNOWN" for x in pt[:n]]
    pm = list(data.get("pred_magnitude") or [])
    if len(pm) < n: pm += ["NONE"]*(n-len(pm))
    data["pred_magnitude"] = [str(x).upper() if str(x).upper() in {"NONE","SLIGHT","MODERATE","MARKED"} else "NONE" for x in pm[:n]]
    pc = list(data.get("pred_confidence") or [])
    if len(pc) < n: pc += [0.5]*(n-len(pc))
    data["pred_confidence"] = [max(0.0, min(1.0, float(x))) if isinstance(x,(int,float,str)) and str(x).replace(".","",1).lstrip("-").isdigit() else 0.5 for x in pc[:n]]
    # coverage exactly equals intervals
    data["coverage"] = intervals[:]
    # chronology boolean
    data["chronology_ok"] = bool(data.get("chronology_ok", True))
    return data

def call_llm_judge(client, model: str, case: Dict[str, Any], rb_hint: Dict[str, Any],
                   temperature: float=0.0, max_retries: int=2, dry_run: bool=False) -> Dict[str, Any]:
    _, intervals = parse_timepoints(case.get("timepoints", []))
    if dry_run:
        return {
            "intervals": intervals,
            "gt_trends": ["STABLE"]*len(intervals),
            "pred_trends": ["UNKNOWN"]*len(intervals),
            "pred_magnitude": ["NONE"]*len(intervals),
            "pred_confidence": [0.0]*len(intervals),
            "coverage": intervals[:],
            "chronology_ok": True
        }
    req = {
        "model": model,
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": build_user_payload(case, rb_hint)}
        ],
        "temperature": temperature,
        # Ask for JSON; we still post-fix lengths to match intervals.
        "response_format": {"type":"json_object"}
    }
    last_err=None
    for _ in range(max_retries+1):
        try:
            resp = client.chat.completions.create(**req)
            text = resp.choices[0].message.content
            data = json.loads(text)
            # ensure required keys exist
            for k in ["intervals","gt_trends","pred_trends","pred_magnitude","pred_confidence","coverage","chronology_ok"]:
                if k not in data:
                    data[k] = [] if k not in ["chronology_ok"] else True
            return _align_arrays(data, intervals)
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"LLM judge failed after retries: {last_err}")

# ---------------------------- TEDS & companions --------------------------------
def teds_alignment(G: List[str], P: List[Optional[str]], weights: List[float],
                   strict_stable_skip: bool=False) -> Tuple[float,List[Tuple[Optional[int],Optional[int]]],float]:
    P_compact = [p for p in P if p is not None]
    n, m = len(G), len(P_compact)
    if len(weights)!=n: weights=[1.0]*n
    mean_w = sum(weights)/len(weights) if weights else 1.0

    def sub_cost(g: str, p: str, wi: float) -> float:
        if g==p: return 0.0
        if (g=="→" and p in ("↑","↓")) or (p=="→" and g in ("↑","↓")): return 1.0*wi
        if (g=="↑" and p=="↓") or (g=="↓" and p=="↑"): return 2.0*wi
        return 1.0*wi

    def gap_cost_gt(wi: float, gt_tok: Optional[str]) -> float:
        if (not strict_stable_skip) and gt_tok=="→": return 0.5*wi
        return 1.0*wi

    def gap_cost_pred() -> float:
        return 1.0*mean_w

    dp = [[0.0]*(m+1) for _ in range(n+1)]
    bt = [[(-1,-1)]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        dp[i][0]=dp[i-1][0]+gap_cost_gt(weights[i-1], G[i-1]); bt[i][0]=(i-1,0)
    for j in range(1,m+1):
        dp[0][j]=dp[0][j-1]+gap_cost_pred(); bt[0][j]=(0,j-1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            c_sub = dp[i-1][j-1] + sub_cost(G[i-1], P_compact[j-1], weights[i-1])
            c_del = dp[i-1][j]   + gap_cost_gt(weights[i-1], G[i-1])
            c_ins = dp[i][j-1]   + gap_cost_pred()
            best = min(c_sub,c_del,c_ins)
            dp[i][j]=best
            bt[i][j] = (i-1,j-1) if best==c_sub else ((i-1,j) if best==c_del else (i,j-1))

    align=[]
    i,j=n,m
    while not (i==0 and j==0):
        pi,pj = bt[i][j]
        if   pi==i-1 and pj==j-1: align.append((i-1,j-1))
        elif pi==i-1 and pj==j  : align.append((i-1,None))
        elif pi==i   and pj==j-1: align.append((None,j-1))
        else: break
        i,j=pi,pj
    align.reverse()

    min_cost = dp[n][m]
    cost_worst = sum(2.0*w for w in weights) + max(0, m-n)*(1.0*mean_w)
    if cost_worst<=0: cost_worst=1.0
    teds = max(0.0, 1.0 - (min_cost/cost_worst))
    return teds, align, min_cost

def trend_f1_signacc_coverage(G: List[str], P: List[Optional[str]], align: List[Tuple[Optional[int],Optional[int]]]) -> Tuple[float,float,float]:
    P_compact=[p for p in P if p is not None]
    TP=FP=FN=0; matches=0; covered=0
    for gi,pj in align:
        gt_tok = G[gi] if gi is not None else None
        pred_tok = P_compact[pj] if (pj is not None and 0<=pj<len(P_compact)) else None
        if gi is not None:
            if pj is not None:
                covered+=1
                if pred_tok==gt_tok: matches+=1
            gt_event = gt_tok in ("↑","↓")
            pred_event = pred_tok in ("↑","↓")
            if gt_event and pred_event:
                if gt_tok==pred_tok: TP+=1
                else: FP+=1; FN+=1
            elif gt_event and not pred_event:
                FN+=1
            elif (not gt_event) and pred_event:
                FP+=1
        else:
            if pred_tok in ("↑","↓"): FP+=1
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    sign_acc = matches/len(G) if G else 0.0
    coverage = covered/len(G) if G else 0.0
    return f1, sign_acc, coverage

# ------------------------------- Fusion logic ----------------------------------
def updown_to_arrow(x: Optional[str]) -> Optional[str]:
    if x=="UP": return "↑"
    if x=="DOWN": return "↓"
    if x=="STABLE": return "→"
    return None  # UNKNOWN -> None

def fuse_llm_and_rules(pred_llm: List[str], conf_llm: List[float],
                       pred_rb: List[Optional[str]], conf_rb: List[float]) -> List[Optional[str]]:
    out=[]
    for i in range(max(len(pred_llm), len(pred_rb))):
        llm_tok = pred_llm[i] if i<len(pred_llm) else "UNKNOWN"
        llm_arrow = updown_to_arrow(llm_tok)
        rb_arrow  = pred_rb[i] if i<len(pred_rb) else None
        llm_conf  = conf_llm[i] if i<len(conf_llm) else 0.0
        rb_conf   = conf_rb[i] if i<len(conf_rb) else 0.0
        if llm_arrow is not None:
            if llm_conf < 0.55 and rb_arrow is not None and rb_conf >= 0.75:
                out.append(rb_arrow)
            else:
                out.append(llm_arrow)
        else:
            out.append(rb_arrow if rb_arrow is not None else None)
    return out

# ------------------------------ Orchestration ----------------------------------
def compute_case_metrics(case_gt: Dict[str,Any],
                         judge: Dict[str,Any],
                         model_steps: List[str],
                         use_time_weights: bool=True,
                         strict_stable_skip: bool=False) -> Dict[str,Any]:
    model_steps = normalize_steps(model_steps)
    G_arrows = [updown_to_arrow(t) for t in judge["gt_trends"]]

    T = len(case_gt.get("timepoints", []))
    overall_default = None
    for s in model_steps:
        if any(kw in (s or "").lower() for kw in ["overall","over time","longitudinally","in summary"]):
            a,_,_ = detect_arrow_with_conf(s)
            if a: overall_default=a; break
    rb_arrows, rb_conf, rb_mag, chronology_rb = rule_based_intervals(model_steps, T, overall_default=overall_default)

    pred_llm = judge["pred_trends"]
    conf_llm = judge.get("pred_confidence", [0.0]*len(pred_llm))
    pred_fused = fuse_llm_and_rules(pred_llm, conf_llm, rb_arrows, rb_conf)

    chronology = 1.0 if judge.get("chronology_ok", True) else 0.0
    chronology = (chronology + chronology_rb)/2.0

    dates = parse_dates_from_gt(case_gt)
    weights = interval_weights_from_dates(dates) if use_time_weights else [1.0]*max(0,len(G_arrows))

    G = [g if g is not None else "→" for g in G_arrows]
    teds, align, _ = teds_alignment(G, pred_fused, weights, strict_stable_skip=strict_stable_skip)
    f1, sign_acc, coverage = trend_f1_signacc_coverage(G, pred_fused, align)
    tac = 0.5*teds + 0.2*f1 + 0.2*sign_acc + 0.1*coverage

    def sseq(x): return " ".join(x) if x else ""
    def arrowify_pred(pred): return " ".join(a if a is not None else "·" for a in pred)

    return {
        "id": case_gt.get("id"),
        "patient_id": case_gt.get("patient_id", case_gt.get("id","")),
        "num_intervals": len(G),
        "gt_seq": sseq(G),
        "pred_seq": arrowify_pred(pred_fused),
        "TEDS": round(teds,4),
        "TrendF1": round(f1,4),
        "SignAcc": round(sign_acc,4),
        "Coverage": round(coverage,4),
        "Chronology": round(chronology,4),
        "TAC": round(tac,4),
    }

def _load_judge_cache_ids(path: str) -> tuple[dict, set]:
    """
    Return (valid_cache_dict, bad_keys).
    Valid rows must have arrays gt_trends & pred_trends.
    Any row with "error" or missing arrays is marked bad (will be re-judged).
    """
    ok, bad = {}, set()
    if not os.path.exists(path):
        return ok, bad
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            key = str(rec.get("patient_id", rec.get("id", ""))).strip()
            if not key:
                continue
            if rec.get("error"):
                bad.add(key)
                # do NOT store this record as valid cache
                continue
            gt = rec.get("gt_trends")
            pr = rec.get("pred_trends")
            if isinstance(gt, list) and isinstance(pr, list) and gt and pr:
                ok[key] = rec
            else:
                bad.add(key)
    return ok, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Ground truth JSON/JSONL")
    ap.add_argument("--pred", required=True, help="Model outputs JSON/JSONL")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI small model for judging")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_retries", type=int, default=2)

    # Parallel + resume controls
    ap.add_argument("--concurrency", type=int, default=6, help="Parallel judge workers")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip cases already in judge cache (default: on)")
    ap.add_argument("--reuse_judge", action="store_true",
                    help="(Compatibility) Same effect as --resume; kept for older runners")
    ap.add_argument("--judge_path", default=None,
                    help="Optional custom judge cache path (default: out_dir/judge_trends.jsonl)")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing judge cache and re-judge all cases")

    # Metric flags
    ap.add_argument("--no_time_weights", action="store_true")
    ap.add_argument("--strict_stable_skip", action="store_true")
    ap.add_argument("--no_tac", action="store_true")
    ap.add_argument("--single_metric", default="TAC",
                    help="Single score key for JSONL: TAC|TEDS|TrendF1|SignAcc|Coverage|Chronology")
    ap.add_argument("--single_only", action="store_true", help="Only (id, patient_id, score)")
    ap.add_argument("--dry_run", action="store_true", help="No API calls; writes placeholders")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    judge_path = args.judge_path or os.path.join(args.out_dir, "judge_trends.jsonl")
    results_csv = os.path.join(args.out_dir, "results.csv")
    results_jsonl = os.path.join(args.out_dir, "results.jsonl")

    gt_list = load_any_json_or_jsonl(args.gt)
    pred_list = load_any_json_or_jsonl(args.pred)
    pred_idx = index_by_key(pred_list)

    # ---- resume: load already-judged ids (valid only) ----
    if args.resume or args.reuse_judge:
        try:
            cache_dict, bad_keys = _load_judge_cache_ids(judge_path)
        except TypeError:
            # Backward-compat if your _load_judge_cache_ids returns only dict:
            tmp = _load_judge_cache_ids(judge_path)
            cache_dict, bad_keys = (tmp, set())
    else:
        cache_dict, bad_keys = {}, set()

    if args.fresh:
        print(f"[fresh] Ignoring existing cache at {judge_path}")
        cache_dict, bad_keys = {}, set()

    print(f"[resume] found {len(cache_dict)} cached items in {judge_path}")

    # Build plan (skip duplicates by key)
    seen_keys: set = set()
    todo: List[Dict[str, Any]] = []
    judged_lines: List[Dict[str, Any]] = []  # will hold valid cache rows + new rows

    for gt in gt_list:
        key = str(gt.get("patient_id", gt.get("id", ""))).strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        # Re-judge if missing or previously marked bad; otherwise keep cached row
        if (key not in cache_dict) or (key in bad_keys):
            todo.append(gt)
        else:
            judged_lines.append(cache_dict[key])

    print(f"[plan] total={len(gt_list)}, to_judge={len(todo)}, concurrency={args.concurrency}")

    # OpenAI client (only if we actually need to judge)
    client = None
    if todo and not args.dry_run:
        if OpenAI is None:
            raise SystemExit('openai>=1.51.0 required. pip install "openai>=1.51.0"')
        client = OpenAI()

    # ---- parallel judging with immediate append ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    file_lock = threading.Lock()  # protect immediate appends

    def worker(gt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = str(gt.get("patient_id", gt.get("id", ""))).strip()
        # Try multiple identifiers to match predictions
        pred = (pred_idx.get(key)
                or pred_idx.get(str(gt.get("id", "")).strip())
                or {})
        if not pred:
            print(f"⚠️  No prediction found for key='{key}'. Using empty steps (will be penalized).", flush=True)

        steps = extract_steps_from_pred(pred)
        steps = normalize_steps(steps)
        visits, intervals = parse_timepoints(gt.get("timepoints", []))

        # Rule-based hint to assist judge
        T = len(gt.get("timepoints", []))
        rb_pred, rb_conf, rb_mag, chron_rb = rule_based_intervals(steps, T)
        rb_hint = {
            "intervals": intervals,
            "pred_arrows_hint": rb_pred,
            "pred_conf_hint": rb_conf,
            "pred_mag_hint": rb_mag,
            "chronology_hint": chron_rb
        }
        case = dict(gt); case["_model_steps"] = steps

        try:
            data = call_llm_judge(
                client=client, model=args.model, case=case, rb_hint=rb_hint,
                temperature=args.temperature, max_retries=args.max_retries, dry_run=args.dry_run
            )
            out_line = {
                "id": gt.get("id"),
                "patient_id": key,
                "intervals": data["intervals"],
                "gt_trends": data["gt_trends"],
                "pred_trends": data["pred_trends"],
                "pred_magnitude": data["pred_magnitude"],
                "pred_confidence": data["pred_confidence"],
                "coverage": data["coverage"],
                "chronology_ok": data["chronology_ok"]
            }
            with file_lock:
                with open(judge_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                    fout.flush()
            print(f"✅ judged {key}")
            return out_line
        except Exception as e:
            err_line = {"id": gt.get("id"), "patient_id": key, "error": str(e)}
            with file_lock:
                with open(judge_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(err_line, ensure_ascii=False) + "\n")
                    fout.flush()
            print(f"❌ judge failed {key}: {e}")
            return None

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futures = [ex.submit(worker, gt) for gt in todo]
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is not None:
                    judged_lines.append(rec)

    # Rebuild judged index (cache + new), ignore error rows
    judged_idx = {
        str(j["patient_id"]): j
        for j in judged_lines
        if isinstance(j, dict) and "patient_id" in j and "error" not in j
    }

    # ---- scoring ----
    rows: List[Dict[str, Any]] = []
    for gt in gt_list:
        key = str(gt.get("patient_id", gt.get("id", ""))).strip()
        if not key:
            continue

        # Steps for display/fusion (even if empty)
        pred = (pred_idx.get(key)
                or pred_idx.get(str(gt.get("id", "")).strip())
                or {})
        steps = extract_steps_from_pred(pred)

        j = judged_idx.get(key)
        if not j:
            # No valid judge row → skip from scoring (or record a stub if you prefer)
            continue

        row = compute_case_metrics(
            case_gt=gt, judge=j, model_steps=steps,
            use_time_weights=(not args.no_time_weights),
            strict_stable_skip=args.strict_stable_skip
        )
        if args.no_tac and "TAC" in row:
            row.pop("TAC", None)
        rows.append(row)

    # Macro averages
    def mean_of(k: str) -> float:
        vals = [r[k] for r in rows if k in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    macro = {
        "id": "MACRO",
        "patient_id": "-",
        "num_intervals": round(sum(r["num_intervals"] for r in rows) / len(rows), 2) if rows else 0,
        "gt_seq": "-", "pred_seq": "-",
        "TEDS": mean_of("TEDS"),
        "TrendF1": mean_of("TrendF1"),
        "SignAcc": mean_of("SignAcc"),
        "Coverage": mean_of("Coverage"),
        "Chronology": mean_of("Chronology"),
    }
    if not args.no_tac:
        macro["TAC"] = mean_of("TAC")

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else \
        ["id","patient_id","num_intervals","gt_seq","pred_seq","TEDS","TrendF1","SignAcc","Coverage","Chronology","TAC"]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        macro_row = {k: (macro.get(k, "") if k in macro else "") for k in fieldnames}
        w.writerow(macro_row)

    # Write JSONL (with single score)
    valid_single = {"TAC","TEDS","TrendF1","SignAcc","Coverage","Chronology"}
    single_key = args.single_metric
    if single_key not in valid_single:
        raise SystemExit(f"--single_metric must be one of {sorted(valid_single)}")

    with open(results_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            out = {"id": r["id"], "patient_id": r["patient_id"],
                   "score_metric": single_key, "score": r[single_key]}
            if not args.single_only:
                out.update(r)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        m = {"id": "MACRO", "patient_id": "-", "score_metric": single_key,
             "score": macro.get(single_key, 0.0)}
        if not args.single_only:
            m.update(macro)
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote judge cache (appended as we went): {judge_path}")
    print(f"[OK] Wrote results: {results_csv} and {results_jsonl}")
    print("Macro:", {k: macro[k] for k in macro
                    if k in ["TEDS","TrendF1","SignAcc","Coverage","Chronology","TAC"] and k in macro})


if __name__=="__main__":
    main()
