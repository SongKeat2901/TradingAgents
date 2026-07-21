#!/usr/bin/env python3
# TRACKED COPY of the mini-deployed native-GSheet Research Summary digest.
# Deploy target: macmini-trueknot:~/gsheet-tool/update_summary.py (mini-only
# ops, gog + Sheets API — see feedback_google_stuff_only_on_mini). Keep this
# repo copy in sync when the mini script changes. wk29 (2026-07-21): added
# _re_ev3 verbatim-EV fallback ('(+X% from spot') so ECHO's Expected-Value
# line (no colon after 'Value') no longer mis-reads the 60%% Bull probability.
"""Deterministic Research Summary updater (auto-update-on-promote).

Scans promoted reports under final/wk NN YYYY/, picks the LATEST report per
watchlist ticker (manifest = ~/gsheet-tool/pdf_ids.tsv), extracts
rating/price/EV from decision.md, auto-builds a Note (next earnings + bull
gate), and writes the Research Summary gsheet via gog. Idempotent: full-range
overwrite + rating colour bands. Adds a "Last Updated (wk)" column beside the
Report PDF link.

Usage:
  update_summary.py            # extract + write the sheet
  update_summary.py --dry-run  # extract + print rows, no write (validate first)
"""
import json, os, re, sys, glob, subprocess, time, datetime as _dt

DRY = "--dry-run" in sys.argv
GOG = "/opt/homebrew/bin/gog"; ACCT = "shianpin@trueknot.sg"
SID = "1VJowGGdxjCPd0jMpZVHJlC-C6aEspf1iJWOVPH0T7dk"
HOME = os.path.expanduser("~")
MANIFEST = f"{HOME}/gsheet-tool/pdf_ids.tsv"
FINAL = f"{HOME}/tkresearch/final"  # LOCAL canonical published store (mount-independent)
ENV = dict(os.environ)


def gog(*a):
    r = subprocess.run([GOG, *a, "-a", ACCT], env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print("GOG ERR", a[:3], r.stderr[:300]); raise SystemExit(1)
    return r.stdout


# watchlist + Drive file IDs
mf = {}
for l in open(MANIFEST).read().splitlines():
    if "\t" in l:
        k, v = l.split("\t", 1); mf[k.strip()] = v.strip()
WATCH = list(mf)


def latest_reports():
    """ticker -> (date, run_dir, wk_label); latest date wins across all weeks."""
    best = {}
    for wkdir in glob.glob(f"{FINAL}/wk *"):
        wk = os.path.basename(wkdir)
        for d in glob.glob(f"{wkdir}/2026-*"):
            m = re.match(r"(\d{4}-\d{2}-\d{2})-([A-Z]{1,6})$", os.path.basename(d))
            if not m or not os.path.isdir(d):
                continue
            date, tkr = m.group(1), m.group(2)
            if tkr in best and best[tkr][0] >= date:
                continue
            best[tkr] = (date, d, wk)
    return best


_re_price = re.compile(r"\*\*Reference price:\*\*\s*\$?([\d,]+\.?\d*)")
# Tolerant to format drift: `**Rating implication:** **HOLD.**` AND
# `**Rating implication: UNDERWEIGHT.**` (rating inside the bold).
_re_rating = re.compile(r"Rating implication:[\s*]*([A-Za-z]+)")
# EV %: three observed phrasings, tried in order. Unicode minus (U+2212) and
# bold can wrap the number/percent.
#   1. `(**−6.93%** from spot $169.27)` / `(+4.89% from spot $303.11)`  (most)
#   2. `EV vs spot $7.03: (...) = −2.15%`                               (SOUN)
#   3. dollar only `Expected Value ... = **$101.0672**` -> compute from price (NOW)
#   4. percent-based `= **+1.82% (≈ $271.62 from spot $266.77)**`            (MRVL)
_re_ev = re.compile(
    r"Expected Value:.*?\(\s*\*{0,2}([+\-−]?\d+\.?\d*)\s*%\*{0,2}\s*from\s+spot", re.S)
_re_ev2 = re.compile(r"EV vs spot.*?=\s*\*{0,2}([+\-−]?\d+\.?\d*)\s*%", re.S)
# 5. verbatim EV parenthetical: `(+1.10% from spot $92.00)` (ECHO — Expected Value
#    line lacks the colon _re_ev needs; match the highly-specific "% from spot").
_re_ev3 = re.compile(r"\(\s*\*{0,2}([+\-−]?\d+\.?\d*)\s*%\*{0,2}\s+from\s+spot", re.S)
_re_name = re.compile(r"^Name:\s*(.+)$", re.M)


def _ev_pct(dec, price):
    m = _re_ev.search(dec) or _re_ev2.search(dec) or _re_ev3.search(dec)
    if m:
        return float(m.group(1).replace("−", "-"))
    seg = dec[dec.find("Expected Value"):][:600] if "Expected Value" in dec else ""
    mp = re.search(r"\*\*\s*([+\-−]?\d+\.?\d*)\s*%", seg)  # bolded EV percent (MRVL)
    if mp:
        return float(mp.group(1).replace("−", "-"))
    md = re.search(r"\*\*\$([\d,]+\.?\d*)", seg)  # bolded EV dollar -> % from price
    if md and price:
        return round((float(md.group(1).replace(",", "")) - price) / price * 100, 2)
    return None


def extract(run_dir):
    dec = open(os.path.join(run_dir, "decision.md")).read()
    pm, rm = _re_price.search(dec), _re_rating.search(dec)
    price = float(pm.group(1).replace(",", "")) if pm else None
    rating = rm.group(1).capitalize() if rm else None
    ev = _ev_pct(dec, price)
    return price, rating, ev, dec


def note_for(run_dir, dec, tkr):
    nxt = ""
    cj = os.path.join(run_dir, "raw", "calendar.json")
    if os.path.isfile(cj):
        try:
            nxt = json.load(open(cj)).get(tkr, {}).get("next_expected", "") or ""
        except Exception:
            pass
    gate = ""
    for line in dec.splitlines():
        if line.strip().startswith("|") and re.search(r"\bBull\b", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells:
                gate = re.sub(r"\s+", " ", cells[-1])[:150]
            break
    parts = []
    if nxt:
        parts.append(f"Next earnings ~{nxt}")
    if gate:
        parts.append(f"bull gate: {gate}")
    return "; ".join(parts) or "—"


def company(run_dir, tkr):
    fj = os.path.join(run_dir, "raw", "financials.json")
    if os.path.isfile(fj):
        try:
            m = _re_name.search(json.load(open(fj)).get("fundamentals", ""))
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    try:
        import yfinance as yf
        i = yf.Ticker(tkr).info
        return i.get("longName") or i.get("shortName") or tkr
    except Exception:
        return tkr


def close(t):
    import yfinance as yf
    for _ in range(3):
        try:
            h = yf.Ticker(t).history(period="7d")
            if len(h):
                return round(float(h["Close"].iloc[-1]), 2)
        except Exception:
            pass
        time.sleep(2)
    return None


REPORTS = latest_reports()
HEADER = ["Report Date", "Ticker", "Company", "Rating", "Price @ Report ($)",
          "EV 12-Mo ($)", "EV 12-Mo (%)", "Current Close ($)", "Move Since (%)",
          "Notes", "Report PDF", "Last Updated (wk)"]
order = {"Overweight": 0, "Hold": 1, "Underweight": 2, "Sell": 3}

rows = []
for tkr in WATCH:
    if tkr not in REPORTS:
        print(f"  [skip] {tkr}: no promoted report found"); continue
    date, run_dir, wk = REPORTS[tkr]
    price, rating, ev, dec = extract(run_dir)
    if not (price and rating and ev is not None):
        print(f"  [warn] {tkr}: incomplete extraction price={price} rating={rating} ev={ev}")
    note = note_for(run_dir, dec, tkr)
    comp = company(run_dir, tkr)
    c = None if DRY else close(tkr)
    mv = round((c - price) / price * 100, 2) if (c and price) else ""
    evd = round(price * (1 + ev / 100), 2) if (price and ev is not None) else ""
    fid = mf.get(tkr)
    link = f'=HYPERLINK("https://drive.google.com/file/d/{fid}/view","PDF")' if fid else ""
    rows.append([date, tkr, comp, rating or "?", price or "", evd,
                 ev if ev is not None else "", (c if c else ""), mv, note, link, wk])

rows.sort(key=lambda r: (order.get(r[3], 9), r[1]))
ratings = [r[3] for r in rows]

if DRY:
    print(f"\n=== DRY RUN: {len(rows)} rows (no write) ===")
    for r in rows:
        print(f"  {r[1]:6} {r[3]:11} px={r[4]} ev%={r[6]} wk={r[11]} | {r[2][:34]:34} | {r[9][:60]}")
    print("\nratings:", {x: ratings.count(x) for x in set(ratings)})
    raise SystemExit(0)

_ts = ("TrueKnot Research Summary   .   Last updated: "
       + _dt.datetime.now().strftime("%Y-%m-%d %H:%M SGT"))
NC = len(HEADER); colL = chr(64 + NC)
out = [[_ts] + [""] * (NC - 1)] + [HEADER] + rows
gog("sheets", "update", SID, f"A1:{colL}40", "--input", "USER_ENTERED",
    "--values-json", json.dumps([[""] * NC for _ in range(40)]))
gog("sheets", "update", SID, "A1", "--input", "USER_ENTERED",
    "--values-json", json.dumps(out))
COL = {"Overweight": {"red": 0.78, "green": 0.91, "blue": 0.79},
       "Hold": {"red": 1, "green": 0.95, "blue": 0.70},
       "Underweight": {"red": 0.96, "green": 0.80, "blue": 0.80},
       "Sell": {"red": 0.86, "green": 0.46, "blue": 0.42}}
for i, rt in enumerate(ratings):
    if rt in COL:
        gog("sheets", "format", SID, f"Sheet1!A{i+3}:{colL}{i+3}", "--format-json",
            json.dumps({"backgroundColor": COL[rt]}), "--format-fields",
            "userEnteredFormat.backgroundColor")
print(f"WROTE {len(rows)} rows. ratings:", {x: ratings.count(x) for x in set(ratings)})
