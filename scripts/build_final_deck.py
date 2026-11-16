"""Build the final presentation by extending proposal deck v10 (project_docs/), keeping its theme and layouts.

Every number on the new slides is read from outputs/results/*.json and outputs/reports/*.json, and the RQ1 wording
comes from src/niftygex_eval/rq1_summary.py, so the deck states results exactly as the reports do. Icons are reused
from deck v10 itself. Needs python-pptx (env/environment.yml).

    PYTHONPATH=src python scripts/build_final_deck.py [--out docs/deliverables/NiftyGEX_Final_Presentation.pptx]
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from niftygex.config import REPO_ROOT
from niftygex_eval.rq1_summary import ladder_improvements, qlike_explained, rq1_summary

_args = argparse.ArgumentParser(description="Build the final presentation from deck v10 and the results files.")
_args.add_argument("--out", default=str(REPO_ROOT / "docs" / "deliverables" / "NiftyGEX_Final_Presentation.pptx"))
REPO = REPO_ROOT
OUT = Path(_args.parse_args().out)
SRC = REPO / "project_docs" / "Long_Attention_Short_Gamma_CS7150_Proposal_v10.pptx"
RES = REPO / "outputs" / "results"
REP = REPO / "outputs" / "reports"
FIG = REP / "figures"

THEME = {"tx1": MSO_THEME_COLOR.TEXT_1, "tx2": MSO_THEME_COLOR.TEXT_2, "bg1": MSO_THEME_COLOR.BACKGROUND_1,
         "bg2": MSO_THEME_COLOR.BACKGROUND_2, **{f"accent{i}": getattr(MSO_THEME_COLOR, f"ACCENT_{i}") for i in range(1, 7)}}
PAPER = RGBColor(252, 252, 251)  # background of the matplotlib figures
CLASSICAL, NEURAL, OURS, EXTRA = "accent5", "accent1", "accent3", "accent2"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


results = load(REP / "results.json")
ladder = {r["model"]: r for r in results["ladder"]}
rq1 = {r["comparison"].split(" vs ")[1]: r for r in results["rq1"]}
rq2 = results["rq2"]
rq3 = results["rq3"]
attention = load(REP / "attention.json")["models"]
e1 = load(RES / "exploratory_e1.json")
e1_tests = {m["test"]: m for m in e1["tests"]["members"]}
e2 = load(RES / "exploratory_e2.json")
e2_tests = {m["test"]: m for m in e2["tests"]["members"]}
e3 = {h: load(RES / f"exploratory_e3_{h}.json") for h in ("1h", "2h")}
e3_tests = {m["test"]: m for m in load(RES / "exploratory_e3.json")["tests"]["members"]}
summary = load(REP / "exploratory_summary.json")
drop = load(RES / "neural_comparison_drop_oi_change_since_previous_close_pct.json")["comparison_table"][0]

NAMES = {"persistence": "Persistence", "har": "HAR-RV", "har_gex": "HAR + GEX", "lightgbm": "Tuned LightGBM",
         "deepsets": "DeepSets", "attention": "Self-attention", "attention_gru": "Attention + GRU", "intraday_har": "Intraday HAR"}


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else (f"{p:.2f}" if p >= 0.1 else f"{p:.2g}")


# ---------------------------------------------------------------- drawing helpers
def color(fmt, key: str) -> None:
    fmt.theme_color = THEME[key]


def rect(slide, x, y, w, h, fill="bg2", name=None, shape=MSO_SHAPE.RECTANGLE, rgb=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    if rgb is not None:
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb
    elif fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); color(sh.fill.fore_color, fill)
    sh.line.fill.background()
    if name:
        sh.name = name
    if sh.has_text_frame:
        sh.text_frame.text = ""
    return sh


def text(slide, x, y, w, h, paragraphs, size=14, col="tx1", bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         name=None, space_after=0, line=None):
    """paragraphs: str | list of paragraphs; a paragraph is a str or a list of (text, {size,bold,col,italic}) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after:
            p.space_after = Pt(space_after)
        if line:
            p.line_spacing = line
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            color(r.font.color, o.get("col", col))
    return box


def icon(slide, x, y, d, fill, png, name):
    rect(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL, name=f"{name} circle")
    s = d * 0.48
    pic = slide.shapes.add_picture(io.BytesIO(ICON_BLOBS[png]), Inches(x + (d - s) / 2), Inches(y + (d - s) / 2), Inches(s), Inches(s))
    pic.name = f"{name} icon"


def figure(slide, x, y, w, path: Path, name, pad=0.12):
    from PIL import Image
    iw, ih = Image.open(path).size
    h = w * ih / iw
    rect(slide, x, y, w + 2 * pad, h + 2 * pad, rgb=PAPER, name=f"{name} card")
    pic = slide.shapes.add_picture(str(path), Inches(x + pad), Inches(y + pad), Inches(w), Inches(h))
    pic.name = name
    return y + h + 2 * pad


def style_chart(chart, size=12):
    chart.font.size = Pt(size)
    chart.font.name = "Calibri"
    color(chart.font.color, "tx2")


def bar_chart(slide, x, y, w, h, categories, series, colors, horizontal=False, fmt="0.000", vmin=None, vmax=None,
              legend=False, name="chart", label_size=12, gap=60):
    data = CategoryChartData()
    data.categories = categories
    for label, values in series:
        data.add_series(label, values)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gframe = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    gframe.name = name
    chart = gframe.chart
    style_chart(chart)
    chart.has_title = False
    plot = chart.plots[0]
    plot.gap_width = gap
    plot.overlap = -10 if len(series) > 1 else 0
    plot.vary_by_categories = False
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(label_size)
    dl.font.bold = True
    color(dl.font.color, "tx1")
    for s_i, s in enumerate(plot.series):
        s.invert_if_negative = False
        cols = colors[s_i]
        if not isinstance(cols, list):
            s.format.fill.solid(); color(s.format.fill.fore_color, cols)
        for p_i in range(len(categories)):
            key = cols[p_i] if isinstance(cols, list) else cols
            pt = s.points[p_i]
            pt.format.fill.solid(); color(pt.format.fill.fore_color, key)
            pt.format.line.fill.background()
    from lxml import etree
    C = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
    for dpt in chart._chartSpace.iter(C + "dPt"):
        inv = dpt.find(C + "invertIfNegative")
        if inv is None:
            inv = etree.SubElement(dpt, C + "invertIfNegative")
            dpt.remove(inv); dpt.insert(1, inv)
        inv.set("val", "0")
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.width = Pt(0.75)
    color(va.major_gridlines.format.line.color, "accent4")
    va.format.line.fill.background()
    va.tick_labels.number_format = "0.0" if fmt == "0.000" else fmt
    va.tick_labels.number_format_is_linked = False
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.tick_labels.font.size = Pt(11)
    ca.tick_labels.font.size = Pt(12)
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ca.format.line.width = Pt(0.75)
    color(ca.format.line.color, "accent4")
    ca.has_major_gridlines = False
    if horizontal:
        ca.reverse_order = True
        va.has_major_gridlines = True
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
        color(chart.legend.font.color, "tx1")
    return chart


# ---------------------------------------------------------------- slide frame
prs = Presentation(str(SRC))
# Icons of deck v10, named s<slide>_<shape name>.png (e.g. s15_win_icon.png).
from pptx.enum.shapes import MSO_SHAPE_TYPE  # noqa: E402
ICON_BLOBS = {f"s{i}_{sh.name.replace(' ', '_')}.{sh.image.ext}": sh.image.blob for i, sl in enumerate(prs.slides, 1)
              for sh in sl.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and "icon" in sh.name}
LAYOUT = next(l for l in prs.slide_layouts if l.name == "CONTENT_2")
SLIDE_NUMBER = next(sh for sh in prs.slides[14].shapes if sh.is_placeholder and sh.placeholder_format.type == PP_PLACEHOLDER.SLIDE_NUMBER)._element


def title_shape(slide):
    return next((sh for sh in slide.placeholders if sh.placeholder_format.type == PP_PLACEHOLDER.TITLE), None)


def footer(slide):
    text(slide, 0.6, 7.0, 8.0, 0.3, [[("What it is      Why      Approach      ", {"col": "tx2"}), ("Results", {"col": "accent3", "bold": True})]],
         size=11, anchor=MSO_ANCHOR.MIDDLE, name="section footer")


def new_slide(title, notes, subtitle=None, hidden=False):
    slide = prs.slides.add_slide(LAYOUT)
    for ph in list(slide.placeholders):
        if ph.placeholder_format.type != PP_PLACEHOLDER.TITLE:
            ph._element.getparent().remove(ph._element)
    tf = title_shape(slide).text_frame
    tf.text = ""
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(36); r.font.bold = True; color(r.font.color, "tx1")
    if subtitle:
        text(slide, 0.6, 1.35, 12.13, 0.4, subtitle, size=14.5, col="tx2", name="subtitle")
    footer(slide)
    slide.shapes._spTree.append(copy.deepcopy(SLIDE_NUMBER))
    ns = slide.notes_slide
    if ns.notes_placeholder is None:
        body = next(sh for sh in prs.slides[14].notes_slide.placeholders if sh.placeholder_format.type == PP_PLACEHOLDER.BODY)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_placeholder.text_frame.text = notes
    if hidden:
        slide._element.set("show", "0")
    return slide


# ---------------------------------------------------------------- existing slides: wording for the final talk
for slide in prs.slides:
    for sh in slide.shapes:
        if sh.name == "section footer":
            for r in sh.text_frame.paragraphs[0].runs:
                if r.text.startswith("Deliverables"):
                    r.text = "Results"
for sh in prs.slides[0].shapes:
    if sh.name == "Text 42":
        runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
        runs[0].text, runs[1].text = "December 2026", "Final presentation"
retitle = {"What we'll predict, and how we'll know it works": "What we predict, and how we know it works",
           "Three questions we'll answer": "The three questions we tested"}
for slide in prs.slides:
    title = title_shape(slide)
    if title is not None and title.text_frame.text in retitle:
        title.text_frame.paragraphs[0].runs[0].text = retitle[title.text_frame.text]
old_slides = list(prs.slides)

# ---------------------------------------------------------------- R0 what the scores mean
rungs = ["persistence", "har", "har_gex", "lightgbm", "deepsets", "attention", "attention_gru"]
ic = {m: ladder[m]["ic_mean_per_walk"] for m in rungs}
walks = ladder["attention_gru"]["walks"]
spread = max(ic[m] for m in rungs[1:]) - min(ic[m] for m in rungs[1:])
tests_rq = load(RES / "rq_tests.json")
q1 = rq1_summary(tests_rq["RQ1"]["comparisons"], ladder, tests_rq["alpha"], tests_rq["RQ1"]["beats_every_rung"])
verdict = {r["rung"]: r for r in q1["rungs"]}
qx = qlike_explained()
import numpy as _np  # noqa: E402
from niftygex_eval.forecast.metrics import qlike as _qlike  # noqa: E402
q_low = float(_qlike(_np.array([0.0]), _np.array([_np.log(0.8)]))[0])
q_high = float(_qlike(_np.array([0.0]), _np.array([_np.log(1.2)]))[0])
s = new_slide("What the scores mean",
              "Three scores, all on the same test periods. RMSE is the typical size of a forecast's miss. QLIKE is the standard error "
              "measure for volatility forecasts: it compares forecast and realized variance as a ratio, so a twenty percent miss counts "
              f"the same on calm and wild days, and it punishes under-predicting more: twenty percent too low costs {q_low:.3f}, twenty "
              f"percent too high costs {q_high:.3f}. That is the costlier mistake for anyone managing risk, and QLIKE is one of the few "
              "losses that still ranks forecasts correctly when volatility can only be measured with noise. IC is ranking accuracy: does "
              "the forecast put calm and wild days in the right order. Lower is better for RMSE and QLIKE, higher for IC.",
              subtitle="Two error scores (lower is better) and one ranking score (higher is better), on the same 26 test periods")
score_rows = [("RMSE · forecast error", "Typical size of a forecast's miss. Lower is better.", "s15_metric_1_icon.png", "accent1"),
              ("QLIKE · forecast error, the standard for volatility",
               "Compares forecast and realized variance as a ratio, so a 20% miss counts the same on calm and wild days. Lower is better.",
               "s15_metric_2_icon.png", "accent2"),
              ("IC · ranking accuracy", "Does the forecast put calm and wild days in the right order? 1 = perfect, 0 = no skill. Higher is better.",
               "s15_metric_0_icon.png", "accent5")]
for i, (head, body, png, fill) in enumerate(score_rows):
    y = 1.95 + i * 1.3
    icon(s, 0.6, y + 0.05, 0.85, fill, png, f"score {i}")
    text(s, 1.7, y, 6.9, 1.1, [[(head, {"bold": True, "size": 19})], [(body, {"size": 14, "col": "tx2"})]], anchor=MSO_ANCHOR.MIDDLE, name=f"score {i} text")
rect(s, 9.0, 1.85, 3.73, 4.9, name="qlike panel")
text(s, 9.2, 2.05, 3.33, 4.6, [[("Why QLIKE?", {"bold": True, "size": 18, "col": "accent3"})],
                               [("Missing a wild day costs more", {"bold": True, "size": 14})],
                               [(f"Forecast 20% too low: {q_low:.3f}", {"size": 13})], [(f"Forecast 20% too high: {q_high:.3f}", {"size": 13})],
                               [(f"Under-predicting costs about {q_low / q_high:.0f}× as much.", {"size": 13, "col": "tx2"})],
                               [("Fair on calm and wild days", {"bold": True, "size": 14})],
                               [("Misses are measured as ratios.", {"size": 13, "col": "tx2"})],
                               [("The field standard", {"bold": True, "size": 14})],
                               [("Ranks forecasts correctly even though true volatility is only measured with noise (Patton, 2011).",
                                 {"size": 13, "col": "tx2"})]], space_after=4, name="qlike text")

# ---------------------------------------------------------------- R1 scoreboard
best_rmse_pct, best_qlike_pct = verdict["lightgbm"]["rmse_lower_pct"], verdict["har_gex"]["qlike_lower_pct"]
s = new_slide("The scoreboard: our model has the lowest error",
              f"Forecast error on the {walks} unseen test periods, lower is better. Attention plus GRU has the lowest RMSE, "
              f"{ladder['attention_gru']['rmse_mean_per_walk']:.3f}, and the lowest QLIKE, {ladder['attention_gru']['qlike_mean_per_walk']:.3f}, of every model. "
              f"Its QLIKE is {verdict['har_gex']['qlike_lower_pct']:.0f} percent below HAR plus GEX and {verdict['lightgbm']['qlike_lower_pct']:.0f} percent below "
              f"tuned LightGBM, both significant. On ranking, IC, it is on par with the best classical models: {ic['attention_gru']:.3f} against "
              f"{ic['lightgbm']:.3f} for LightGBM and {ic['har_gex']:.3f} for HAR plus GEX.",
              subtitle=f"Forecast error of every model on the same {walks} unseen test periods (lower is better)")
colors_rungs = [[CLASSICAL] * 4 + [NEURAL] * 2 + [OURS]]
short_names = ["Persistence", "HAR-RV", "HAR + GEX", "LightGBM", "DeepSets", "Self-attn", "Attn + GRU"]
text(s, 0.6, 1.85, 4.2, 0.3, "RMSE", size=15, bold=True, name="rmse head")
bar_chart(s, 0.45, 2.1, 4.35, 4.7, short_names, [("RMSE", [ladder[m]["rmse_mean_per_walk"] for m in rungs])], colors_rungs,
          horizontal=True, vmin=0, vmax=0.45, name="rmse chart", gap=35, label_size=11)
text(s, 4.95, 1.85, 4.2, 0.3, "QLIKE", size=15, bold=True, name="qlike head")
bar_chart(s, 4.8, 2.1, 4.35, 4.7, short_names, [("QLIKE", [ladder[m]["qlike_mean_per_walk"] for m in rungs])], colors_rungs,
          horizontal=True, vmin=0, vmax=0.45, name="qlike chart", gap=35, label_size=11)
rect(s, 9.42, 1.85, 3.31, 4.95, name="callout panel")
text(s, 9.62, 2.0, 2.91, 4.7, [[("Our model", {"size": 14, "col": "tx2"})],
                               [(f"{ladder['attention_gru']['rmse_mean_per_walk']:.3f}", {"bold": True, "size": 34, "col": "accent3"}), ("  RMSE", {"size": 13})],
                               [(f"{ladder['attention_gru']['qlike_mean_per_walk']:.3f}", {"bold": True, "size": 34, "col": "accent3"}), ("  QLIKE", {"size": 13})],
                               [("lowest of every model", {"bold": True, "size": 13})],
                               [(f"QLIKE {verdict['har_gex']['qlike_lower_pct']:.0f}% below HAR + GEX and {verdict['lightgbm']['qlike_lower_pct']:.0f}% below "
                                 f"LightGBM (both significant)", {"size": 12.5, "col": "tx2"})],
                               [("Ranking (IC)", {"bold": True, "size": 13})],
                               [(f"{ic['attention_gru']:.3f}, on par with LightGBM {ic['lightgbm']:.3f} and HAR + GEX {ic['har_gex']:.3f}",
                                 {"size": 12.5, "col": "tx2"})]], space_after=5, name="callout text")

# ---------------------------------------------------------------- R2 Q1
s = new_slide("Q1: Does reading every contract predict better?",
              f"{q1['headline']} {q1['loss']} {q1['gex'] or ''} {q1['rule']}",
              subtitle="Attention + GRU against each competitor on the same 26 test periods: error (lower is better) and ranking")
cols = [(0.8, 2.6, "Against"), (3.5, 2.6, "RMSE"), (6.2, 2.6, "QLIKE"), (8.9, 3.7, "Ranking (IC gain, 95% interval)")]
for x, w, head in cols:
    text(s, x, 1.9, w, 0.3, head, size=13, bold=True, col="tx2", name=f"head {head}")
for i, m in enumerate(["persistence", "har", "har_gex", "lightgbm"]):
    v = verdict[m]
    y = 2.3 + i * 0.86
    rect(s, 0.6, y - 0.05, 12.13, 0.72, name=f"row {i} panel")
    text(s, 0.8, y, 2.6, 0.62, [[(NAMES[m], {"bold": True, "size": 16})], [(f"won {v['wins']} of {v['walks']} periods on IC", {"size": 11, "col": "tx2"})]],
         anchor=MSO_ANCHOR.MIDDLE, name=f"row {i} name")
    for (x, w, _), pct_v, status, p in ((cols[1], v["rmse_lower_pct"], v["rmse_status"], v["dm_squared_p"]),
                                        (cols[2], v["qlike_lower_pct"], v["qlike_status"], v["dm_qlike_p"])):
        sig = status == "lower"
        text(s, x, y, w, 0.62, [[(f"{abs(pct_v):.1f}% {'lower' if pct_v >= 0 else 'higher'}", {"bold": True, "size": 16, "col": "accent1" if sig else "tx1"})],
                                [(("significant" if sig else "not significant") + f" · p {p_text(p)}", {"size": 11, "col": "accent1" if sig else "tx2"})]],
             anchor=MSO_ANCHOR.MIDDLE, name=f"row {i} {x}")
    ahead = v["ic_status"] == "ahead"
    text(s, cols[3][0], y, cols[3][1], 0.62, [[(f"{v['ic_diff']:+.3f}", {"bold": True, "size": 16, "col": "accent1" if ahead else "tx1"}),
                                              (f"   {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f}", {"size": 12, "col": "tx2"})],
                                             [(v["ic_status"], {"size": 11, "col": "accent1" if ahead else "tx2"})]],
         anchor=MSO_ANCHOR.MIDDLE, name=f"row {i} ic")
rect(s, 0.6, 5.85, 12.13, 0.95, name="answer panel")
icon(s, 0.8, 5.97, 0.7, "accent3", "s15_win_icon.png", "answer")
gx = verdict["har_gex"]
text(s, 1.7, 5.85, 10.85, 0.95, [[("Answer: it does well. ", {"bold": True}),
                                   (f"Lowest error of every model, and it beats GEX on QLIKE ({gx['qlike_lower_pct']:.0f}% lower). On ranking it matches the best "
                                    "classical models but not by the clear margin we hoped for, so the strict pre-registered rule "
                                    f"(a significant ranking gain over every model) is {'met' if q1['beats_every_rung'] else 'not met'}.", {})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE, name="answer text")

# ---------------------------------------------------------------- R3 Q2
interactions, history = rq2
s = new_slide("Q2: Switch a part off, what is lost?",
              "The switch-off tests from the proposal, Holm-corrected as a family of two. Letting contracts compare notes, "
              "self-attention against DeepSets, does not help. Adding memory of the last hour, the GRU, helps a little but "
              "not significantly. We also dropped the OI-change field that the attention maps pointed to; that single run "
              "is exploratory. Since the model does not beat the classical rungs, there is little gain to explain.",
              subtitle="Each test compares the model with and without one part, on the same 26 test periods (Holm-corrected)")
cards = [("Contracts compare notes", "self-attention vs DeepSets", interactions, "s7_attention_icon.png", "accent1"),
         ("Memory of the last hour", "attention + GRU vs one snapshot", history, "s7_gru_icon.png", "accent2")]
for i, (head, sub, r, png, fill) in enumerate(cards):
    x = 0.6 + i * 4.11
    rect(s, x, 1.95, 3.91, 3.75, name=f"card {i}")
    icon(s, x + 1.47, 2.15, 1.0, fill, png, f"card {i}")
    text(s, x + 0.2, 3.3, 3.53, 0.65, [[(head, {"bold": True, "size": 18})], [(sub, {"col": "tx2", "size": 12})]], align=PP_ALIGN.CENTER, name=f"card {i} head")
    text(s, x + 0.2, 4.05, 3.53, 0.8, f"{r['ic_diff']:+.3f}", size=44, bold=True, col="accent3", align=PP_ALIGN.CENTER, name=f"card {i} stat")
    text(s, x + 0.2, 4.95, 3.53, 0.6, [[("IC change when switched on", {"size": 12})],
                                       [(f"Holm p {p_text(r['p_holm'])} · {'significant' if r['reject'] else 'not significant'}", {"col": "tx2", "size": 12})]],
         align=PP_ALIGN.CENTER, name=f"card {i} note")
x = 0.6 + 2 * 4.11
base_ic = ladder["attention_gru"]["ic_mean_per_walk"]
rect(s, x, 1.95, 3.91, 3.75, name="card 2")
icon(s, x + 1.47, 2.15, 1.0, "accent5", "s13_note_1_icon.png", "card 2")
text(s, x + 0.2, 3.3, 3.53, 0.65, [[("Drop the OI-change field", {"bold": True, "size": 18})], [("exploratory, one run", {"col": "tx2", "size": 12})]],
     align=PP_ALIGN.CENTER, name="card 2 head")
text(s, x + 0.2, 4.05, 3.53, 0.8, f"{drop['ic_mean_per_walk']:.3f}", size=44, bold=True, col="tx1", align=PP_ALIGN.CENTER, name="card 2 stat")
text(s, x + 0.2, 4.95, 3.53, 0.6, [[("IC without the field", {"size": 12})], [(f"vs {base_ic:.3f} with it; not tested", {"col": "tx2", "size": 12})]],
     align=PP_ALIGN.CENTER, name="card 2 note")
rect(s, 0.6, 5.95, 12.13, 0.8, name="answer panel")
icon(s, 0.8, 6.02, 0.65, "accent3", "s9_rq_1_icon.png", "answer")
n_sig = sum(r["reject"] for r in rq2)
text(s, 1.65, 5.95, 10.9, 0.8, [[("Answer: ", {"bold": True}),
                                  (f"{n_sig} of {len(rq2)} switch-off tests {'is' if n_sig == 1 else 'are'} significant. No single part of the model makes a detectable difference.", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE, name="answer text")

# ---------------------------------------------------------------- R4 Q3
s = new_slide("Q3: Does it survive the rule change?",
              f"The November 2024 weekly-expiry rationalisation cut the number of weekly expiries. We compare the model's per-period "
              f"gain over tuned LightGBM before and after the change. Before: {rq3['mean_gain_before']:+.3f} over {rq3['walks_before']} "
              f"periods; after: {rq3['mean_gain_after']:+.3f} over {rq3['walks_after']}. Welch p {rq3['p_two_sided']:.2f}: no detectable "
              f"change. With only {rq3['walks_after']} periods after the change, only a large shift would show up.",
              subtitle="Mean IC gain of attention + GRU over tuned LightGBM, per test period, before and after Nov 2024")
bar_chart(s, 0.45, 1.85, 7.4, 4.95, [f"Before (n = {rq3['walks_before']})", f"After (n = {rq3['walks_after']})"],
          [("gain", [rq3["mean_gain_before"], rq3["mean_gain_after"]])], [["accent5", "accent2"]], fmt="+0.000;-0.000",
          vmin=-0.03, vmax=0.03, name="regime chart", gap=120, label_size=16)
rect(s, 8.3, 1.85, 4.43, 4.95, name="regime panel")
icon(s, 8.5, 2.05, 0.9, "accent2", "s9_rq_2_icon.png", "regime")
text(s, 8.5, 3.15, 4.03, 3.5, [[("Answer: no detectable change", {"bold": True, "size": 19})],
                               [("Welch t-test p ", {}), (f"{rq3['p_two_sided']:.2f}", {"bold": True, "col": "accent3"})],
                               [(f"Only {rq3['walks_after']} test periods fall after the rule change, so this test can only catch a large shift.", {"col": "tx2"})],
                               [("Lot-size changes and volatility regimes are exploratory splits (report, section on regimes).", {"col": "tx2", "size": 12})]],
     size=14.5, space_after=10, name="regime text")

# ---------------------------------------------------------------- R5 attention maps
ag = attention["attention_gru"]
s = new_slide("Where the model looks: attention maps",
              f"The pooling weights tell us which contracts the model relied on. On almost every test day they are a plain average: "
              f"the median effective share is {ag['effective_share']['median']:.3f}, where 1 means every contract weighted equally. "
              f"Only {ag['concentrated_days_any_head']} of {ag['prediction_snapshots']:,} days are concentrated, and on "
              f"{ag['concentrated_top_is_largest_oi_change']} of them the top contract is the day's largest change in open interest, "
              "usually a newly opened strike. This matches Q2: the attention step adds little beyond averaging.",
              subtitle="Largest weight the model gives any single contract, each test day at 15:15 (1 = plain average)")
figure(s, 0.6, 1.9, 8.1, FIG / "attention_concentration.png", "attention figure")
stats = [(f"{ag['effective_share']['median']:.3f}", "median effective share", "1 = every contract weighted equally", "s7_attention_icon.png", "accent1"),
         (f"{ag['concentrated_days_any_head']} of {ag['prediction_snapshots']:,}", "test days where attention concentrates", "effective share below 0.9", "s10_stat_0_icon.png", "accent2"),
         (f"{ag['concentrated_top_is_largest_oi_change']} of {ag['concentrated_days_any_head']}", "of those: top contract = largest OI change", "usually a newly opened strike", "s12_step_4_icon.png", "accent3")]
for i, (big, head, sub, png, fill) in enumerate(stats):
    y = 1.9 + i * 1.12
    icon(s, 9.25, y + 0.1, 0.7, fill, png, f"stat {i}")
    text(s, 10.15, y, 2.6, 1.0, [[(big, {"bold": True, "size": 24})], [(head, {"size": 12.5})], [(sub, {"size": 11, "col": "tx2"})]], name=f"stat {i} text")
rect(s, 0.6, 5.95, 12.13, 0.8, name="reading panel")
icon(s, 0.8, 6.02, 0.65, "accent1", "s7_attention_icon.png", "reading")
text(s, 1.65, 5.95, 10.9, 0.8, [[("Reading: ", {"bold": True}), ("the model spreads its attention evenly, so it behaves like a sum over contracts, much like GEX itself.", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE, name="reading text")

# ---------------------------------------------------------------- R6 exploratory overview
rows = summary["rows"]
best_plus = max((e1_tests[n] for n in ("har_plus_residual_lightgbm_vs_har", "har_plus_residual_deepsets_vs_har", "har_plus_residual_attention_vs_har")),
                key=lambda m: m["mean_diff"])
best_e2 = max(e2_tests.values(), key=lambda m: m["mean_diff"])
e2_rejects = sum(m["reject_holm"] for m in e2_tests.values())
s = new_slide("Beyond the plan: three follow-up questions",
              "After the pre-registered tests we asked three follow-up questions, each registered in its own configuration file before "
              "any run, with Holm correction inside each experiment. They are exploratory: they explain the main result but do not change it. "
              "E1: used as a correction to HAR, the chain does help. E2: more history does not help. E3: for the next one to two hours, "
              "the attention model is significantly worse than simple volatility models.",
              subtitle="Exploratory: registered before running, separate from Q1–Q3, Holm-corrected within each experiment")
e3_1, e3_2 = (e3_tests[f"attention_gru_vs_best_classical_{h}"] for h in ("1h", "2h"))
cols = [("E1", "Use the chain to correct HAR", rows[0]["verdict"].split(":")[0], "accent1",
         [[("HAR + correction ", {}), (f"{best_plus['mean_diff']:+.3f}", {"bold": True}), (f" IC (Holm p {p_text(best_plus['p_holm'])})", {})],
          [("Average of 3 forecasts ", {}), (f"{e1_tests['combination_vs_har']['mean_diff']:+.3f}", {"bold": True}),
           (f" (Holm p {p_text(e1_tests['combination_vs_har']['p_holm'])})", {})]], "s12_step_2_icon.png"),
        ("E2", "Give the model more history", rows[3]["verdict"], "tx2",
         [[("Best: L = 72 ", {}), (f"{best_e2['mean_diff']:+.3f}", {"bold": True}), (f" IC (Holm p {p_text(best_e2['p_holm'])})", {})],
          [(f"{e2_rejects} of {len(e2_tests)} tests significant", {})]], "s12_step_3_icon.png"),
        ("E3", "Forecast the next 1–2 hours", "Worse", "accent2",
         [[("1 h: ", {}), (f"{e3_1['mean_diff']:+.3f}", {"bold": True}), (f" IC vs best (Holm p {p_text(e3_1['p_holm'])})", {})],
          [("2 h: ", {}), (f"{e3_2['mean_diff']:+.3f}", {"bold": True}), (f" IC vs best (Holm p {p_text(e3_2['p_holm'])})", {})]], "s10_clock_icon.png")]
for i, (tag, q_, verdict_, fill, evidence, png) in enumerate(cols):
    x = 0.6 + i * 4.11
    rect(s, x, 1.95, 3.91, 3.75, name=f"{tag} card")
    icon(s, x + 0.25, 2.15, 0.8, fill if fill != "tx2" else "accent5", png, tag)
    text(s, x + 1.25, 2.15, 2.5, 0.8, [[(tag, {"bold": True, "size": 13, "col": "tx2"})], [(q_, {"bold": True, "size": 16})]], anchor=MSO_ANCHOR.MIDDLE, name=f"{tag} head")
    text(s, x + 0.25, 3.2, 3.43, 0.7, verdict_, size=30, bold=True, col=fill if fill != "tx2" else "tx1", name=f"{tag} verdict")
    text(s, x + 0.25, 4.05, 3.43, 1.5, evidence, size=13.5, space_after=6, name=f"{tag} evidence")
rect(s, 0.6, 5.95, 12.13, 0.8, name="overall panel")
icon(s, 0.8, 6.02, 0.65, "accent3", "s6_why_0_icon.png", "overall")
text(s, 1.65, 5.95, 10.9, 0.8, [[("Together: ", {"bold": True}),
                                  ("the chain adds information only as a correction to a volatility model, not on its own.", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE, name="overall text")

# ---------------------------------------------------------------- R7 E1 detail
sm = e1["summary"]
e1_models = [("har", "HAR-RV", CLASSICAL), ("attention_gru_hpc", "Attention + GRU", OURS), ("state_mlp", "Volatility state only (MLP)", NEURAL),
             ("residual_attention", "HAR + attention correction", EXTRA), ("residual_lightgbm", "HAR + LightGBM correction", EXTRA),
             ("hybrid_attention_gru", "Hybrid: attention + GRU sees HAR", EXTRA), ("combination", "Average of HAR, LightGBM, ours", EXTRA)]
s = new_slide("E1: the chain helps as a correction to HAR",
              "Next-day IC for the E1 variants on the same 26 test periods. A correction model predicts what HAR gets wrong from the "
              "chain; added to HAR it lifts the IC significantly. A hybrid that gives attention plus GRU the HAR forecast, and a plain "
              "average of three forecasts, do best. The control: a small network that sees only the volatility state, no contracts, "
              "does about as well as attention plus GRU, so on its own the chain adds nothing detectable for the next day. "
              "Attention plus GRU here is the V100 rerun, the reference for all exploratory comparisons.",
              subtitle="Next-day IC on the same 26 test periods (exploratory; attention + GRU is the Explorer V100 reference)")
bar_chart(s, 0.45, 2.15, 8.6, 4.65, [label for _, label, _ in e1_models], [("IC", [sm[k]["ic"] for k, _, _ in e1_models])],
          [[c for _, _, c in e1_models]], horizontal=True, vmin=0, vmax=0.6, name="E1 chart", gap=40)
rect(s, 9.42, 1.85, 3.31, 4.95, name="E1 panel")
resid_lo = min(e1_tests[t]["mean"] for t in ("residual_lightgbm_ic", "residual_deepsets_ic", "residual_attention_ic"))
resid_hi = max(e1_tests[t]["mean"] for t in ("residual_lightgbm_ic", "residual_deepsets_ic", "residual_attention_ic"))
state = e1["secondary"]["state_mlp_vs_attention_gru_hpc"]["ic"]
text(s, 9.62, 2.05, 2.91, 4.6, [[("What HAR misses is predictable", {"bold": True, "size": 15})],
                                [(f"Correction models rank HAR's errors with IC {resid_lo:.2f}–{resid_hi:.2f}; all {sum(m['reject_holm'] for m in e1_tests.values())} "
                                  f"of {len(e1_tests)} E1 tests are significant.", {"size": 12.5, "col": "tx2"})],
                                [("But the volatility state does most of it", {"bold": True, "size": 15})],
                                [(f"Without any contracts the state-only model scores {sm['state_mlp']['ic']:.3f} vs {sm['attention_gru_hpc']['ic']:.3f} "
                                  f"(95% interval {state['ci95'][0]:+.3f} to {state['ci95'][1]:+.3f}).", {"size": 12.5, "col": "tx2"})],
                                [("Reading the chain on top of HAR helps", {"bold": True, "size": 15})],
                                [(f"The hybrid beats attention + GRU by {e1_tests['hybrid_vs_attention_gru']['mean_diff']:+.3f} "
                                  f"(Holm p {p_text(e1_tests['hybrid_vs_attention_gru']['p_holm'])}).", {"size": 12.5, "col": "tx2"})]],
     space_after=8, name="E1 text")
text(s, 0.6, 1.85, 8.4, 0.3, [[("classical", {"col": "accent5", "bold": True}), ("     ours", {"col": "accent3", "bold": True}),
                              ("     no contracts", {"col": "accent1", "bold": True}), ("     E1 corrections and combinations", {"col": "accent2", "bold": True})]],
     size=12, name="legend")

# ---------------------------------------------------------------- R8 E2 + E3
s = new_slide("More history doesn't help; over hours, simple wins",
              f"Left, E2: giving the next-day model a longer intraday window or the previous days' chains moves the IC by at most "
              f"{best_e2['mean_diff']:+.3f}; {e2_rejects} of {len(e2_tests)} tests are significant. Right, E3: a new target, volatility over "
              "the next one and two hours. Here intraday HAR and tuned LightGBM are clearly ahead of attention plus GRU at both horizons, "
              "and the gap is significant after Holm correction.",
              subtitle="E2: next-day IC with more history (V100)          E3: IC for volatility over the next 1 and 2 hours")
e2_runs = {"L = 12 (ref.)": sm["attention_gru_hpc"]["ic"]}
for key, label in (("window_24", "L = 24"), ("window_72", "L = 72"), ("multiday_5", "K = 5 days"), ("multiday_10", "K = 10 days")):
    e2_runs[label] = e2_tests[f"{key}_vs_l12"]["mean_a"]
bar_chart(s, 0.45, 1.85, 5.9, 4.4, list(e2_runs), [("IC", list(e2_runs.values()))], [[OURS] + [NEURAL] * 4], vmin=0, vmax=0.6, name="E2 chart", gap=50)
text(s, 0.6, 6.3, 5.7, 0.5, [[("E2: ", {"bold": True}), (f"{e2_rejects} of {len(e2_tests)} tests significant (Holm)", {})]], size=14, name="E2 note")
e3_models = ["intraday_har", "lightgbm", "attention_gru"]
bar_chart(s, 6.75, 1.85, 6.1, 4.4, ["Next 1 hour", "Next 2 hours"],
          [(NAMES[m], [e3[h]["scores"][m]["ic_mean_per_walk"] for h in ("1h", "2h")]) for m in e3_models],
          [CLASSICAL, NEURAL, OURS], vmin=0, vmax=0.7, legend=True, name="E3 chart", gap=60, label_size=11)
text(s, 6.9, 6.3, 5.8, 0.5, [[("E3: ", {"bold": True}), (f"ours {e3_1['mean_diff']:+.3f} and {e3_2['mean_diff']:+.3f} vs the best classical model "
                                                           f"(Holm p {p_text(e3_1['p_holm'])}, {p_text(e3_2['p_holm'])})", {})]], size=14, name="E3 note")

# ---------------------------------------------------------------- R9 takeaways
s = new_slide("What we learned",
              "Four takeaways. One: our model has the lowest forecast error of every model and beats GEX on QLIKE, and on ranking it "
              "matches the best classical models, though not by the clear margin we hoped for. Two: attention ends up averaging, "
              "so it behaves like a sum over contracts, much like GEX. Three: no sign of a break at the rule change, with limited power. "
              "Four, from the exploratory follow-ups: the chain does carry information, but only as a correction to a volatility model. "
              "The tests were fixed in advance, so these are real results, not tuning.")
take = [("Our model has the lowest error of every model",
         f"Lowest RMSE and QLIKE; QLIKE {verdict['har_gex']['qlike_lower_pct']:.0f}% below GEX (HAR + GEX). On ranking it matches the best "
         "classical models, not by a clear margin (Q1).", "s9_rq_0_icon.png", "accent3"),
        ("Attention ends up averaging",
         f"Pooling is a plain average on all but {ag['concentrated_days_any_head']} of {ag['prediction_snapshots']:,} days; no switched-off part matters (Q2).",
         "s7_attention_icon.png", "accent1"),
        ("No sign of a break at the rule change", f"Gain over LightGBM similar before and after Nov 2024 (p {rq3['p_two_sided']:.2f}), with only "
         f"{rq3['walks_after']} periods after (Q3).", "s9_rq_2_icon.png", "accent2"),
        ("The chain helps as a correction", f"HAR + correction or a forecast average adds {best_plus['mean_diff']:+.3f} to "
         f"{e1_tests['combination_vs_har']['mean_diff']:+.3f} IC; on its own, and over hours, it doesn't (exploratory).", "s12_step_2_icon.png", "accent5")]
for i, (head, body, png, fill) in enumerate(take):
    y = 1.6 + i * 1.22
    icon(s, 0.6, y + 0.05, 0.8, fill, png, f"take {i}")
    text(s, 1.65, y, 7.2, 1.0, [[(head, {"bold": True, "size": 19})], [(body, {"size": 14, "col": "tx2"})]], anchor=MSO_ANCHOR.MIDDLE, name=f"take {i} text")
rect(s, 9.42, 1.6, 3.31, 3.2, name="promise panel")
text(s, 9.67, 1.85, 2.81, 4.4, [[("We said in October", {"bold": True, "size": 17, "col": "accent3"})],
                                [("“The effect may be small, so tests must be careful.”", {"italic": True, "size": 16})],
                                [("It was: the model does well, with the lowest error of all, but the ranking gain is smaller than the "
                                  "test can confirm. The tests were fixed in advance, so this is a real result, not tuning.", {"size": 14, "col": "tx2"})]],
     space_after=12, name="promise text")

# ---------------------------------------------------------------- R10 delivered
s = new_slide("What we delivered",
              "Everything promised in the proposal is delivered: the leakage-checked pipeline, the fair comparison on the seven-rung "
              "ladder, the attention maps, and the report and code. The exploratory neural runs used Northeastern's Explorer cluster, "
              "with every model compared on the same GPU type. Every result records its code version and configuration.")
deliv = [("Clean data pipeline", "every row carries an availability time; leakage tests on every source", "s16_deliv_0_icon.png"),
         ("Fair comparison", "7 models, 26 walk-forward test periods, tests fixed in advance", "s16_deliv_1_icon.png"),
         ("Attention maps", "pooling weights for every test day and contract", "s16_deliv_2_icon.png"),
         ("Report and code", "every run records its code version and configuration", "s16_deliv_3_icon.png")]
for i, (head, body, png) in enumerate(deliv):
    x = 0.6 + i * 3.08
    rect(s, x, 1.6, 2.9, 2.5, name=f"deliv {i}")
    icon(s, x + 0.2, 1.8, 0.75, "accent1", png, f"deliv {i}")
    text(s, x + 0.2, 2.75, 2.5, 1.25, [[(head, {"bold": True, "size": 16})], [(body, {"size": 12.5, "col": "tx2"})]], name=f"deliv {i} text")
text(s, 0.6, 4.45, 12.13, 0.4, "Timeline", size=17, bold=True, col="accent3", name="timeline head")
rect(s, 0.75, 5.25, 9.9, 0.03, fill="tx2", name="timeline line")
steps = [("Oct 8", "Proposal"), ("Oct 29", "Data pipeline, first baselines"), ("November", "Models, switch-off and rule-change tests, E1–E3"),
         ("December", "Final presentation, code and report")]
for i, (when, what) in enumerate(steps):
    x = 0.75 + i * 3.3
    rect(s, x - 0.11, 5.155, 0.22, 0.22, fill="accent3" if i == len(steps) - 1 else "accent1", shape=MSO_SHAPE.OVAL, name=f"milestone {i}")
    text(s, x - 0.11, 5.5, 3.0, 0.8, [[(when, {"bold": True, "size": 15})], [(what, {"size": 12.5, "col": "tx2"})]], name=f"milestone {i} text")

# ---------------------------------------------------------------- backups (hidden)
def table(slide, x, y, w, header, body, widths, size=12):
    gf = slide.shapes.add_table(len(body) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(0.4 * (len(body) + 1)))
    t = gf.table
    t.first_row = True
    t.horz_banding = False
    for j, wd in enumerate(widths):
        t.columns[j].width = Inches(wd)
    for i, row in enumerate([header, *body]):
        t.rows[i].height = Inches(0.4)
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.fill.solid(); color(cell.fill.fore_color, "accent4" if i == 0 else "bg2")
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = ""
            r = tf.paragraphs[0].add_run(); r.text = str(value)
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            r.font.size = Pt(size); r.font.bold = i == 0
            color(r.font.color, "tx1")
    return gf


s = new_slide("Backup: full scoreboard",
              "Backup slide, hidden in slideshow. The full ladder with the 95% interval of the per-period IC, and the two loss scores. "
              "Historical mean has no IC because it predicts a constant.", hidden=True)
body = []
for r in results["ladder"]:
    ic_ = r["ic_mean_per_walk"]
    body.append([r["label"], "—" if ic_ is None else f"{ic_:.3f}", "—" if r["ic_ci95"] is None else f"{r['ic_ci95'][0]:.3f} to {r['ic_ci95'][1]:.3f}",
                 f"{r['rmse_mean_per_walk']:.3f}", f"{r['qlike_mean_per_walk']:.3f}"])
table(s, 0.6, 1.6, 12.13, ["Model", "IC", "95% interval", "RMSE", "QLIKE"], body, [4.13, 1.8, 2.6, 1.8, 1.8], size=14)

s = new_slide("Backup: E1 tests (Holm within E1)",
              "Backup slide, hidden in slideshow. All nine registered E1 tests, Holm-corrected within E1. The first three test whether the "
              "correction models rank HAR's errors at all; the rest compare forecasts with HAR or with attention plus GRU.", hidden=True)
LABEL = {"residual_lightgbm_ic": "LightGBM correction: IC on HAR's errors", "residual_deepsets_ic": "DeepSets correction: IC on HAR's errors",
         "residual_attention_ic": "Attention correction: IC on HAR's errors", "har_plus_residual_lightgbm_vs_har": "HAR + LightGBM correction vs HAR",
         "har_plus_residual_deepsets_vs_har": "HAR + DeepSets correction vs HAR", "har_plus_residual_attention_vs_har": "HAR + attention correction vs HAR",
         "hybrid_vs_har": "Hybrid vs HAR", "hybrid_vs_attention_gru": "Hybrid vs attention + GRU", "combination_vs_har": "Average of 3 forecasts vs HAR"}
body = [[LABEL.get(m["test"], m["test"]), f"{m.get('mean_diff', m.get('mean')):+.3f}", f"{m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f}",
         p_text(m["p_holm"]), "yes" if m["reject_holm"] else "no"] for m in e1["tests"]["members"]]
table(s, 0.6, 1.6, 12.13, ["Test", "IC (gain)", "95% interval", "Holm p", "Significant"], body, [5.33, 1.6, 2.4, 1.4, 1.4], size=13)

s = new_slide("Backup: E3 by time of day",
              "Backup slide, hidden in slideshow. IC within each snapshot time, so time-of-day effects cannot drive the score. "
              "The classical models stay ahead of attention plus GRU at every time.", hidden=True)
figure(s, 0.6, 1.6, 11.9, FIG / "e3_ic_by_time_of_day.png", "E3 figure")

s = new_slide("Backup: attention on two days",
              "Backup slide, hidden in slideshow. Pooling weights across the chain on a typical day, flat at one, and on the most "
              "concentrated day, where one newly opened put gets most of the weight.", hidden=True)
figure(s, 0.6, 1.6, 11.9, FIG / "attention_case_days.png", "attention days figure")

# ---------------------------------------------------------------- order: proposal 1-15, results, old backups, new backups, proposal plan
ids = prs.slides._sldIdLst
items = list(ids)
old, new = items[:len(old_slides)], items[len(old_slides):]
plan_slide = old[15]
plan_slide_obj = old_slides[15]
plan_slide_obj._element.set("show", "0")
MAIN_NEW = 11   # R0 (scores) ... R10 (delivered); the rest are hidden backups
order = old[:15] + new[:MAIN_NEW] + old[16:] + new[MAIN_NEW:] + [plan_slide]
for el in items:
    ids.remove(el)
for el in order:
    ids.append(el)
# cached slide-number text (PowerPoint recomputes the field)
for n, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.type == PP_PLACEHOLDER.SLIDE_NUMBER:
            for p in sh.text_frame.paragraphs:
                for fld in p._p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}fld"):
                    fld.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = str(n)
prs.save(str(OUT))
print(OUT, len(prs.slides), "slides")
