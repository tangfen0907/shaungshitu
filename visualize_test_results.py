import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_RUN_DIR = Path("results/genesis/genesis_experiment087")
DEFAULT_DATASET_DIR = Path("dataset/Genesis")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive HTML view of the current FuKAN test results: "
            "raw variables, anomaly score, predictions, and ground truth."
        )
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--score", type=str, default="score_local_sum")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--variables",
        type=str,
        default="0,1,2,3,4,6,7,12,13",
        help="Variable list like 'all', '0,1,4', or '0-5,12'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run_dir>/test_result_visualizations/<score>_<window>.html",
    )
    parser.add_argument(
        "--plotly_mode",
        choices=("cdn", "inline"),
        default="inline",
        help="cdn makes a smaller HTML; inline makes a self-contained offline HTML.",
    )
    parser.add_argument("--plot_height", type=int, default=560, help="Height of each plot in pixels.")
    parser.add_argument("--page_width", type=str, default="96vw", help="CSS width for plots, e.g. 96vw or 1400px.")
    return parser.parse_args()


def load_array(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return np.load(path)


def parse_variables(spec: str, num_variables: int):
    spec = str(spec or "all").strip().lower()
    if spec == "all":
        return list(range(num_variables))

    selected = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid variable range: {item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))

    variables = sorted(selected)
    for idx in variables:
        if idx < 0 or idx >= num_variables:
            raise ValueError(f"Variable index out of range: {idx}; valid range is 0-{num_variables - 1}")
    return variables


def resolve_window(length: int, start, end):
    start = 0 if start is None else max(0, int(start))
    end = length if end is None else min(length, int(end))
    if start >= end:
        raise ValueError(f"Invalid window: start={start}, end={end}, length={length}")
    return start, end


def binary_segments(values):
    values = np.asarray(values).reshape(-1)
    segments = []
    start = None
    for idx, value in enumerate(values):
        active = int(value) != 0
        if active and start is None:
            start = idx
        elif not active and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


def read_threshold(run_dir: Path, score_name: str):
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    families = payload.get("score_families") or payload.get("component_families") or {}
    score_payload = families.get(score_name) or {}
    value = score_payload.get("threshold")
    return None if value is None else float(value)


def score_file(run_dir: Path, score_name: str, suffix: str):
    path = run_dir / f"{score_name}_{suffix}.npy"
    if path.exists():
        return path
    component_path = run_dir / f"component_{score_name}.npy"
    if suffix == "test_scores" and component_path.exists():
        return component_path
    raise FileNotFoundError(f"Could not find {score_name}_{suffix}.npy under {run_dir}")


def default_output_path(run_dir: Path, score_name: str, start: int, end: int):
    out_dir = run_dir / "test_result_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{score_name}_{start}_{end}.html"


def load_plotly_script(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "plotly-2.35.2.min.js"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    try:
        from plotly.offline import get_plotlyjs

        script = get_plotlyjs()
    except Exception:
        url = "https://cdn.plot.ly/plotly-2.35.2.min.js"
        with urllib.request.urlopen(url, timeout=30) as response:
            script = response.read().decode("utf-8")

    cache_path.write_text(script, encoding="utf-8")
    return script


def plotly_loader_html(mode: str, cache_dir: Path):
    if mode == "cdn":
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    script = load_plotly_script(cache_dir)
    return f"<script>\n{script}\n</script>"


def to_json_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {"i", "u", "b"}:
        return "[" + ",".join(str(int(x)) for x in arr.reshape(-1)) + "]"
    return "[" + ",".join(f"{float(x):.8g}" for x in arr.reshape(-1)) + "]"


def build_html(
    *,
    dataset_name,
    score_name,
    output_path,
    x,
    variables,
    series,
    score,
    pred,
    truth,
    threshold,
    full_segments,
    pred_segments,
    plotly_mode,
    plot_height,
    page_width,
):
    div_ids = ["score_plot"] + [f"var_{idx}_plot" for idx in variables]
    plot_divs = "\n".join(f'<div id="{div_id}" class="plot"></div>' for div_id in div_ids)
    threshold_line = "null" if threshold is None else f"{threshold:.10g}"

    variable_payload = []
    for idx in variables:
        variable_payload.append(
            "{"
            f"idx:{int(idx)},"
            f"values:{to_json_array(series[:, idx])}"
            "}"
        )

    plotly_loader = plotly_loader_html(plotly_mode, output_path.parent)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{dataset_name} {score_name} Test Results</title>
  {plotly_loader}
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: Arial, Helvetica, sans-serif;
      color: #172033;
      background: #f7f8fb;
    }}
    .header {{
      width: {page_width};
      max-width: none;
      margin: 0 auto 18px auto;
      line-height: 1.45;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 6px 0;
      font-weight: 700;
    }}
    .meta {{
      color: #526070;
      font-size: 13px;
    }}
    .plot {{
      width: {page_width};
      max-width: none;
      height: {int(plot_height)}px;
      margin: 14px auto;
      background: white;
      border: 1px solid #e1e5ed;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{dataset_name} Test Result View</h1>
    <div class="meta">
      score={score_name} | output={output_path.as_posix()} | points={len(x)} |
      true_anomaly={int(np.sum(truth != 0))} | pred_anomaly={int(np.sum(pred != 0))} |
      threshold={threshold if threshold is not None else "none"}
    </div>
  </div>
  {plot_divs}
  <script>
    const x = {to_json_array(x)};
    const score = {to_json_array(score)};
    const pred = {to_json_array(pred)};
    const truth = {to_json_array(truth)};
    const threshold = {threshold_line};
    const truthSegments = {json.dumps([[int(a), int(b)] for a, b in full_segments])};
    const predSegments = {json.dumps([[int(a), int(b)] for a, b in pred_segments])};
    const variables = [{",".join(variable_payload)}];

    function segmentShapes(segments, color, opacity) {{
      return segments.map(([a, b]) => ({{
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: x[a],
        x1: x[Math.max(a, b - 1)],
        y0: 0,
        y1: 1,
        fillcolor: color,
        opacity: opacity,
        line: {{width: 0}},
        layer: "below"
      }}));
    }}

    const baseShapes = [
      ...segmentShapes(truthSegments, "#2ca25f", 0.17),
      ...segmentShapes(predSegments, "#f28e2b", 0.14)
    ];

    const scoreShapes = [...baseShapes];
    if (threshold !== null) {{
      scoreShapes.push({{
        type: "line",
        xref: "x",
        yref: "y",
        x0: x[0],
        x1: x[x.length - 1],
        y0: threshold,
        y1: threshold,
        line: {{color: "#d62728", width: 1.5, dash: "dash"}}
      }});
    }}

    const layoutBase = {{
      margin: {{l: 64, r: 28, t: 42, b: 42}},
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      hovermode: "x unified",
      dragmode: "zoom",
      xaxis: {{title: "test point index", rangeslider: {{visible: true, thickness: 0.08}}}},
      legend: {{orientation: "h", y: 1.12}},
    }};

    const plotConfig = {{
      responsive: true,
      scrollZoom: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }};

    Plotly.newPlot("score_plot", [
      {{x, y: score, type: "scatter", mode: "lines", name: "{score_name}", line: {{color: "#1f4e79", width: 1.2}}}},
      {{x, y: truth, type: "scatter", mode: "lines", name: "truth", yaxis: "y2", line: {{color: "#2ca25f", width: 1}}}},
      {{x, y: pred, type: "scatter", mode: "lines", name: "pred", yaxis: "y2", line: {{color: "#f28e2b", width: 1}}}}
    ], {{
      ...layoutBase,
      title: "Anomaly Score / Truth / Prediction",
      shapes: scoreShapes,
      yaxis: {{title: "score"}},
      yaxis2: {{title: "label", overlaying: "y", side: "right", range: [-0.1, 1.1], showgrid: false}}
    }}, plotConfig);

    for (const item of variables) {{
      const divId = `var_${{item.idx}}_plot`;
      Plotly.newPlot(divId, [
        {{x, y: item.values, type: "scatter", mode: "lines", name: `dim ${{item.idx}}`, line: {{color: "#364152", width: 1.1}}}},
        {{x, y: truth, type: "scatter", mode: "lines", name: "truth", yaxis: "y2", line: {{color: "#2ca25f", width: 1}}}},
        {{x, y: pred, type: "scatter", mode: "lines", name: "pred", yaxis: "y2", line: {{color: "#f28e2b", width: 1}}}}
      ], {{
        ...layoutBase,
        title: `Raw Test Variable dim ${{item.idx}}`,
        shapes: baseShapes,
        yaxis: {{title: "raw value"}},
        yaxis2: {{title: "label", overlaying: "y", side: "right", range: [-0.1, 1.1], showgrid: false}}
      }}, plotConfig);
    }}
  </script>
</body>
</html>
"""
    return html


def main():
    args = parse_args()
    run_dir = args.run_dir
    dataset_dir = args.dataset_dir

    test_series = load_array(dataset_dir / "Genesis_test.npy", "Genesis_test.npy")
    truth_path = run_dir / "y_true.npy"
    truth = load_array(truth_path, "y_true.npy").reshape(-1).astype(np.int32)
    score = load_array(score_file(run_dir, args.score, "test_scores"), f"{args.score}_test_scores").reshape(-1)
    pred = load_array(score_file(run_dir, args.score, "pred_labels"), f"{args.score}_pred_labels").reshape(-1).astype(np.int32)

    length = min(len(test_series), len(truth), len(score), len(pred))
    test_series = np.asarray(test_series[:length], dtype=np.float32)
    truth = truth[:length]
    score = np.asarray(score[:length], dtype=np.float32)
    pred = pred[:length]

    start, end = resolve_window(length, args.start, args.end)
    variables = parse_variables(args.variables, test_series.shape[1])
    output_path = args.output or default_output_path(run_dir, args.score, start, end)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    window_truth = truth[start:end]
    window_pred = pred[start:end]
    window_score = score[start:end]
    window_series = test_series[start:end]
    x = np.arange(start, end, dtype=np.int64)

    threshold = read_threshold(run_dir, args.score)
    html = build_html(
        dataset_name="Genesis",
        score_name=args.score,
        output_path=output_path,
        x=x,
        variables=variables,
        series=window_series,
        score=window_score,
        pred=window_pred,
        truth=window_truth,
        threshold=threshold,
        full_segments=binary_segments(window_truth),
        pred_segments=binary_segments(window_pred),
        plotly_mode=args.plotly_mode,
        plot_height=args.plot_height,
        page_width=args.page_width,
    )
    output_path.write_text(html, encoding="utf-8")

    print(f"saved_html = {output_path}")
    print(f"run_dir = {run_dir}")
    print(f"score = {args.score}")
    print(f"window = [{start}, {end})")
    print(f"variables = {variables}")
    print(f"truth_anomaly_points = {int(np.sum(window_truth != 0))}")
    print(f"pred_anomaly_points = {int(np.sum(window_pred != 0))}")


if __name__ == "__main__":
    main()
