import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_DATASET_DIR = Path("dataset/Genesis")
DEFAULT_OUTPUT = Path("results/genesis_truth_visualization/genesis_truth_full.html")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize raw Genesis test data with ground-truth anomaly labels only.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variables", type=str, default="all", help="Variable list like all, 0,1,2, or 0-5,12.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--plot_height", type=int, default=520)
    parser.add_argument("--page_width", type=str, default="96vw")
    return parser.parse_args()


def load_plotly_script(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "plotly-2.35.2.min.js"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    try:
        from plotly.offline import get_plotlyjs

        script = get_plotlyjs()
    except Exception:
        with urllib.request.urlopen("https://cdn.plot.ly/plotly-2.35.2.min.js", timeout=30) as response:
            script = response.read().decode("utf-8")
    cache_path.write_text(script, encoding="utf-8")
    return script


def parse_variables(spec: str, num_variables: int):
    spec = str(spec or "all").strip().lower()
    if spec == "all":
        return list(range(num_variables))
    selected = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    values = sorted(selected)
    for value in values:
        if value < 0 or value >= num_variables:
            raise ValueError(f"Variable index out of range: {value}; valid range is 0-{num_variables - 1}")
    return values


def resolve_window(length: int, start, end):
    start = 0 if start is None else max(0, int(start))
    end = length if end is None else min(length, int(end))
    if start >= end:
        raise ValueError(f"Invalid window: start={start}, end={end}, length={length}")
    return start, end


def binary_segments(values):
    segments = []
    start = None
    for idx, value in enumerate(np.asarray(values).reshape(-1)):
        active = int(value) != 0
        if active and start is None:
            start = idx
        elif not active and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


def to_json_array(values):
    arr = np.asarray(values).reshape(-1)
    if arr.dtype.kind in {"i", "u", "b"}:
        return "[" + ",".join(str(int(x)) for x in arr) + "]"
    return "[" + ",".join(f"{float(x):.8g}" for x in arr) + "]"


def build_html(output_path: Path, x, series, labels, variables, segments, plot_height: int, page_width: str):
    plotly_script = load_plotly_script(output_path.parent)
    variable_payload = []
    for idx in variables:
        variable_payload.append("{" + f"idx:{int(idx)},values:{to_json_array(series[:, idx])}" + "}")
    divs = "\n".join(f'<div id="var_{idx}_plot" class="plot"></div>' for idx in variables)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Genesis Ground Truth Anomaly Labels</title>
  <script>
{plotly_script}
  </script>
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
      margin: 0 auto 18px auto;
      line-height: 1.45;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 6px 0;
    }}
    .meta {{
      color: #526070;
      font-size: 13px;
    }}
    .plot {{
      width: {page_width};
      height: {int(plot_height)}px;
      margin: 14px auto;
      background: white;
      border: 1px solid #e1e5ed;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Genesis Raw Test Data With Ground-Truth Labels</h1>
    <div class="meta">
      points={len(x)} | anomaly_points={int(np.sum(labels != 0))} |
      anomaly_segments={json.dumps([[int(a), int(b - 1)] for a, b in segments])}
    </div>
  </div>
  {divs}
  <script>
    const x = {to_json_array(x)};
    const truth = {to_json_array(labels)};
    const truthSegments = {json.dumps([[int(a), int(b)] for a, b in segments])};
    const variables = [{",".join(variable_payload)}];

    function segmentShapes(segments) {{
      return segments.map(([a, b]) => ({{
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: x[a],
        x1: x[Math.max(a, b - 1)],
        y0: 0,
        y1: 1,
        fillcolor: "#2ca25f",
        opacity: 0.18,
        line: {{width: 0}},
        layer: "below"
      }}));
    }}

    const layoutBase = {{
      margin: {{l: 64, r: 28, t: 42, b: 42}},
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      hovermode: "x unified",
      dragmode: "zoom",
      xaxis: {{title: "test point index", rangeslider: {{visible: true, thickness: 0.08}}}},
      legend: {{orientation: "h", y: 1.12}},
      shapes: segmentShapes(truthSegments)
    }};
    const plotConfig = {{
      responsive: true,
      scrollZoom: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }};

    for (const item of variables) {{
      Plotly.newPlot(`var_${{item.idx}}_plot`, [
        {{x, y: item.values, type: "scatter", mode: "lines", name: `dim ${{item.idx}} raw`, line: {{color: "#364152", width: 1.1}}}},
        {{x, y: truth, type: "scatter", mode: "lines", name: "ground truth label", yaxis: "y2", line: {{color: "#2ca25f", width: 1}}}}
      ], {{
        ...layoutBase,
        title: `Raw Test Variable dim ${{item.idx}}`,
        yaxis: {{title: "raw value"}},
        yaxis2: {{title: "truth label", overlaying: "y", side: "right", range: [-0.1, 1.1], showgrid: false}}
      }}, plotConfig);
    }}
  </script>
</body>
</html>
"""
    return html


def main():
    args = parse_args()
    test = np.load(args.dataset_dir / "Genesis_test.npy").astype(np.float32)
    labels = np.load(args.dataset_dir / "Genesis_test_label.npy").reshape(-1).astype(np.int32)
    length = min(len(test), len(labels))
    test = test[:length]
    labels = labels[:length]
    start, end = resolve_window(length, args.start, args.end)
    variables = parse_variables(args.variables, test.shape[1])
    x = np.arange(start, end, dtype=np.int64)
    window_labels = labels[start:end]
    window_series = test[start:end]
    segments = binary_segments(window_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(
            output_path=args.output,
            x=x,
            series=window_series,
            labels=window_labels,
            variables=variables,
            segments=segments,
            plot_height=args.plot_height,
            page_width=args.page_width,
        ),
        encoding="utf-8",
    )
    print(f"saved_html = {args.output}")
    print(f"window = [{start}, {end})")
    print(f"variables = {variables}")
    print(f"truth_anomaly_points = {int(np.sum(window_labels != 0))}")
    print(f"truth_segments = {[(int(a + start), int(b - 1 + start)) for a, b in segments]}")


if __name__ == "__main__":
    main()
