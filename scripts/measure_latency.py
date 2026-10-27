"""
@file measure_latency.py
@brief CPU inference latency for each ablation architecture.

@details
Phase 4. Parameter count is a poor proxy for deployment cost in this model.
The CNN stack parallelises; the LSTM does not -- it walks 156 timesteps in
sequence, and that walk is most of the wall clock on a CPU. Two architectures
with the same parameter count can therefore differ substantially in latency,
and the architecture the search selected is deliberately parameter-cheap and
timestep-expensive: it uses the *longest* sequence available (`4,2,2`, pooling
16x) with a small recurrent state.

So "246K parameters is enough" is not yet a deployment claim. "A recording is
scored in X seconds on a clinic workstation with no GPU" is, and this measures
it.

@par No trained weights needed
Latency is a property of the architecture, not of what it learned. Every point
is instantiated from its config and timed with random input, so this runs
before, during or after the GPU sweep and needs no checkpoints.

@par What a recording costs
The corpus averages **274 segments per recording** (743,578 segments over
2,717 recordings), so per-recording latency is 274 forward passes. That is the
number a clinician's wait is made of, and it is reported directly rather than
left as a multiplication for the reader.

@par Why median, and why fixed threads
A shared login node has other people on it. The mean of a timing sample is
dragged by whatever else was scheduled; the median is not, so the median over
repeats is reported with the inter-quartile spread beside it as an honesty
check -- a wide spread means the machine was busy and the number should be
re-taken. Thread count is pinned and reported, because a latency figure
without a core count is not a measurement.

@par Batch size
Batch 1 is the latency-critical case: one segment in, one score out, nothing to
amortise over. Larger batches amortise Python and kernel-launch overhead and
are what a real scoring pass would use, since all 274 segments of a recording
are available at once. Both are reported; the per-recording figure uses the
batched path, which is the honest deployment number.

@par Usage:
@verbatim
python -m scripts.measure_latency --points configs/ablation_shape.json
python -m scripts.measure_latency --points configs/ablation_shape.json \
    --threads 1,4,8 --batch 1,32 --repeats 20 --out results/latency.json
python -m scripts.measure_latency --points configs/ablation_shape.json --quick
@endverbatim
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Optional

## @brief Segments in an average recording: 743,578 over 2,717 recordings.
SEGMENTS_PER_RECORDING = 274

## @brief Input shape per segment: 19 channels, 10 s at 250 Hz.
N_CHANNELS = 19
N_TIMESTEPS = 2500


def build(overrides: dict[str, Any]) -> Any:
    """
    @brief Instantiate one architecture on the CPU, in eval mode.

    @details
    Goes through `build_model_config`, the same entry point the trainer uses,
    so a config this accepts is a config that trains.

    @param overrides The point's config override tree.
    @return The model, on CPU, in eval mode.
    """
    from eeg_cnn_lstm.models.model import CNN_LSTM
    from eeg_cnn_lstm.models.train import build_model_config

    model = CNN_LSTM(build_model_config(overrides["model"]))
    model.eval()
    return model


def time_forward(
    model: Any,
    batch: int,
    repeats: int,
    warmup: int,
    channels: int,
    timesteps: int,
) -> tuple[float, float]:
    """
    @brief Time one forward pass, robustly.

    @details
    Warmup passes are discarded: the first call through a fresh model pays for
    lazy allocation and cache population that no later call pays again, and
    including it would overstate steady-state latency by a wide margin.

    @param model The model to time.
    @param batch Segments per forward pass.
    @param repeats Timed iterations.
    @param warmup Discarded iterations before timing.
    @param channels Input channels.
    @param timesteps Samples per segment.
    @return Tuple `(median_seconds, iqr_seconds)` for one forward pass.
    """
    import torch

    x = torch.randn(batch, channels, timesteps)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times.sort()
    med = st.median(times)
    iqr = times[int(0.75 * (len(times) - 1))] - times[int(0.25 * (len(times) - 1))]
    return med, iqr


def measure(
    points: list[dict[str, Any]],
    threads: list[int],
    batches: list[int],
    repeats: int,
    warmup: int,
    channels: int,
    timesteps: int,
) -> list[dict[str, Any]]:
    """
    @brief Measure every point at every thread count and batch size.

    @param points Configuration entries carrying `overrides`.
    @param threads Thread counts to pin for each measurement.
    @param batches Batch sizes to time.
    @param repeats Timed iterations per setting.
    @param warmup Discarded iterations per setting.
    @param channels Input channels.
    @param timesteps Samples per segment.
    @return One record per point, with a nested timing per thread/batch.
    """
    import torch

    out = []
    for i, p in enumerate(points, 1):
        model = build(p["overrides"])
        n_par = int(model.num_parameters())
        rec: dict[str, Any] = {
            "name": p["name"],
            "axis": p.get("axis"),
            "level": p.get("level"),
            "n_parameters": n_par,
            "timings": {},
        }
        print(f"[{i}/{len(points)}] {p['name']:20} {n_par:>9,} params", flush=True)
        for t in threads:
            torch.set_num_threads(t)
            for b in batches:
                med, iqr = time_forward(model, b, repeats, warmup, channels, timesteps)
                per_seg_ms = med / b * 1000.0
                rec["timings"][f"t{t}_b{b}"] = {
                    "threads": t,
                    "batch": b,
                    "median_s": med,
                    "iqr_s": iqr,
                    "per_segment_ms": per_seg_ms,
                    "per_recording_s": per_seg_ms * SEGMENTS_PER_RECORDING / 1000.0,
                    "noisy": iqr > 0.25 * med,
                }
        out.append(rec)
    return out


def report(records: list[dict[str, Any]], threads: list[int], batches: list[int]) -> None:
    """
    @brief Print the latency table, sorted by parameter count.

    @details
    Sorting by parameter count puts the question on the page directly: if
    latency does not rise monotonically with parameters, then parameter count
    is the wrong cost axis and the deliverable should report seconds.

    @param records Output of `measure`.
    @param threads Thread counts measured.
    @param batches Batch sizes measured.
    """
    rs = sorted(records, key=lambda r: r["n_parameters"])
    for t in threads:
        for b in batches:
            key = f"t{t}_b{b}"
            print(f"\n--- {t} thread(s), batch {b} ---")
            print(f"  {'point':20} {'params':>10} {'ms/segment':>11} {'s/recording':>12}  ")
            for r in rs:
                d = r["timings"][key]
                flag = "  <- noisy" if d["noisy"] else ""
                print(
                    f"  {r['name']:20} {r['n_parameters']:>10,} "
                    f"{d['per_segment_ms']:>11.2f} {d['per_recording_s']:>12.2f}{flag}"
                )

    key = f"t{threads[0]}_b{batches[-1]}"
    lo = min(rs, key=lambda r: r["timings"][key]["per_recording_s"])
    hi = max(rs, key=lambda r: r["timings"][key]["per_recording_s"])
    print(f"\nAt {threads[0]} thread(s), batch {batches[-1]}:")
    print(
        f"  fastest {lo['name']} at {lo['timings'][key]['per_recording_s']:.2f} s/recording "
        f"({lo['n_parameters']:,} params)"
    )
    print(
        f"  slowest {hi['name']} at {hi['timings'][key]['per_recording_s']:.2f} s/recording "
        f"({hi['n_parameters']:,} params)"
    )

    # Does latency actually follow parameter count? If not, say so plainly.
    pars = [r["n_parameters"] for r in rs]
    secs = [r["timings"][key]["per_recording_s"] for r in rs]
    inversions = sum(
        1
        for i in range(len(rs))
        for j in range(i + 1, len(rs))
        if pars[i] < pars[j] and secs[i] > secs[j]
    )
    total = len(rs) * (len(rs) - 1) // 2
    print(f"\n  pairs where the SMALLER model is SLOWER: {inversions}/{total}")
    if inversions > total * 0.15:
        print(
            "  -> Latency does not track parameter count. Report seconds, not\n"
            "     parameters, as the deployment cost: the LSTM walks its sequence\n"
            "     in order and a shorter sequence can beat a smaller model."
        )

    if any(d["noisy"] for r in rs for d in r["timings"].values()):
        print(
            "\n  Some measurements are marked noisy (IQR above a quarter of the\n"
            "  median). The node was busy; re-take those before quoting them."
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Namespace of options.
    """
    p = argparse.ArgumentParser(description="CPU inference latency per architecture.")
    p.add_argument(
        "--points",
        default="configs/ablation_shape.json",
        help="Point set to measure; any file with `configs[].overrides`",
    )
    p.add_argument("--threads", default="1,4", help="Comma-separated thread counts")
    p.add_argument("--batch", default="1,32", help="Comma-separated batch sizes")
    p.add_argument("--repeats", type=int, default=20, help="Timed iterations")
    p.add_argument("--warmup", type=int, default=3, help="Discarded iterations")
    p.add_argument("--channels", type=int, default=N_CHANNELS)
    p.add_argument("--timesteps", type=int, default=N_TIMESTEPS)
    p.add_argument("--out", default=None, help="Write results JSON here")
    p.add_argument(
        "--quick",
        action="store_true",
        help="One thread, batch 32, 5 repeats; a smoke test, not a measurement",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    @brief CLI entrypoint.
    @return Process exit code.
    """
    a = parse_args(argv)
    path = Path(a.points)
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    doc = json.loads(path.read_text())
    points = [c for c in doc.get("configs", []) if "overrides" in c]
    if not points:
        print(f"error: no points with 'overrides' in {path}", file=sys.stderr)
        return 1

    threads = [1] if a.quick else [int(x) for x in a.threads.split(",") if x.strip()]
    batches = [32] if a.quick else [int(x) for x in a.batch.split(",") if x.strip()]
    repeats = 5 if a.quick else a.repeats

    try:
        import torch
    except ImportError:
        print("error: torch is required", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"CPU latency, {len(points)} architectures")
    print(f"  machine : {platform.processor() or platform.machine()}")
    print(f"  torch   : {torch.__version__}")
    print(f"  threads : {threads}   batches: {batches}   repeats: {repeats}")
    print(f"  input   : {a.channels} x {a.timesteps}  ({SEGMENTS_PER_RECORDING} per recording)")
    print("=" * 78)
    if a.quick:
        print("--quick: a smoke test, not a measurement. Do not quote these.\n")

    records = measure(points, threads, batches, repeats, a.warmup, a.channels, a.timesteps)
    report(records, threads, batches)

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "segments_per_recording": SEGMENTS_PER_RECORDING,
                    "torch": torch.__version__,
                    "machine": platform.processor() or platform.machine(),
                    "threads": threads,
                    "batches": batches,
                    "repeats": repeats,
                    "quick": bool(a.quick),
                    "records": records,
                },
                indent=2,
            )
        )
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())