"""
MobileNetV2 FP32 Software Baseline Benchmark
Wall-clock median latency over 10 timed runs (after 5 warmup passes).
Peak RSS memory tracked via psutil.
"""
import time, statistics, os, gc
import torch
import torchvision.models as models
import psutil

WARMUP_RUNS  = 5
TIMED_RUNS   = 10
BATCH_SIZE   = 1
INPUT_SHAPE  = (BATCH_SIZE, 3, 224, 224)
DEVICE       = "cpu"

proc = psutil.Process(os.getpid())

model = models.mobilenet_v2(weights=None).to(DEVICE)
model.eval()

dummy = torch.randn(*INPUT_SHAPE)

# ── Warmup ────────────────────────────────────────────────────────────────────
with torch.no_grad():
    for _ in range(WARMUP_RUNS):
        _ = model(dummy)

gc.collect()

# ── Baseline RSS before timed runs ────────────────────────────────────────────
mem_before_mb = proc.memory_info().rss / 1024**2
peak_rss_mb   = mem_before_mb

# ── Timed runs ────────────────────────────────────────────────────────────────
latencies_ms = []

with torch.no_grad():
    for i in range(TIMED_RUNS):
        t0 = time.perf_counter()
        out = model(dummy)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        rss = proc.memory_info().rss / 1024**2
        if rss > peak_rss_mb:
            peak_rss_mb = rss

# ── Stats ─────────────────────────────────────────────────────────────────────
median_ms  = statistics.median(latencies_ms)
mean_ms    = statistics.mean(latencies_ms)
stdev_ms   = statistics.stdev(latencies_ms)
min_ms     = min(latencies_ms)
max_ms     = max(latencies_ms)
p90_ms     = sorted(latencies_ms)[int(0.9 * TIMED_RUNS) - 1]

throughput = (BATCH_SIZE / (median_ms / 1000.0))   # samples/sec
model_mb   = sum(p.numel() * p.element_size()
                 for p in model.parameters()) / 1024**2
delta_rss  = peak_rss_mb - mem_before_mb

SEP  = "=" * 55
LINE = "-" * 55
print(SEP)
print("  MobileNetV2 FP32 - Software Baseline Benchmark")
print(SEP)
print(f"  Warmup runs    : {WARMUP_RUNS}")
print(f"  Timed runs     : {TIMED_RUNS}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Input shape    : {list(INPUT_SHAPE)}")
print(f"  Device         : {DEVICE.upper()}")
print(LINE)
print(f"  Median latency : {median_ms:.3f} ms")
print(f"  Mean latency   : {mean_ms:.3f} ms")
print(f"  Std dev        : {stdev_ms:.3f} ms")
print(f"  Min latency    : {min_ms:.3f} ms")
print(f"  Max latency    : {max_ms:.3f} ms")
print(f"  P90 latency    : {p90_ms:.3f} ms")
print(f"  Throughput     : {throughput:.1f} samples/sec")
print(LINE)
print(f"  Model params   : {sum(p.numel() for p in model.parameters()):,}")
print(f"  Model size     : {model_mb:.2f} MB  (FP32 weights)")
print(f"  RSS before     : {mem_before_mb:.1f} MB")
print(f"  Peak RSS       : {peak_rss_mb:.1f} MB")
print(f"  Delta RSS      : {delta_rss:.1f} MB")
print(LINE)
print("Per-run latencies (ms):")
for i, t in enumerate(latencies_ms):
    print(f"  Run {i+1:02d}: {t:.3f} ms")
print(SEP)

# ── Emit structured data for the markdown writer ──────────────────────────────
import json, pathlib
results = {
    "median_ms":    round(median_ms,  3),
    "mean_ms":      round(mean_ms,    3),
    "stdev_ms":     round(stdev_ms,   3),
    "min_ms":       round(min_ms,     3),
    "max_ms":       round(max_ms,     3),
    "p90_ms":       round(p90_ms,     3),
    "throughput":   round(throughput, 1),
    "model_mb":     round(model_mb,   2),
    "peak_rss_mb":  round(peak_rss_mb,1),
    "delta_rss_mb": round(delta_rss,  1),
    "latencies_ms": [round(x, 3) for x in latencies_ms],
    "num_params":   sum(p.numel() for p in model.parameters()),
}
out = pathlib.Path(r"C:\Users\Husai\Ece-410-hardware-ai\project\m1\baseline_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"Results JSON: {out}")
