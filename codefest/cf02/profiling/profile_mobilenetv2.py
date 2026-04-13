import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

OUTPUT_PATH = r"C:\Users\Husai\Ece-410-hardware-ai\codefest\cf02\profiling\project_profile.txt"
NUM_RUNS = 10

model = models.mobilenet_v2(weights=None)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Warmup (not counted)
with torch.no_grad():
    for _ in range(2):
        model(dummy_input)

with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for i in range(NUM_RUNS):
            with record_function(f"inference_run_{i}"):
                model(dummy_input)

table = prof.key_averages(group_by_input_shape=True).table(
    sort_by="cpu_time_total", row_limit=200
)

print(table)

with open(OUTPUT_PATH, "w") as f:
    f.write(f"MobileNetV2 FP32 Inference Profiling\n")
    f.write(f"Batch size: 1 | Input: 3x224x224 | Runs: {NUM_RUNS}\n")
    f.write(f"Device: CPU | torch: {torch.__version__}\n")
    f.write("=" * 80 + "\n\n")
    f.write(table)

print(f"\nProfile saved to: {OUTPUT_PATH}")
