# Heilmeier Catechism — Project
**Hussain Alquzweni — ECE 410, Spring 2026**
**Project Topic: Workload Analysis and Hardware Efficiency of Quantized MobileNetV2**

---

## Question 1: What are you trying to do?

I want to measure how much computation, memory, and energy a popular image-recognition
neural network (MobileNetV2) requires when running at full precision (FP32) versus a
compressed, lower-precision version (INT8 quantization). The goal is to understand
concretely how quantization reduces hardware cost — in terms of multiply-accumulate
operations, memory bandwidth, and arithmetic intensity — and to determine whether these
savings justify the potential loss in accuracy. Ultimately, I want to identify which
layers of MobileNetV2 are the real bottlenecks and propose a hardware/software partition
that could realistically accelerate the model on a custom chip.

---

## Question 2: How is it done today, and what are the limits of current practice?

Today, MobileNetV2 is widely deployed on mobile and edge devices for image classification.
Engineers typically apply post-training quantization (PTQ) using frameworks like PyTorch or
TensorFlow Lite to convert FP32 models to INT8. While tools like `torch.profiler` can
measure layer-wise runtime, they do not easily expose a side-by-side hardware efficiency
comparison between FP32 and INT8 at the layer level.

Profiling MobileNetV2 FP32 inference on an Intel i7-12700KF reveals the fundamental
limit: the dominant kernel — a depthwise Conv2d operating on a [1, 384, 14, 14] activation
map with a [384, 1, 3, 3] filter — has an **arithmetic intensity of only 2.20 FLOP/byte**,
placing it nearly 5× below the processor's ridge point of 10 FLOP/byte. This layer is
**memory-bound**, not compute-bound: the CPU's DRAM bandwidth (76.8 GB/s) is the
bottleneck, and more than 30% of inference time is spent in this kernel family. Most
existing analyses focus on accuracy drop rather than identifying these hardware efficiency
limits at the roofline level.

---

## Question 3: What is new in your approach and why do you think it will be successful?

This project performs a structured, roofline-guided workload accounting of MobileNetV2 in
both FP32 and INT8, computing MACs, weight memory, activation memory, and arithmetic
intensity for each precision — and then proposes a concrete hardware accelerator design
based on those measurements.

The key innovation in the approach is coupling quantization with architectural
co-design: rather than simply benchmarking INT8 accuracy, the project designs an
accelerator that targets the identified bottleneck directly. Because the depthwise
convolution family is memory-bound at AI = 2.20 FLOP/byte, the proposed hardware
accelerator will use **on-chip SRAM weight buffering** to reuse filter weights across
the output spatial map, lifting the effective arithmetic intensity to approximately
**50 FLOP/byte** — pushing the operating point past the i7-12700KF ridge and into the
compute-bound regime where added MAC throughput yields proportional speedup.
INT8 quantization reinforces this by halving weight and activation data volumes,
further raising AI and reducing required interface bandwidth to the accelerator.

This approach is likely to succeed because the profiling data, roofline model, and
HW/SW partition rationale are already established from Codefest 2, the quantization
toolchain (PyTorch PTQ API) is well-documented, and the analytical framework maps
directly onto the hardware concepts covered in this course.
