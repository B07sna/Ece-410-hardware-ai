# Heilmeier Catechism — Project Draft
**Hussain Alquzweni — ECE 410, Spring 2026**  
**Project Topic: Workload Analysis and Hardware Efficiency of Quantized MobileNetV2**

---

## Question 1: What are you trying to do?

Articulate your objectives using absolutely no jargon.

I want to measure how much computation, memory, and energy a popular image-recognition neural network (MobileNetV2) requires when running at full precision (FP32) versus a compressed, lower-precision version (INT8 quantization). The goal is to understand concretely how quantization reduces hardware cost — in terms of multiply-accumulate operations, memory bandwidth, and arithmetic intensity — and whether these savings are worth the potential loss in accuracy.

---

## Question 2: How is it done today, and what are the limits of current practice?

Today, MobileNetV2 is widely deployed in mobile and edge devices for tasks like image classification. Engineers typically apply post-training quantization (PTQ) using frameworks like PyTorch or TensorFlow Lite to convert FP32 models to INT8. While tools like torchinfo and torch.profiler can measure layer-wise parameter counts and runtime, they do not easily expose a side-by-side hardware efficiency comparison (MACs, memory bytes, arithmetic intensity) between FP32 and INT8 variants at the layer level. Most existing analyses focus on accuracy drop rather than systematically quantifying the hardware workload reduction layer by layer.

---

## Question 3: What is new in your approach and why do you think it will be successful?

This project will perform a structured, layer-by-layer workload accounting of MobileNetV2 in both FP32 and INT8, computing MACs, weight memory, activation memory, and arithmetic intensity for each precision. By applying the same analytical framework introduced in this course (identical to what was done for ResNet-18 in Codefest 1), the analysis will make the hardware efficiency tradeoffs concrete and comparable. This approach is straightforward to implement using PyTorch's built-in quantization API and torchinfo, and directly connects classroom concepts to a real-world deployment scenario. It is likely to succeed because the tools, methods, and model are all well-documented and accessible.
