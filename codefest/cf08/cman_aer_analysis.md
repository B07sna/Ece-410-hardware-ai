# CMAN AER Bandwidth Analysis — CodeFest 8

**System Parameters**

| Parameter | Value |
|-----------|-------|
| N | 1024 neurons |
| f | 50 Hz mean firing rate per neuron |
| Packet size | 20 bits (10-bit address + 6-bit timestamp + 4-bit framing) |

---

## Task 1 — Mean Aggregate Spike Rate

**Formula:**

$$R = N \times f$$

**Substitution:**

$$R = 1024 \text{ neurons} \times 50 \text{ Hz} = 51{,}200 \text{ spikes/s}$$

**Result:** R = **51,200 spikes/s**

---

## Task 2 — Mean AER Bandwidth

**Formula:**

$$B = R \times b_{\text{packet}}$$

**Substitution:**

$$B = 51{,}200 \text{ spikes/s} \times 20 \text{ bits/spike} = 1{,}024{,}000 \text{ bit/s}$$

**Convert to Mbit/s:**

$$B = \frac{1{,}024{,}000}{10^6} = 1.024 \text{ Mbit/s}$$

**Result:** B = **1.024 Mbit/s**

---

## Task 3 — Interface Comparison

| Interface | Max Bandwidth | Supports 1.024 Mbit/s? | Notes |
|-----------|--------------|------------------------|-------|
| SPI | 50 Mbit/s | Y | Widely supported; 48.8× headroom over mean |
| I2C (HS mode) | 3.4 Mbit/s | Y | 3.3× headroom; adequate at mean rate |
| AXI4-Lite | 100 Mbit/s | Y | 97.7× headroom; full memory-mapped bus |

**Lowest-complexity interface that suffices:** **I2C (High-Speed mode, 3.4 Mbit/s)**

I2C requires only two wires (SDA, SCL) and no chip-select logic, making it the minimum-complexity option whose bandwidth exceeds the 1.024 Mbit/s mean AER requirement. SPI would be the next step up if burst margin proves insufficient (see Task 4).

---

## Task 4 — Burst Analysis

**Scenario:** 25% of N = 1024 neurons fire within a 1 ms window.

### Peak Spike Count in Burst Window

$$N_{\text{burst}} = 0.25 \times 1024 = 256 \text{ neurons}$$

### Peak Spike Rate

$$R_{\text{peak}} = \frac{N_{\text{burst}}}{\Delta t} = \frac{256 \text{ spikes}}{1 \times 10^{-3} \text{ s}} = 256{,}000 \text{ spikes/s}$$

### Peak AER Bandwidth

$$B_{\text{peak}} = R_{\text{peak}} \times b_{\text{packet}} = 256{,}000 \times 20 = 5{,}120{,}000 \text{ bit/s}$$

$$B_{\text{peak}} = 5.12 \text{ Mbit/s}$$

### Burst-to-Mean Ratio

$$\text{Ratio} = \frac{B_{\text{peak}}}{B_{\text{mean}}} = \frac{5.12 \text{ Mbit/s}}{1.024 \text{ Mbit/s}} = 5\times$$

### Interface Verdict

| Interface | Max BW | Absorbs 5.12 Mbit/s Burst? | Buffering Required |
|-----------|--------|----------------------------|--------------------|
| I2C (HS) | 3.4 Mbit/s | **No** | Yes — burst exceeds capacity |
| SPI | 50 Mbit/s | **Yes** | Minimal |
| AXI4-Lite | 100 Mbit/s | **Yes** | None |

**I2C cannot absorb the burst peak (5.12 Mbit/s > 3.4 Mbit/s).** A FIFO buffer is required.

### FIFO Depth Estimate

Excess data rate during the 1 ms burst window (using I2C at 3.4 Mbit/s):

$$\Delta B = B_{\text{peak}} - B_{\text{I2C}} = 5.12 - 3.4 = 1.72 \text{ Mbit/s}$$

Bits accumulated in the 1 ms burst:

$$\text{Bits buffered} = \Delta B \times \Delta t = 1.72 \times 10^6 \text{ bit/s} \times 1 \times 10^{-3} \text{ s} = 1{,}720 \text{ bits}$$

Minimum FIFO depth (in 20-bit AER packets):

$$\text{FIFO depth} = \left\lceil \frac{1{,}720}{20} \right\rceil = \lceil 86 \rceil = \textbf{86 packets}$$

A safe design uses **128-entry deep FIFO** (next power of two) to absorb the worst-case burst without packet loss on I2C. If I2C complexity is acceptable but the 128-entry FIFO is not, upgrading to **SPI eliminates the buffering requirement entirely**.

---

## Task 5 — Frame-Based Comparison

**Frame scheme:** 1024 neurons sampled every 1 ms, 1 bit per neuron.

### Frame Size and Rate

$$F_{\text{size}} = N \times 1 \text{ bit} = 1024 \text{ bits/frame}$$

$$F_{\text{rate}} = \frac{1}{1 \text{ ms}} = 1000 \text{ frames/s}$$

### Frame Bandwidth

$$B_{\text{frame}} = F_{\text{size}} \times F_{\text{rate}} = 1024 \times 1000 = 1{,}024{,}000 \text{ bit/s}$$

$$\boxed{B_{\text{frame}} = 1.024 \text{ Mbit/s}}$$

### AER-to-Frame Ratio at f = 50 Hz

$$\text{Ratio} = \frac{B_{\text{AER}}}{B_{\text{frame}}} = \frac{1.024 \text{ Mbit/s}}{1.024 \text{ Mbit/s}} = 1.0$$

At f = 50 Hz the two schemes consume **identical bandwidth**.

### Crossover Firing Rate f_crossover

Set AER bandwidth equal to frame bandwidth and solve for f:

$$N \times f_{\text{cross}} \times b_{\text{packet}} = B_{\text{frame}}$$

$$f_{\text{cross}} = \frac{B_{\text{frame}}}{N \times b_{\text{packet}}} = \frac{1{,}024{,}000 \text{ bit/s}}{1024 \times 20 \text{ bit}} = \frac{1{,}024{,}000}{20{,}480}$$

$$\boxed{f_{\text{cross}} = 50 \text{ Hz}}$$

**Implication:** AER is more bandwidth-efficient than dense frame encoding for any mean firing rate below 50 Hz, which makes it the preferred protocol for sparse, low-activity neural populations — but frame encoding becomes competitive or superior once average activity reaches or exceeds 50 spikes/s per neuron.

---

## Summary Table

| Task | Result |
|------|--------|
| Mean aggregate spike rate R | 51,200 spikes/s |
| Mean AER bandwidth B | 1.024 Mbit/s |
| Lowest-complexity sufficient interface | I2C HS (3.4 Mbit/s) |
| Peak burst bandwidth (25% × 1024 in 1 ms) | 5.12 Mbit/s |
| Burst-to-mean ratio | 5× |
| I2C burst verdict | Needs 128-entry FIFO buffer |
| Frame bandwidth (1 bit/neuron, 1 ms period) | 1.024 Mbit/s |
| AER-to-frame ratio at f = 50 Hz | 1.0 |
| Crossover firing rate f_crossover | **50 Hz** |
