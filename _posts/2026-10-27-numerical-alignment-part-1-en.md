---
layout: post
title: "Numerical Alignment in LLM Acceleration"
subtitle: "Part 1: Where Errors Come From"
date: 2026-10-27 09:00:00 +0800
lang: en
translation_key: numerical-alignment-part-1
permalink: /blog/numerical-alignment-part-1/en/
author: Di Mou
series: "LLM Infra · Engineering Notes"
description: "Why identical inputs can produce different results across GPU kernel implementations, explained through BF16 rounding and a sum you can work out by hand."
---

**Numerical differences are a recurring problem in LLM infrastructure.** Switch a kernel backend or move the same computation to another GPU, and the output can change—even when the inputs and mathematical formula stay the same.

These notes draw on my experience optimizing inference. This first post starts with rounding error: where it comes from, and why a different implementation can produce a different answer. Comments and discussion are welcome.

## 1. What kind of difference are we looking at?

There are three useful places to look: **implementation bugs**, such as an incorrect index or a missing term in a sum; **different precision formats**, such as FP32, FP16, and FP8, which can approximate the same input differently and introduce quantization error, while also changing the precision of subsequent arithmetic; and **different implementations using the same formats**, where changes to tiling or summation order can affect the result even when input, output, and accumulation formats match. **This post focuses on the third case.** To understand it, we first need to see how even a single addition can introduce rounding error within one fixed format.

## 2. Computers cannot represent every number exactly

A floating-point format stores a number using a finite number of bits. Some values are exactly representable; others must be approximated. The same limitation applies to intermediate results.

For example, take the decimal number **1.001** and choose the nearest representable value in each format:

| Format | Value actually stored |
|---|---:|
| FP32 | Approximately 1.0010000467300415 |
| BF16 | 1 |

Neither format stores 1.001 exactly. FP32 simply gets much closer.

Think of a floating-point format as a ruler with a limited set of tick marks. A number that falls between two ticks has to be mapped to a representable value according to a rule. That step is **rounding**. The example above uses round to nearest.

Four common rounding modes are listed below. The examples round to integers to make the direction easy to see; actual floating-point operations round to values representable in the target format.

| Mode | Rule | 1.6 | −1.6 |
|---|---|---:|---:|
| `rn` | Round to nearest; ties to even | 2 | −2 |
| `rz` | Round toward zero | 1 | −1 |
| `ru` | Round toward positive infinity | 2 | −1 |
| `rd` | Round toward negative infinity | 1 | −2 |

**Basic CUDA floating-point arithmetic on NVIDIA GPUs defaults to `rn`: round to nearest, ties to even.** Other modes can be selected with the corresponding instructions. Here, “even” means that the last retained bit of the binary significand is 0.

The difference between the rounded value and the exact value is **rounding error**. Its magnitude is the absolute rounding error:

```text
Absolute rounding error = |rounded value − exact value|
```

Storing 1.001 as BF16 gives 1, so the absolute error is `|1 − 1.001| = 0.001`. We already have an error before any matrix multiplication begins.

Within `[1, 2)`, the spacing between adjacent values is `0.0078125` for BF16, about `0.0000001192` for FP32, and `0.125` for FP8 E4M3FN. Different formats retain different amounts of detail.

One detail is easy to miss: **floating-point spacing changes with magnitude.** In `[256, 512)`, BF16's spacing is 2. Floating-point formats do not keep a fixed number of decimal places; FP32 follows the same principle.

Comparing formats makes rounding easier to see. Now let us keep the format fixed and look at the arithmetic itself.

## 3. Errors introduced during computation

**For this example, every addition is rounded to BF16 using round to nearest, ties to even.** Consider `256 + 1 = 257`. Both inputs are exactly representable in BF16, but 257 is not. Its binary representation is `1.00000001₂ × 2⁸`, which needs **9 significant binary bits**. BF16 has **8**, including the implicit leading 1.

The two neighboring BF16 values are:

- `256 = 1.0000000₂ × 2⁸`: the last retained bit is **0**.
- `258 = 1.0000001₂ × 2⁸`: the last retained bit is **1**.

257 lies exactly halfway between them. Ties to even selects **256**, whose last retained bit is 0. This is about the retained binary significand, not whether 256 and 258 are even integers in decimal.

Under these conditions, **`256 + 1` produces 256, with an absolute error of 1.**

Add one more 1, and the effect of evaluation order becomes visible. All inputs are still exactly representable, and the exact answer is `256 + 1 + 1 = 258`. We keep the same rule: **round to BF16 after each addition, then use that rounded value in the next step.**

| Grouping | First addition | Second addition | Final result |
|---|---|---|---:|
| `(256 + 1) + 1` | 256 + 1 = 257 → rounds to 256 | 256 + 1 = 257 → rounds to 256 again | 256 |
| `256 + (1 + 1)` | 1 + 1 = 2, represented exactly | 256 + 2 = 258, represented exactly | 258 |

In the first grouping, both additions produce the unrepresentable value 257. Each increment of 1 is lost to rounding, leaving a final absolute error of 2. In the second grouping, the two 1s are added first. Both steps are exact, so the final error is 0.

**The inputs, precision format, and rounding rule are unchanged. Only the grouping differs.** Grouping changes the intermediate values, and those values can round differently. Floating-point addition is therefore not associative.

The BF16 rounding after every addition is essential to this example. With FP32 accumulation, **both groupings give exactly 258 in this case**, and converting that result to BF16 leaves it unchanged. FP32 still has rounding error in general; these particular additions happen to be exact. Matching input dtypes is therefore only part of the picture: accumulation precision and evaluation order also matter. GPU kernels can produce different intermediate values by changing their tiling, partial sums, and merge order.

NVIDIA's [*Floating Point and IEEE 754 Compliance for NVIDIA GPUs*](https://docs.nvidia.com/cuda/floating-point/index.html#example-algorithms) compares three implementations of the same dot product that can produce different results. Each computes:

```text
a₁b₁ + a₂b₂ + a₃b₃ + a₄b₄
```

The diagrams below are NVIDIA's Figures 2, 3, and 4 from that document.

**Serial implementation.** Multiplications and additions are separate operations. Products are accumulated along a single chain, with rounding at each operation.

![NVIDIA Figure 2: serial dot-product implementation](/assets/img/blog/numerical-alignment-part-1/nv_serial_method.png)

**Fused multiply-add (FMA).** Each operation adds a product to the running sum with only one rounding for that fused multiply-add. The complete dot product still involves multiple rounding steps.

![NVIDIA Figure 3: dot product using fused multiply-add](/assets/img/blog/numerical-alignment-part-1/nv_fma_method.png)

**Tree-based summation.** Sum the first two terms and the last two terms separately, then combine the partial sums. The two branches can run in parallel, and the grouping of additions changes.

![NVIDIA Figure 4: tree-based dot-product implementation](/assets/img/blog/numerical-alignment-part-1/nv_parallel_method.png)

All three implementations use IEEE 754-compliant operations, yet their final results can differ. Whether multiplication and addition are fused, how additions are grouped, and where rounding occurs all affect the answer.

## References

1. Nathan Whitehead and Alex Fit-Florea. [*Floating Point and IEEE 754 Compliance for NVIDIA GPUs*](https://docs.nvidia.com/cuda/floating-point/index.html). NVIDIA.
2. Paulius Micikevicius et al. [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433v2). 2022.
3. PyTorch. [*Numerical accuracy*](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).
4. David Goldberg. [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). 1991.
