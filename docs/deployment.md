# Deployment Guide

## Quantization Pipeline

1. PyTorch → ONNX export
2. ONNX → INT8 quantization
3. Benchmark: FP32 vs FP16 vs INT8

## Benchmarks

<!-- TODO: Add after Phase 4 -->

| Format | Latency (ms) | mAP@0.5 | Size (MB) |
|--------|--------------|---------|-----------|
| PyTorch FP32 | TBD | TBD | TBD |
| ONNX FP32 | TBD | TBD | TBD |
| ONNX FP16 | TBD | TBD | TBD |
| ONNX INT8 | TBD | TBD | TBD |

## Production Deployment

- **Platform:** Railway
- **Stack:** Docker + FastAPI + React

---

*TODO: Add deployment instructions*
