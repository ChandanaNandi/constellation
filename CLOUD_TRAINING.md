# Cloud GPU Training Guide

## Quick Start (RunPod RTX 4090)

### 1. Launch Instance
- Go to [runpod.io](https://runpod.io)
- Select **RTX 4090** ($0.34/hr)
- Choose **PyTorch 2.x** template
- Add **50GB+ persistent storage** for dataset

### 2. Upload Dataset
From your M4 Pro, upload BDD100K to RunPod:

```bash
# Option A: Use runpodctl (recommended)
runpodctl send data/bdd100k_yolo

# Option B: Use scp
scp -r data/bdd100k_yolo root@<pod-ip>:/workspace/
```

### 3. Clone & Setup

```bash
# SSH into pod
ssh root@<pod-ip>

# Clone repo
cd /workspace
git clone https://github.com/ChandanaNandi/constellation.git
cd constellation

# Install dependencies
pip install -r requirements.txt

# Move dataset (if uploaded separately)
mv /workspace/bdd100k_yolo data/
```

### 4. Run Training

```bash
# Full training (recommended)
python train.py \
    --epochs 50 \
    --batch-size 32 \
    --device cuda \
    --data-dir data/bdd100k_yolo

# With W&B logging (optional)
wandb login
python train.py \
    --epochs 50 \
    --batch-size 32 \
    --device cuda \
    --use-wandb
```

### 5. Monitor Progress

Training will print:
```
Epoch 10/50: loss=1.12, cls=0.18, bbox=0.42, mAP@50=0.152
```

Expected timeline on RTX 4090:
- ~5 min per epoch
- ~4-5 hours total for 50 epochs
- **Total cost: ~$2-3**

### 6. Download Results

```bash
# From your M4 Pro
scp root@<pod-ip>:/workspace/constellation/checkpoints/best.pt ./checkpoints/
```

---

## Expected Results

| Epochs | mAP@50 | Notes |
|--------|--------|-------|
| 5 | ~4% | What you have now (1K images) |
| 10 | ~8-12% | Early training |
| 25 | ~15-20% | Decent baseline |
| 50 | ~20-28% | Full training |

---

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python train.py --batch-size 16  # or 8
```

### Slow Data Loading
Increase workers:
```bash
python train.py --num-workers 8
```

### W&B Issues
Skip W&B logging:
```bash
python train.py  # runs without --use-wandb
```

---

## Cost Estimate

| Provider | GPU | $/hr | 50 epochs | Total |
|----------|-----|------|-----------|-------|
| RunPod | RTX 4090 | $0.34 | ~5 hrs | **$1.70** |
| RunPod | A40 | $0.79 | ~4 hrs | **$3.16** |
| Lambda | A10 | $1.10 | ~6 hrs | **$6.60** |

**Budget $10-15 for training + debugging.**

---

## After Training

1. Download `checkpoints/best.pt`
2. Run inference locally:
   ```bash
   python inference.py --checkpoint checkpoints/best.pt --num-images 20
   ```
3. Check `output/visualizations/` for results
4. Note your mAP@50 score for resume/portfolio
