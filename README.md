# 🌌 Constellation

> Multi-task vision system for autonomous driving scenes. Built with PyTorch, inspired by Tesla's HydraNet architecture.

---

## Overview

Constellation is an end-to-end autonomous driving perception system that demonstrates:

- **Auto-labeling Pipeline**: YOLOv8 object detection + MobileSAM segmentation
- **Multi-task Learning**: Shared backbone with task-specific heads (detection, lanes, depth)
- **Shadow Mode**: Compare model predictions against ground truth for validation
- **Data Engine**: Active learning loop to identify and fix hard cases

**Tech Stack**: Python, PyTorch, FastAPI, React, PostgreSQL, Docker

---

## Architecture

<!-- TODO: Add architecture diagram and description -->

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 20+ (for frontend)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ChandanaNandi/constellation.git
cd constellation

# Copy environment file
cp .env.example .env

# Start services
docker-compose up

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

### Local Development (without Docker)

```bash
# Backend
cd backend
pip install -e ".[dev,ml]"
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## Roadmap

- [x] **Phase 1:** Foundation & Data Engine
- [ ] **Phase 2:** Multi-Task Model Architecture
- [ ] **Phase 3:** Training & Cloud GPU
- [ ] **Phase 4:** Shadow Mode + Quantization
- [ ] **Phase 5:** Polish & Deploy

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by **Chandana Reddy**
