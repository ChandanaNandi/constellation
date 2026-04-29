# 🌌 Constellation

> Multi-task vision system for autonomous driving scenes. Built with PyTorch, inspired by Tesla's HydraNet architecture.

---

## Overview

<!-- TODO: Add project description after Phase 1 -->

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
git clone https://github.com/YOUR_USERNAME/constellation.git
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

- [ ] **Phase 1:** Foundation & Data Engine
- [ ] **Phase 2:** Multi-Task Model Architecture
- [ ] **Phase 3:** Training & Cloud GPU
- [ ] **Phase 4:** Shadow Mode + Quantization
- [ ] **Phase 5:** Polish & Deploy

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by **Chandana Reddy**
