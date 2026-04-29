#!/bin/bash
# Setup development environment for Constellation

set -e

echo "🌌 Constellation - Environment Setup"
echo "====================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "❌ Python 3.11+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version: $PYTHON_VERSION"

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    echo "✅ Node.js version: $NODE_VERSION"
else
    echo "❌ Node.js not found. Install from https://nodejs.org/"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    echo "✅ Docker version: $DOCKER_VERSION"
else
    echo "❌ Docker not found. Install Docker Desktop."
    exit 1
fi

echo ""
echo "📦 Installing backend dependencies..."
cd backend
pip install -e ".[dev,ml]" --quiet
cd ..

echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent
cd ..

echo ""
echo "🔑 Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
    echo "   → Edit .env to add your WANDB_API_KEY"
else
    echo "✅ .env already exists"
fi

echo ""
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis

echo ""
echo "⏳ Waiting for PostgreSQL..."
sleep 5

echo ""
echo "🗄️  Running database migrations..."
cd backend
alembic upgrade head 2>/dev/null || echo "   (No migrations to run yet)"
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start development:"
echo "  Backend:  cd backend && uvicorn app.main:app --reload"
echo "  Frontend: cd frontend && npm run dev"
echo "  Or:       docker-compose up"
echo ""
echo "Endpoints:"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Frontend: http://localhost:5173"
