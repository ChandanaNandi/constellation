"""Tests for health check endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test that health endpoint returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "Constellation"
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test that root endpoint returns welcome message."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "docs" in data
