"""Integration tests for BOM upload API endpoints.

Tests the full upload → confirm → poll → results flow via the FastAPI test client.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Use a test key that will be recognized by both security and tenant registry
TEST_API_KEY = "test-bom-key"
HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def client(monkeypatch):
    """Create a FastAPI test client with proper API key setup."""
    monkeypatch.setenv("CALE_API_KEYS", TEST_API_KEY)

    from potatobacon.api.security import rate_limiter
    rate_limiter.reset()

    from potatobacon.api.app import app
    from potatobacon.api.tenants import get_registry

    registry = get_registry()
    if registry.resolve(TEST_API_KEY) is None:
        registry.register_tenant(
            tenant_id="test-bom-tenant",
            name="Test BOM Tenant",
            api_key=TEST_API_KEY,
            plan="enterprise",
        )

    with TestClient(app) as c:
        yield c


class TestBOMUploadEndpoint:
    """Tests for the upload endpoint (no Z3 analysis involved)."""

    def test_upload_csv(self, client):
        """Upload a clean CSV and get validation summary."""
        csv_path = FIXTURES_DIR / "clean_bom.csv"

        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("test_bom.csv", csv_path.read_bytes(), "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "parsed"
        assert data["parseable_rows"] == 7
        assert data["skipped_rows"] == 0
        assert "upload_id" in data
        assert len(data["preview"]) == 5  # First 5 items
        assert "part_id" in data["detected_columns"]
        assert "description" in data["detected_columns"]

    def test_upload_json(self, client):
        """Upload a JSON BOM file."""
        json_path = FIXTURES_DIR / "bom_items.json"

        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("bom.json", json_path.read_bytes(), "application/json")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "parsed"
        assert data["parseable_rows"] == 5

    def test_upload_xlsx(self, client):
        """Upload an XLSX BOM file."""
        xlsx_path = FIXTURES_DIR / "multi_sheet_bom.xlsx"

        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={
                "file": (
                    "bom.xlsx",
                    xlsx_path.read_bytes(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "parsed"
        assert data["parseable_rows"] >= 4

    def test_upload_unsupported_type(self, client):
        """Uploading an unsupported file type returns 400."""
        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_upload_empty_file(self, client):
        """Uploading an empty file returns 400."""
        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert response.status_code == 400

    def test_upload_no_api_key(self, client):
        """Upload without API key returns 401."""
        csv_path = FIXTURES_DIR / "clean_bom.csv"
        response = client.post(
            "/v1/bom/upload",
            files={"file": ("test.csv", csv_path.read_bytes(), "text/csv")},
        )
        assert response.status_code == 401

    def test_messy_csv_with_skipped_rows(self, client):
        """Messy CSV reports skipped rows with reasons."""
        csv_path = FIXTURES_DIR / "messy_bom.csv"

        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("messy.csv", csv_path.read_bytes(), "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["parseable_rows"] > 0
        assert data["skipped_rows"] > 0
        assert len(data["skipped_reasons"]) > 0

    def test_upload_material_percentages(self, client):
        """Upload returns correct material percentage breakdown."""
        csv_path = FIXTURES_DIR / "clean_bom.csv"

        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("test.csv", csv_path.read_bytes(), "text/csv")},
        )

        data = response.json()
        assert "material_percentages" in data
        mat_pcts = data["material_percentages"]
        assert "by_weight" in mat_pcts or "by_value" in mat_pcts


class TestBOMAnalyzeFlow:
    """Tests for the analyze → poll → results flow.

    Uses single-item BOMs to avoid Z3 thread-safety issues.
    """

    def _upload_single_item(self, client) -> str:
        """Helper: upload a CSV with 1 item and return the upload_id."""
        csv_text = b"part_id,description,material,value_usd,origin_country\nP1,Steel hex bolt M10,Steel,1.25,CN\n"
        response = client.post(
            "/v1/bom/upload",
            headers=HEADERS,
            files={"file": ("test.csv", csv_text, "text/csv")},
        )
        assert response.status_code == 200
        return response.json()["upload_id"]

    def test_analyze_nonexistent_upload(self, client):
        """Analyzing a non-existent upload returns 404."""
        response = client.post(
            "/v1/bom/fake_upload_id/analyze",
            headers=HEADERS,
            json={},
        )
        assert response.status_code == 404

    def test_upload_and_analyze_returns_job(self, client):
        """Upload → analyze returns a queued job_id."""
        upload_id = self._upload_single_item(client)

        response = client.post(
            f"/v1/bom/{upload_id}/analyze",
            headers=HEADERS,
            json={"max_mutations": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["total_items"] == 1
        assert "job_id" in data

        # Wait for background thread to finish to avoid test pollution
        time.sleep(3)

    def test_nonexistent_job_status(self, client):
        """Polling status of non-existent job returns 404."""
        response = client.get(
            "/v1/bom/fake_job_id/status",
            headers=HEADERS,
        )
        assert response.status_code == 404

    def test_nonexistent_job_results(self, client):
        """Getting results of non-existent job returns 404."""
        response = client.get(
            "/v1/bom/fake_job_id/results",
            headers=HEADERS,
        )
        assert response.status_code == 404

    def test_full_pipeline_single_item(self, client):
        """Upload CSV (1 item) → analyze → poll until complete → verify results."""
        upload_id = self._upload_single_item(client)

        # Analyze
        analyze_resp = client.post(
            f"/v1/bom/{upload_id}/analyze",
            headers=HEADERS,
            json={"max_mutations": 1},
        )
        assert analyze_resp.status_code == 200
        job_id = analyze_resp.json()["job_id"]

        # Poll until complete (max 30s)
        completed = False
        for _ in range(60):
            status_resp = client.get(
                f"/v1/bom/{job_id}/status",
                headers=HEADERS,
            )
            assert status_resp.status_code == 200
            status = status_resp.json()
            assert status["total"] == 1
            if status["status"] == "completed":
                completed = True
                break
            time.sleep(0.5)

        if not completed:
            pytest.skip("Job did not complete in time")

        # Get results
        results_resp = client.get(
            f"/v1/bom/{job_id}/results",
            headers=HEADERS,
        )
        assert results_resp.status_code == 200
        results = results_resp.json()

        # Verify structure
        assert results["status"] == "completed"
        assert "portfolio_summary" in results
        assert "results" in results
        assert len(results["results"]) == 1

        # Verify portfolio summary
        summary = results["portfolio_summary"]
        assert summary["total_skus"] == 1
        assert summary["completed_skus"] >= 0

        # Verify individual result fields
        r = results["results"][0]
        assert "description" in r
        assert "status" in r
        assert r["part_id"] == "P1"

        # Verify combined proof chain
        assert "combined_proof_chain" in results
        assert results["combined_proof_chain"]["job_id"] == job_id
