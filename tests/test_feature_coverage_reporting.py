import asyncio
import json
import tempfile

from main import run_pipeline


def test_feature_coverage_summary_and_report_file():
    with tempfile.TemporaryDirectory() as tmp:
        report_path = f"{tmp}/coverage.json"
        result = asyncio.run(
            run_pipeline(
                mock_mode=True,
                contract_path="artifacts/upstream/model_contract.json",
                event="tornado",
                window="f13-24",
                inference_backend="auto",
                coverage_report_path=report_path,
            )
        )
        cov = result["feature_coverage"]
        assert cov["total_features"] == 2005
        assert cov["implemented_features"] <= cov["total_features"]
        assert cov["missing_features"] >= 0

        payload = json.loads(open(report_path).read())
        assert payload["event"] == "tornado"
        assert payload["window"] == "f13-24"
        assert "coverage" in payload
