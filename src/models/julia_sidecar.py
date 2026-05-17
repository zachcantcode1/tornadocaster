import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class JuliaSidecarRunner:
    """
    Bridges Python orchestration to native Julia model inference.
    """

    def __init__(self, script_path: str = "scripts/julia_predict_proxy.jl"):
        self.script_path = script_path

    @staticmethod
    def is_available() -> bool:
        return shutil.which("julia") is not None

    def predict(self, model_path: str, features: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("Julia runtime is not installed or not on PATH.")

        script = Path(self.script_path)
        if not script.exists():
            raise FileNotFoundError(f"Julia sidecar script not found: {script}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            req = tmp / "request.json"
            out = tmp / "predictions.json"

            req.write_text(
                json.dumps(
                    {
                        "model_path": str(Path(model_path).resolve()),
                        "features": features.tolist(),
                        "output_path": str(out),
                    }
                )
            )

            proc = subprocess.run(
                ["julia", str(script.resolve()), str(req.resolve())],
                capture_output=True,
                text=True,
                env={**dict(__import__("os").environ), "JULIA_NUM_THREADS": "1"},
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Julia sidecar inference failed.\n"
                    f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
                )
            if not out.exists():
                raise RuntimeError("Julia sidecar did not write predictions output file.")
            payload = json.loads(out.read_text())
            preds = np.asarray(payload.get("predictions", []), dtype=np.float32)
            if preds.ndim != 1:
                raise RuntimeError("Julia sidecar returned invalid prediction shape.")
            mode = payload.get("mode", "")
            if mode != "native_mctb":
                err = payload.get("error", "")
                raise RuntimeError(
                    "Julia sidecar did not execute native MemoryConstrainedTreeBoosting inference. "
                    f"mode={mode} error={err}"
                )
            return preds
