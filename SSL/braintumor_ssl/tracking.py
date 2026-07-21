"""Optional experiment tracking (MLflow → DagsHub) for metrics + figure artifacts.

Turned on with ``mlflow: true`` in the config. The tracking server and credentials come from the
ENVIRONMENT, never the repo, so no secret is ever committed:

    export MLFLOW_TRACKING_USERNAME=<dagshub-username>
    export MLFLOW_TRACKING_PASSWORD=<dagshub-token>          # a DagsHub access token

These are also auto-loaded from a gitignored ``.env`` at the repo root (``_load_dotenv``), so running
from ``SSL/`` or the repo root picks them up without a manual ``source``.
This repository's DagsHub tracking URI is the default. ``MLFLOW_TRACKING_URI`` can override it
for forks or local tests. When ``mlflow`` is false, MLflow isn't installed, or anything goes
wrong, every method below is a no-op — a logging error must never abort an expensive training run.
"""
from __future__ import annotations

import math
import os


DEFAULT_TRACKING_URI = "https://dagshub.com/alilivanturk/IDH-Classifier.mlflow"


def _load_dotenv() -> None:
    """Populate os.environ from the nearest .env (CWD then parents), WITHOUT overwriting vars that
    are already set. Lets DagsHub credentials in a gitignored .env be picked up automatically, so
    forgetting to `source` it just works. No dependency, best-effort, never raises. Values in .env
    are never printed."""
    d = os.getcwd()
    for _ in range(5):                       # search CWD and up to 4 parent dirs
        path = os.path.join(d, ".env")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        if line.startswith("export "):
                            line = line[len("export "):]
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:   # never clobber an explicit env var
                            os.environ[key] = val
            except Exception:
                pass
            return
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


class Tracker:
    """Thin, fail-safe MLflow wrapper. All methods no-op unless tracking is on and healthy."""

    def __init__(self, cfg: dict, run_name: str | None = None, tags: dict | None = None):
        self.on = bool(cfg.get("mlflow"))
        self._mlflow = None
        if not self.on:
            return
        try:
            import mlflow
        except Exception as e:
            print(f"[mlflow] disabled (import failed: {e})")
            self.on = False
            return
        _load_dotenv()                       # pick up DagsHub creds from a gitignored .env if present
        # Environment wins so a fork/test can override the repository default for one command.
        uri = (os.environ.get("MLFLOW_TRACKING_URI")
               or cfg.get("mlflow_tracking_uri")
               or DEFAULT_TRACKING_URI)
        experiment = (os.environ.get("MLFLOW_EXPERIMENT_NAME")
                      or cfg.get("mlflow_experiment", "simsiam-brats"))
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            system_metrics = bool(
                cfg.get("mlflow_system_metrics", False)
                or os.environ.get("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "").lower()
                in {"1", "true", "yes"}
            )
            try:
                mlflow.start_run(run_name=run_name, log_system_metrics=system_metrics)
            except Exception as e:
                if not system_metrics:
                    raise
                # Missing psutil/pynvml must not suppress the model metrics and artifacts.
                if mlflow.active_run():
                    mlflow.end_run(status="KILLED")
                print(f"[mlflow] system metrics disabled ({e})")
                mlflow.start_run(run_name=run_name, log_system_metrics=False)
            if tags:
                mlflow.set_tags(tags)
            self._mlflow = mlflow
            self.run_id = mlflow.active_run().info.run_id
            print(f"[mlflow] tracking -> {mlflow.get_tracking_uri()} "
                  f"(experiment={experiment}, run={run_name}, run_id={self.run_id})")
        except Exception as e:
            print(f"[mlflow] disabled (start failed: {e})")
            self.on = False
            self._mlflow = None

    def log_params(self, params: dict) -> None:
        if not self._mlflow:
            return
        try:                                     # params must be scalar-ish; stringify the rest
            clean = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                     for k, v in params.items() if v is not None}
            self._mlflow.log_params(clean)
        except Exception as e:
            print(f"[mlflow] log_params skipped: {e}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self._mlflow:
            return
        try:                                     # numeric + finite only (drop NaN/inf/None)
            clean = {k: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float)) and math.isfinite(float(v))}
            if clean:
                self._mlflow.log_metrics(clean, step=step)
        except Exception as e:
            print(f"[mlflow] log_metrics skipped: {e}")

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if not self._mlflow or not path or not os.path.exists(path):
            return
        try:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as e:
            print(f"[mlflow] log_artifact skipped ({path}): {e}")

    def log_dict(self, data: dict, artifact_file: str) -> None:
        """Log structured metadata (for example the resolved training config) as JSON."""
        if not self._mlflow:
            return
        try:
            self._mlflow.log_dict(data, artifact_file)
        except Exception as e:
            print(f"[mlflow] log_dict skipped ({artifact_file}): {e}")

    def close(self) -> None:
        if not self._mlflow:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            pass
