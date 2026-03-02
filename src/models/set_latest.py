from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _parse_run_id_from_model_name(name: str) -> str:
    # model_logreg_<run_id>.joblib
    stem = Path(name).stem
    if stem.startswith("model_logreg_"):
        return stem.replace("model_logreg_", "")
    raise ValueError("Não foi possível inferir run_id do nome do modelo.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/models"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", type=str)
    g.add_argument("--model-path", type=str)
    args = p.parse_args()

    artifacts_dir: Path = args.artifacts_dir

    if args.run_id:
        run_id = args.run_id
        model_path = artifacts_dir / f"model_logreg_{run_id}.joblib"
    else:
        model_path = Path(args.model_path)
        run_id = _parse_run_id_from_model_name(model_path.name)

    metrics_path = artifacts_dir / f"metrics_{run_id}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics não encontrado: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(metrics["threshold"])

    latest = {
        "run_id": run_id,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "threshold": threshold,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (artifacts_dir / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(artifacts_dir / "latest.json"))


if __name__ == "__main__":
    main()