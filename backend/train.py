"""Regenerate the listener segmentation model from the survey CSV.

Run from the backend/ directory (needs requirements-train.txt):

    python train.py

Writes models/segments.json: KMeans centroids, the feature names they were
trained on, and a persona per segment. It's plain JSON (no pickle), so the API
loads it without scikit-learn and without executing code from the file.
Training is deterministic: the same data produces the same artifact.
"""

import json
import os

from app.ml.segmenter import ARTIFACT_NAME
from app.ml.training import train

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BACKEND_DIR, "..", "data", "Spotify_user_research.csv")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")


def main() -> None:
    artifact = train(DATA_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, ARTIFACT_NAME)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")

    metrics = artifact["metrics"]
    print(f"Trained on {artifact['training']['rows']} survey rows → {path}")
    print("silhouette by k: " + ", ".join(f"k={k}: {s}" for k, s in metrics["silhouette_by_k"].items()))
    print(f"chosen k={metrics['k']} (smallest segment {metrics['min_segment_share']:.1%})")
    for p in artifact["personas"]:
        print(f"  [{p['id']}] {p['name']} ({p['share']:.0%}) — {p['description']}")


if __name__ == "__main__":
    main()
