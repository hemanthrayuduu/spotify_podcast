"""Regenerate the ML model artifacts from the survey CSV.

Run from the backend/ directory:

    python train.py

This trains the KMeans segmentation model on data/Spotify_user_research.csv
and writes the artifacts consumed at serving time into backend/models/.

Security note: the resulting .pkl files are loaded via pickle at startup, which
executes arbitrary code in the file. Only ever load artifacts produced by this
script from trusted data — never load a .pkl from an untrusted source.
"""

import os

from app.ml.analyzer import SpotifyUserAnalyzer

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BACKEND_DIR, "..", "data", "Spotify_user_research.csv")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")


def main() -> None:
    analyzer = SpotifyUserAnalyzer(data_path=DATA_PATH, model_dir=MODEL_DIR)
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.train_cluster_model(n_clusters=3)
    analyzer.create_segment_profiles()
    analyzer.save_models()
    print(f"Model artifacts written to {MODEL_DIR}")


if __name__ == "__main__":
    main()
