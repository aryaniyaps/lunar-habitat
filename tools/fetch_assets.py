"""Download NVIDIA SimReady robot packages from huggingface.co/datasets/nvidia/isaacsim-assets."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "nvidia/isaacsim-assets"
ROOT = Path(__file__).resolve().parents[1] / "assets" / "nvidia"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="+", help="package folder, e.g. frankarobotics_frankapanda_franka")
    args = ap.parse_args()
    for pkg in args.packages:
        path = snapshot_download(
            REPO,
            repo_type="dataset",
            allow_patterns=[f"{pkg}/*"],
            ignore_patterns=["*.mdl", "*/.thumbs/*", "*_flex.usd", "*Logo_Textures*"],
            local_dir=ROOT,
        )
        print(f"{pkg} -> {Path(path) / pkg}")


if __name__ == "__main__":
    main()
