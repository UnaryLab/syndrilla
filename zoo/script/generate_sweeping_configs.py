"""Write the point folders of zoo/script/sweeping_configs.yaml; run from the repo root.

Calls syndrilla.sweep.generate. Without -r, the points of each decoder go to
zoo/<decoder>_sweeping/; with -r, all points go to zoo/<run_dir>/.
"""

import argparse
import os

from syndrilla.sweep import generate
from syndrilla.utils import read_yaml


def main():
    parser = argparse.ArgumentParser(description="Generate sweeping configurations.")
    parser.add_argument(
        "-r",
        "--run_dir",
        type=str,
        default=None,
        help="The run directory to store outputs. This should be a sub directory under zoo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite point folders that already hold a result yaml.",
    )
    args = parser.parse_args()

    config = read_yaml("zoo/script/sweeping_configs.yaml")
    if args.run_dir is not None:
        generate(config, os.path.join("zoo/", args.run_dir), args.force)
        return
    for decoder in config["decoder"]:
        generate(
            {**config, "decoder": [decoder]}, f"zoo/{decoder}_sweeping", args.force
        )


if __name__ == "__main__":
    main()
