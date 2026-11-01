"""Read-only checks for running the current MetaDrive experiment on another GPU host."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    import torch
    import metadrive
    from metadrive.envs.metadrive_env import MetaDriveEnv  # Initialize vehicle imports first.
    from metadrive.component.vehicle.vehicle_type import vehicle_type
    from metadrive.component.pgblock.t_intersection import TInterSection

    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchrl", "tensordict", "numpy", "gymnasium", "hydra-core",
                 "metadrive-simulator", "panda3d", "tensorboard")}
    from metadrive.envs import metadrive_env
    package = Path(metadrive_env.__file__).resolve().parents[1]
    assets = package / "assets"
    result = {"python": platform.python_version(), "versions": versions,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "metadrive_module": str(package / "__init__.py"),
              "metadrive_assets_exist": assets.is_dir(), "t_intersection_id": TInterSection.ID,
              "vehicle_types": sorted(vehicle_type), "gpus": []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            result["gpus"].append({"visible_index": index, "name": device.name,
                                   "memory_gib": round(device.total_memory / 2**30, 2)})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append("Use Python3.11, matching pyproject.toml.")
    if not result["cuda_available"] and not args.allow_cpu:
        errors.append("CUDA is unavailable; check the installed PyTorch build and NVIDIA driver.")
    if not assets.is_dir():
        errors.append("MetaDrive assets are missing; install or transfer the matching assets before training.")
    if TInterSection.ID != "T":
        errors.append("Unexpected MetaDrive T-intersection implementation.")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
