"""Run isolated GPU exports and references on the preparation machine."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_sample import gpu_name_matches

def select_gpu(devices, expected='RX 7800 XT'):
    matches = [device for device in devices if gpu_name_matches(device['name'], expected)]
    if len(matches) != 1:
        raise ValueError(f'Expected one {expected} device; found {len(matches)} in {devices}')
    return matches[0]

def hardware_inventory(python):
    code = ('import json,torch; print(json.dumps({"hip":torch.version.hip,"devices":'
            '[{"index":i,"name":torch.cuda.get_device_name(i),"vram_bytes":torch.cuda.get_device_properties(i).total_memory}'
            ' for i in range(torch.cuda.device_count())]}))')
    result = subprocess.run([python, '-c', code], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path, help="Existing MongoIA project, for the shared GPU lock")
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--models", nargs="+", choices=("nano", "small"), default=["nano", "small"])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    scripts = Path(__file__).resolve().parent
    args.out_root.mkdir(parents=True, exist_ok=True)
    hardware = hardware_inventory(args.python)
    (args.out_root / 'hardware.json').write_text(json.dumps(hardware, indent=2) + '\n', encoding='utf-8')
    gpu = select_gpu(hardware['devices'])
    hardware['selected'] = gpu
    hardware['expected_gpu'] = 'RX 7800 XT'
    (args.out_root / 'hardware.json').write_text(json.dumps(hardware, indent=2) + '\n', encoding='utf-8')
    sys.path.insert(0, str(args.project_root / "scripts" / "teacher"))
    from _common import gpu_lock
    # The established project lock serializes this job with existing trainers.
    with gpu_lock("snapdragon-published-export-reference"):
        subprocess.run([args.python, str(scripts / "download_models.py"), "--config", str(args.config),
                        "--root", str(args.models_root), "--models", *args.models], check=True)
        for name in args.models:
            spec = cfg["models"][name]
            source = args.models_root / spec["repo_id"].split("/")[-1] / spec["revision"]
            subprocess.run([args.python, str(scripts / "verify_manifest.py"), str(source)], check=True)
            package = args.out_root / (name + "-dml-fp16")
            if (package / "export_manifest.json").is_file():
                subprocess.run([args.python, str(scripts / "verify_manifest.py"), str(package)], check=True)
            else:
                subprocess.run([args.python, str(scripts / "export_gpu.py"), "--source", str(source), "--out", str(package)], check=True)
            # A fresh process releases all model and HIP memory between tiers.
            subprocess.run([args.python, str(scripts / "run_sample.py"), "--model", str(source), "--corpus", str(args.corpus),
                            "--backend", "hf", "--device", f"cuda:{gpu['index']}", "--expected-gpu", "RX 7800 XT",
                            "--out", str(args.out_root / (name + "-baseline.json"))], check=True)
    subprocess.run([args.python, str(scripts / "qnn_probe.py"), "--config", str(args.config),
                    "--out", str(args.out_root / "preparation-inventory.json")], check=True)


if __name__ == "__main__":
    main()
