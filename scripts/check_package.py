from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED = [
    "README.md",
    "detail.md",
    ".gitignore",
    "configs/paths.example.env",
    "configs/ratio010_mainline.json",
    "configs/stage1_ratio010.yaml",
    "experiments/ablation_monuseg_r0_raw_fixedop/train_r0.py",
    "experiments/stage1_uniform_p084_protocol/select_stage1_protocol.py",
    "experiments/vit_region_prior_sam/03_generate_sam_point_regions.py",
    "experiments/vit_region_prior_sam/05_convert_sam_region_to_hover_targets.py",
    "experiments/monuseg_official_hovernet/generate_weak_hovernet_patches_from_manifest.py",
    "experiments/official_hovernet/train_official_hovernet_patch_fullsup.py",
    "experiments/official_hovernet/hovernet_evaluation_common.py",
    "tools/instance_pruning/tune_pruning_on_val.py",
    "third_party/hover_net/models/hovernet/net_desc.py",
    "scripts/01_stage1_train_select.sh",
    "scripts/01_generate_selected_targets.sh",
    "scripts/02_train_hovernet.sh",
    "scripts/03_calibrate_val.sh",
    "scripts/prepare_sparse_points.py",
    "scripts/read_experiment_config.py",
    "scripts/run_end_to_end.sh",
    "scripts/run_full_pipeline_smoke.sh",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    syntax_errors = []
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")

    forbidden_dirs = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_dir()
        and path.name in {
            "__pycache__",
            ".git",
            ".idea",
            ".vscode",
            ".vs",
            ".pytest_cache",
            "debug",
            "logs",
            "runs",
        }
    ]
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".pyc", ".pyo", ".log", ".pt", ".pth", ".tar", ".iml"}
            or path.name in {"nohup.out", "debug.log"}
            or path.name.startswith("paths.local")
            or path.name.startswith("copilot.data.migration")
        )
    ]
    private_path_markers = (
        "/home/administrator",
        "/mnt/g/",
        "/root/autodl-tmp",
        "C:\\Users\\Administrator",
        "D:\\",
        "G:\\",
    )
    private_path_hits = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".py", ".sh", ".json", ".yaml", ".yml", ".md", ".env", ".txt", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in private_path_markers:
            if marker in text:
                private_path_hits.append(f"{path.relative_to(ROOT).as_posix()}: {marker}")
    mainline = json.loads((ROOT / "configs/ratio010_mainline.json").read_text(encoding="utf-8"))
    checks = {
        "required_files_present": not missing,
        "python_syntax": not syntax_errors,
        "no_cache_or_nested_git_dirs": not forbidden_dirs,
        "no_outputs_or_binary_weights": not forbidden_files,
        "no_private_absolute_paths": not private_path_hits,
        "ratio_is_0.10": mainline["sparse_points"]["ratio"] == 0.1,
        "stage2_is_sam_only_plus_cluster_fallback": (
            mainline["stage2"]["candidate_selection_mode"] == "sam_only"
            and mainline["stage2"]["fallback"] == "cluster"
        ),
        "train_gt_disabled": mainline["stage1"]["curriculum_use_gt_points"] is False,
        "all_experiment_seeds_are_configured": all(
            isinstance(value, int) and value >= 0
            for value in (
                mainline["sparse_points"]["annotation_seed"],
                mainline["stage1"]["seed"],
                mainline["hovernet"]["seed"],
            )
        ),
    }
    report = {
        "checks": checks,
        "missing": missing,
        "syntax_errors": syntax_errors,
        "forbidden_directories": forbidden_dirs,
        "forbidden_files": forbidden_files,
        "private_absolute_paths": private_path_hits,
        "python_file_count": len(python_files),
        "mainline_config_sha256": sha256(ROOT / "configs/ratio010_mainline.json"),
    }
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
