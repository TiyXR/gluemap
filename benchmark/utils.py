"""Utility functions for benchmark configuration and dataset discovery."""

import glob
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy

import yaml


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file with support for inheritance via _base_.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "_base_" in config:
        base_path = config_path.parent / config["_base_"]
        base_config = load_config(str(base_path))
        del config["_base_"]
        config = deep_merge(base_config, config)

    return config


def discover_datasets(
    images_pattern: str,
    gt_pattern: Optional[str] = None,
    filter_datasets: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Discover datasets matching the given patterns.

    Args:
        images_pattern: Pattern for images path with {dataset} placeholder
            e.g., "datasets/ETH3D/train/{dataset}/images"
        gt_pattern: Pattern for ground truth path with {dataset} placeholder
            e.g., "datasets/ETH3D/train/{dataset}/colmap"
        filter_datasets: Optional list of dataset names to filter to

    Returns:
        List of tuples (dataset_name, images_path, gt_path)
    """
    # Convert pattern to glob pattern
    glob_pattern = images_pattern.replace("{dataset}", "*")
    matches = sorted(glob.glob(glob_pattern))

    # Extract dataset names from matches
    # Find the position of {dataset} in the pattern
    pattern_parts = images_pattern.split("{dataset}")
    if len(pattern_parts) != 2:
        raise ValueError(
            f"Pattern must contain exactly one {{dataset}} placeholder: {images_pattern}"
        )

    prefix, suffix = pattern_parts
    prefix_len = len(prefix)
    suffix_len = len(suffix)

    datasets = []
    for match in matches:
        # Extract dataset name from the matched path
        if suffix_len > 0:
            dataset_name = match[prefix_len:-suffix_len]
        else:
            dataset_name = match[prefix_len:]

        # Skip if filtering and not in filter list
        if filter_datasets is not None and dataset_name not in filter_datasets:
            continue

        images_path = match
        gt_path = gt_pattern.replace("{dataset}", dataset_name) if gt_pattern else None

        # Verify paths exist
        if not os.path.isdir(images_path):
            print(f"Warning: Images path does not exist: {images_path}")
            continue

        datasets.append((dataset_name, images_path, gt_path))

    return datasets


def print_summary(
    results: Dict[str, Dict[str, Any]],
    failures: Dict[str, str],
) -> None:
    """Print a summary of benchmark results.

    Args:
        results: Dictionary mapping dataset names to metric dictionaries
        failures: Dictionary mapping dataset names to error messages
    """
    total = len(results) + len(failures)
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {total}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failures)}")
    print()

    if results:
        print("Successful datasets:")
        for dataset_name, metrics in sorted(results.items()):
            metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"  {dataset_name}: {metrics_str}")

    if failures:
        print("\nFailed datasets:")
        for dataset_name, error in sorted(failures.items()):
            print(f"  {dataset_name}: {error}")

    print("=" * 60)


def load_experiments(
    experiment_files: List[str],
    gt_pattern: str,
    output_pattern: str,
    preprocess_pattern: Optional[str] = None,
) -> List[Tuple[str, str, str, str, List[str], Optional[str]]]:
    """Load experiments from text files.

    Each line in the experiment file is CSV format:
        benchmark_name,images_base_path,scene_name,experiment_name,image1,image2,...

    Args:
        experiment_files: List of paths to experiment definition files
        gt_pattern: Pattern for ground truth path with {scene} placeholder
        output_pattern: Pattern for output path with {scene}/{exp_name} placeholders
        preprocess_pattern: Optional pattern for scene-level preprocessing path
            with {scene} placeholder. SALAD features will be shared across
            experiments in the same scene.

    Returns:
        List of tuples (experiment_name, images_path, gt_path, output_path, image_list, preprocess_path)
    """
    experiments = []

    for file_path in experiment_files:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 5:
                    print(f"Warning: Skipping invalid line (need at least 5 fields): {line}")
                    continue

                # Parse fields: benchmark_name, images_path, scene_name, experiment_name, images...
                benchmark_name = parts[0]
                images_path = parts[1]
                scene_name = parts[2]
                exp_name = parts[3]
                image_list = [x for x in parts[4:] if x]  # Filter out empty fields
                image_list = None if not image_list else image_list  # Set to None if empty

                # Construct experiment name with scene/exp_name structure
                experiment_name = f"{scene_name}/{exp_name}"

                # Substitute patterns ({benchmark} uses field 0, e.g. "CAB", "HGE")
                gt_path = gt_pattern.replace("{scene}", scene_name).replace("{benchmark}", benchmark_name)
                output_path_exp = output_pattern.replace("{scene}", scene_name).replace("{exp_name}", exp_name).replace("{benchmark}", benchmark_name)

                # Scene-level preprocess path for SALAD feature reuse
                preprocess_path = None
                if preprocess_pattern:
                    preprocess_path = preprocess_pattern.replace("{scene}", scene_name).replace("{benchmark}", benchmark_name)

                experiments.append(
                    (experiment_name, images_path, gt_path, output_path_exp, image_list, preprocess_path)
                )

    return experiments


def load_datasets_from_config(
    config: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[List[Tuple[str, str, str, str, Optional[List[str]], Optional[str]]], str]:
    """Load datasets or experiments from config.

    Handles both experiment-based config (with 'experiments' key) and
    pattern-based dataset discovery (with 'images_pattern' key).

    Args:
        config: Configuration dictionary
        verbose: Whether to print found datasets

    Returns:
        Tuple of (datasets, output_pattern) where datasets is a list of tuples:
            (name, images_path, gt_path, output_path, images_list, preprocess_path)
    """
    benchmark_opts = config.get("benchmark", {})
    filter_datasets = benchmark_opts.get("datasets", None)

    # Check for dataset_groups, experiments, or single-pattern config
    dataset_groups = config.get("dataset_groups", None)
    experiments_config = config.get("experiments", None)

    datasets = []
    output_patterns = []

    if dataset_groups:
        # Multiple dataset groups (e.g., ETH3D train + test in one config)
        for group in dataset_groups:
            group_images = group["images_pattern"]
            group_gt = group.get("gt_pattern")
            group_output = group["output_pattern"]
            group_name = group.get("name", "")
            output_patterns.append(group_output)

            discovered = discover_datasets(group_images, group_gt, filter_datasets)
            for name, img, gt in discovered:
                display_name = f"{group_name}/{name}" if group_name else name
                datasets.append(
                    (display_name, img, gt, group_output.replace("{dataset}", name), None, None)
                )

    if experiments_config:
        # Load experiments from text files (can combine with dataset_groups)
        exp_datasets = load_experiments(
            experiments_config["files"],
            experiments_config["gt_pattern"],
            experiments_config["output_pattern"],
            experiments_config.get("preprocess_pattern"),
        )
        datasets.extend(exp_datasets)
        output_patterns.append(experiments_config["output_pattern"])

    if not dataset_groups and not experiments_config:
        # Discover datasets using patterns
        images_pattern = config["images_pattern"]
        gt_pattern = config.get("gt_pattern")
        output_pattern = config["output_pattern"]

        discovered = discover_datasets(images_pattern, gt_pattern, filter_datasets)
        # Convert to same format with None for images_list and preprocess_path
        datasets = [
            (name, img, gt, output_pattern.replace("{dataset}", name), None, None)
            for name, img, gt in discovered
        ]
        output_patterns.append(output_pattern)

    # Derive output_pattern from collected patterns
    if output_patterns:
        if len(output_patterns) == 1:
            output_pattern = output_patterns[0]
        else:
            output_pattern = os.path.commonpath(
                [p.replace("{dataset}", "").replace("{scene}", "").replace("{exp_name}", "")
                 for p in output_patterns]
            ) if output_patterns else "results"
    else:
        output_pattern = "results"

    if verbose:
        print(f"\nFound {len(datasets)} dataset(s)/experiment(s):")
        for item in datasets:
            name = item[0]
            images_list = item[4] if len(item) > 4 else None
            if images_list is not None:
                print(f"  - {name} ({len(images_list)} images)")
            else:
                print(f"  - {name}")
        print()

    return datasets, output_pattern


def get_output_dir_from_pattern(output_pattern: str) -> str:
    """Extract output directory from pattern by removing placeholders.

    Args:
        output_pattern: Pattern with placeholders like {dataset}, {scene}, {exp_name}

    Returns:
        Output directory path
    """
    output_dir = os.path.dirname(
        output_pattern.replace("{dataset}", "")
        .replace("{benchmark}", "")
        .replace("{scene}", "")
        .replace("{exp_name}", "")
    )
    return output_dir if output_dir else "."


def save_results_csv(
    results: Dict[str, Dict[str, Any]],
    failures: Dict[str, str],
    output_path: str,
    csv_name: str = "benchmark_results.csv",
) -> str:
    """Save benchmark results to a CSV file.

    Args:
        results: Dictionary mapping dataset names to metric dictionaries
        failures: Dictionary mapping dataset names to error messages
        output_path: Directory to save the results
        csv_name: Name of the CSV file (default: benchmark_results.csv)

    Returns:
        Path to the saved CSV file
    """
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, csv_name)

    # Collect all metric keys from successful results
    metric_keys = set()
    for metrics in results.values():
        metric_keys.update(metrics.keys())
    metric_keys = sorted(metric_keys)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["dataset", "status", "error"] + metric_keys
        writer.writerow(header)

        # Write successful results
        for dataset_name in sorted(results.keys()):
            metrics = results[dataset_name]
            row = [dataset_name, "success", ""]
            for key in metric_keys:
                row.append(metrics.get(key, ""))
            writer.writerow(row)

        # Write failures
        for dataset_name, error in sorted(failures.items()):
            row = [dataset_name, "failed", error]
            row.extend([""] * len(metric_keys))
            writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")
    return csv_path
