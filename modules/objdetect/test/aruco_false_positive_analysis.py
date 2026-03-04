#!/usr/bin/env python3
"""
Analyze false positive sweeps from CV_ArucoDetectFalsePositives.

Expected input is the CSVs written by the test in
`aruco_false_positive_results/summary.csv` (configurable).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_summary(path):
    data = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dictionary = row["dictionary"]
                threshold = float(row["valid_bit_id_threshold"])
                total_images = int(row["total_images"])
                total_detections = int(row["total_detections"])
                images_with_detections = int(row["images_with_detections"])
                mean_confidence = float(row["mean_confidence"])
                max_confidence = float(row["max_confidence"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row in {path}: {row}") from exc

            detections_per_image = (
                total_detections / total_images if total_images else 0.0
            )
            image_detection_rate = (
                images_with_detections / total_images if total_images else 0.0
            )

            data[dictionary].append(
                {
                    "threshold": threshold,
                    "total_detections": total_detections,
                    "detections_per_image": detections_per_image,
                    "images_with_detections": images_with_detections,
                    "image_detection_rate": image_detection_rate,
                    "mean_confidence": mean_confidence,
                    "max_confidence": max_confidence,
                }
            )
    return data


def load_per_id(path):
    data = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dictionary = row["dictionary"]
                threshold = float(row["valid_bit_id_threshold"])
                marker_id = int(row["id"])
                detection_count = int(row["detection_count"])
                mean_confidence = float(row["mean_confidence"])
                max_confidence = float(row["max_confidence"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row in {path}: {row}") from exc
            data[(dictionary, threshold)].append(
                {
                    "id": marker_id,
                    "detection_count": detection_count,
                    "mean_confidence": mean_confidence,
                    "max_confidence": max_confidence,
                }
            )
    return data


def load_confidence_samples(path):
    data = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dictionary = row["dictionary"]
                threshold = float(row["valid_bit_id_threshold"])
                confidences = row["confidences"]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row in {path}: {row}") from exc

            if not confidences:
                continue
            for item in confidences.split(";"):
                item = item.strip()
                if not item:
                    continue
                try:
                    data[(dictionary, threshold)].append(float(item))
                except ValueError as exc:
                    raise ValueError(f"Bad confidence value: {item}") from exc
    return data


def plot_metric(data, metric_key, ylabel, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run with --no-plot."
        ) from exc

    plt.figure(figsize=(8, 5))
    for dictionary, entries in sorted(data.items()):
        entries_sorted = sorted(entries, key=lambda item: item["threshold"])
        thresholds = [entry["threshold"] for entry in entries_sorted]
        values = [entry[metric_key] for entry in entries_sorted]
        plt.plot(thresholds, values, marker="o", label=dictionary)

    plt.xlabel("validBitIdThreshold")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)


def plot_id_histograms(per_id_data, threshold, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run with --no-plot."
        ) from exc

    plotted = False
    for (dictionary, entry_threshold), entries in sorted(per_id_data.items()):
        if abs(entry_threshold - threshold) > 1e-6:
            continue
        if not entries:
            continue
        ids = [entry["id"] for entry in entries]
        counts = [entry["detection_count"] for entry in entries]
        if not ids:
            continue

        plt.figure(figsize=(9, 4))
        plt.bar(ids, counts, width=1.0)
        plt.xlabel("Marker id")
        plt.ylabel("False positive count")
        plt.title(f"{dictionary} @ validBitIdThreshold={threshold:.2f}")
        plt.tight_layout()
        output_path = os.path.join(
            output_dir, f"false_positive_ids_{dictionary}_thr{threshold:.2f}.png"
        )
        plt.savefig(output_path, dpi=160)
        plotted = True
    return plotted


def plot_confidence_histograms(confidence_data, threshold, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run with --no-plot."
        ) from exc

    plotted = False
    for (dictionary, entry_threshold), samples in sorted(confidence_data.items()):
        if abs(entry_threshold - threshold) > 1e-6:
            continue
        if not samples:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(samples, bins=20, range=(0.0, 1.0))
        plt.xlabel("Detection confidence")
        plt.ylabel("False positive count")
        plt.title(f"{dictionary} @ validBitIdThreshold={threshold:.2f}")
        plt.tight_layout()
        output_path = os.path.join(
            output_dir, f"false_positive_confidence_{dictionary}_thr{threshold:.2f}.png"
        )
        plt.savefig(output_path, dpi=160)
        plotted = True
    return plotted


def print_best_thresholds(data):
    print("Best threshold per dictionary (lowest detections per image):")
    for dictionary, entries in sorted(data.items()):
        best = min(entries, key=lambda item: item["detections_per_image"])
        print(
            f"  {dictionary}: threshold={best['threshold']:.2f}, "
            f"detections/image={best['detections_per_image']:.6f}, "
            f"image rate={best['image_detection_rate']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot false positives vs validBitIdThreshold."
    )
    parser.add_argument(
        "--summary",
        default="aruco_false_positive_results/summary.csv",
        help="Path to summary.csv generated by the C++ test.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for plots. Defaults to the summary.csv directory.",
    )
    parser.add_argument(
        "--per-id",
        default="",
        help="Path to per_id.csv (defaults to summary.csv directory).",
    )
    parser.add_argument(
        "--per-image",
        default="",
        help="Path to per_image.csv (defaults to summary.csv directory).",
    )
    parser.add_argument(
        "--hist-threshold",
        type=float,
        default=0.5,
        help="Threshold value for histograms.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (only print summary stats).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    args = parser.parse_args()

    summary_path = args.summary
    if not os.path.isfile(summary_path):
        print(f"Missing summary.csv: {summary_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.dirname(summary_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    data = load_summary(summary_path)
    if not data:
        print("No data found in summary.csv", file=sys.stderr)
        return 1

    print_best_thresholds(data)

    if not args.no_plot:
        plot_metric(
            data,
            "detections_per_image",
            "False positives per image",
            os.path.join(output_dir, "false_positives_per_image.png"),
        )
        plot_metric(
            data,
            "image_detection_rate",
            "Images with false positives (rate)",
            os.path.join(output_dir, "false_positive_image_rate.png"),
        )
        plot_metric(
            data,
            "total_detections",
            "False positives total (count)",
            os.path.join(output_dir, "false_positives_total.png"),
        )
        plot_metric(
            data,
            "mean_confidence",
            "False positive mean confidence",
            os.path.join(output_dir, "false_positive_mean_confidence.png"),
        )
        plot_metric(
            data,
            "max_confidence",
            "False positive max confidence",
            os.path.join(output_dir, "false_positive_max_confidence.png"),
        )

        per_id_path = args.per_id or os.path.join(
            os.path.dirname(summary_path), "per_id.csv"
        )
        if os.path.isfile(per_id_path):
            per_id_data = load_per_id(per_id_path)
            if not plot_id_histograms(per_id_data, args.hist_threshold, output_dir):
                print(
                    f"No per-id data for threshold={args.hist_threshold:.2f}",
                    file=sys.stderr,
                )
        else:
            print(f"Missing per_id.csv: {per_id_path}", file=sys.stderr)

        per_image_path = args.per_image or os.path.join(
            os.path.dirname(summary_path), "per_image.csv"
        )
        if os.path.isfile(per_image_path):
            confidence_data = load_confidence_samples(per_image_path)
            if not plot_confidence_histograms(
                confidence_data, args.hist_threshold, output_dir
            ):
                print(
                    f"No confidence samples for threshold={args.hist_threshold:.2f}",
                    file=sys.stderr,
                )
        else:
            print(f"Missing per_image.csv: {per_image_path}", file=sys.stderr)

        if args.show:
            import matplotlib.pyplot as plt

            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
