# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Extract RoboGauge TensorBoard scalars into a standalone log directory.

Example:
    python scripts/tools/extract_robogauge_tensorboard.py logs/rsl_rl/go2_moe_cts_v4
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


DEFAULT_OUTPUT_ROOT = Path("logs/rsl_rl/only_robogauge")
DEFAULT_TAG_PREFIX = "RoboGauge/"
PROGRESS_BAR_WIDTH = 32
SCALAR_SIZE_GUIDANCE = {event_accumulator.SCALARS: 0}


@dataclass
class MatchedEventDir:
    event_dir: Path
    tags: list[str]
    raw_scalar_count: int
    deduplicated_scalar_count: int


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool) -> None:
        self.total = total
        self.label = label
        self.enabled = enabled and total > 0
        self.current = 0
        self._render_step = max(total // 100, 1)
        self._last_rendered = -1
        self._last_message_len = 0

        if self.enabled:
            self.render()

    def update(self, advance: int = 1) -> None:
        if not self.enabled:
            return

        self.current = min(self.current + advance, self.total)
        if self.current == self.total or self.current - self._last_rendered >= self._render_step:
            self.render()

    def finish(self) -> None:
        if not self.enabled:
            return

        self.current = self.total
        self.render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def render(self) -> None:
        progress = self.current / self.total
        filled_width = int(progress * PROGRESS_BAR_WIDTH)
        bar = "#" * filled_width + "-" * (PROGRESS_BAR_WIDTH - filled_width)
        message = f"\r{self.label} [{bar}] {self.current}/{self.total} ({progress:.0%})"
        padding = " " * max(self._last_message_len - len(message), 0)

        sys.stderr.write(message + padding)
        sys.stderr.flush()
        self._last_rendered = self.current
        self._last_message_len = len(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RoboGauge scalars from TensorBoard event files.")
    parser.add_argument("source_dir", type=Path, help="Directory that contains TensorBoard event files.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name of the output experiment folder. Defaults to the source directory name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for extracted logs. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default=DEFAULT_TAG_PREFIX,
        help=f"TensorBoard scalar tag prefix to extract. Defaults to {DEFAULT_TAG_PREFIX}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output experiment directory before writing new extracted logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching RoboGauge scalars without writing TensorBoard files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def normalize_tag_prefix(tag_prefix: str) -> str:
    if tag_prefix.endswith("/"):
        return tag_prefix
    return f"{tag_prefix}/"


def collect_event_dirs(source_dir: Path, excluded_dir: Path) -> list[Path]:
    event_dirs = set()
    for event_file in source_dir.rglob("events.out.tfevents*"):
        if is_relative_to(event_file.resolve(), excluded_dir):
            continue
        event_dirs.add(event_file.parent.resolve())
    return sorted(event_dirs)


def get_robogauge_tags(event_dir: Path, tag_prefix: str) -> tuple[event_accumulator.EventAccumulator, list[str]]:
    accumulator = event_accumulator.EventAccumulator(str(event_dir), size_guidance=SCALAR_SIZE_GUIDANCE)
    accumulator.Reload()
    tags = sorted(tag for tag in accumulator.Tags().get("scalars", []) if tag.startswith(tag_prefix))
    return accumulator, tags


def count_scalars(accumulator: event_accumulator.EventAccumulator, tags: list[str]) -> tuple[int, int]:
    raw_count = 0
    deduplicated_count = 0
    for tag in tags:
        scalars = accumulator.Scalars(tag)
        raw_count += len(scalars)
        deduplicated_count += len({scalar.step for scalar in scalars})
    return raw_count, deduplicated_count


def get_deduplicated_scalars(
    accumulator: event_accumulator.EventAccumulator,
    tag: str,
) -> list[event_accumulator.ScalarEvent]:
    scalars_by_step = {}
    for scalar in accumulator.Scalars(tag):
        scalars_by_step[scalar.step] = scalar
    return [scalars_by_step[step] for step in sorted(scalars_by_step)]


def scan_event_dirs(event_dirs: list[Path], tag_prefix: str, progress_enabled: bool) -> list[MatchedEventDir]:
    matched_event_dirs = []
    progress_bar = ProgressBar(len(event_dirs), "Scanning event dirs", progress_enabled)
    try:
        for event_dir in event_dirs:
            accumulator, tags = get_robogauge_tags(event_dir, tag_prefix)
            if tags:
                raw_scalar_count, deduplicated_scalar_count = count_scalars(accumulator, tags)
                matched_event_dirs.append(
                    MatchedEventDir(
                        event_dir=event_dir,
                        tags=tags,
                        raw_scalar_count=raw_scalar_count,
                        deduplicated_scalar_count=deduplicated_scalar_count,
                    )
                )
            progress_bar.update()
    finally:
        progress_bar.finish()
    return matched_event_dirs


def extract_scalars(
    matched_event_dir: MatchedEventDir,
    destination_dir: Path,
    dry_run: bool,
    progress_bar: ProgressBar | None,
) -> int:
    if dry_run:
        return matched_event_dir.deduplicated_scalar_count

    accumulator = event_accumulator.EventAccumulator(str(matched_event_dir.event_dir), size_guidance=SCALAR_SIZE_GUIDANCE)
    accumulator.Reload()
    destination_dir.mkdir(parents=True, exist_ok=True)
    scalar_count = 0
    writer = SummaryWriter(log_dir=str(destination_dir), flush_secs=1)
    try:
        for tag in matched_event_dir.tags:
            for scalar in get_deduplicated_scalars(accumulator, tag):
                writer.add_scalar(tag, scalar.value, scalar.step, walltime=scalar.wall_time)
                scalar_count += 1
                if progress_bar is not None:
                    progress_bar.update()
    finally:
        writer.close()
    return scalar_count


def prepare_output_dir(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        print(f"[WARN] Output directory already exists: {output_dir}")
        print("[WARN] Existing TensorBoard events will remain. Use --overwrite to rebuild it from scratch.")
    output_dir.mkdir(parents=True, exist_ok=True)


def should_show_progress(disable_progress: bool) -> bool:
    return not disable_progress and sys.stderr.isatty()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tag_prefix = normalize_tag_prefix(args.tag_prefix)
    experiment_name = args.experiment_name or source_dir.name
    output_dir = output_root / experiment_name

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    prepare_output_dir(output_dir, args.overwrite, args.dry_run)
    event_dirs = collect_event_dirs(source_dir, output_root)
    if not event_dirs:
        print(f"No TensorBoard event files found under: {source_dir}")
        return

    total_tags = 0
    total_scalars = 0
    progress_enabled = should_show_progress(args.no_progress)
    matched_event_dirs = scan_event_dirs(event_dirs, tag_prefix, progress_enabled)

    if not matched_event_dirs:
        print(f"No scalar tags matching '{tag_prefix}' were found under: {source_dir}")
        return

    write_progress = None
    if not args.dry_run:
        write_total = sum(matched_event_dir.deduplicated_scalar_count for matched_event_dir in matched_event_dirs)
        write_progress = ProgressBar(write_total, "Writing scalars", progress_enabled)

    result_lines = []
    try:
        for matched_event_dir in matched_event_dirs:
            relative_dir = matched_event_dir.event_dir.relative_to(source_dir)
            destination_dir = output_dir / relative_dir
            scalar_count = extract_scalars(
                matched_event_dir,
                destination_dir,
                args.dry_run,
                write_progress,
            )

            total_tags += len(matched_event_dir.tags)
            total_scalars += scalar_count
            skipped_scalars = matched_event_dir.raw_scalar_count - scalar_count
            action = "Would extract" if args.dry_run else "Extracted"
            result_lines.append(
                f"{action} {scalar_count} scalars from {len(matched_event_dir.tags)} tags: "
                f"{matched_event_dir.event_dir} ({skipped_scalars} duplicate scalars skipped)"
            )
    finally:
        if write_progress is not None:
            write_progress.finish()

    for result_line in result_lines:
        print(result_line)

    print(
        f"Done. Matched {len(matched_event_dirs)} run directories, {total_tags} tags, "
        f"and {total_scalars} scalar points."
    )
    if not args.dry_run:
        print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
