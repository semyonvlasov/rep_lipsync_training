from __future__ import annotations

import logging

from face_processing.models import FrameData, Segment

logger = logging.getLogger(__name__)


def split_into_segments(
    frame_data: list[FrameData],
    min_length: int = 50,
    max_length: int = 750,
) -> tuple[list[Segment], list[Segment]]:
    """Split frame_data into continuous runs of good frames.

    Long runs exceeding max_length are subdivided:
    - If length > 2 * max_length: cut max_length chunks, repeat
    - If max_length < length <= 2 * max_length: split into 2 roughly equal halves

    Returns:
        (exportable, dropped) — exportable segments have length >= min_length,
        dropped segments are too short.
    """
    if max_length > 0 and max_length < 2 * min_length:
        raise ValueError(
            f"max_segment_length ({max_length}) must be >= 2 * min_segment_length ({min_length})"
        )

    # Step 1: find continuous runs of good frames
    raw_runs: list[tuple[int, int]] = []  # (start, end) exclusive
    run_start: int | None = None

    for i, fd in enumerate(frame_data):
        if not fd.is_bad:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                raw_runs.append((run_start, i))
                run_start = None

    if run_start is not None:
        raw_runs.append((run_start, len(frame_data)))

    # Step 2: subdivide long runs
    split_runs: list[tuple[int, int]] = []
    for start, end in raw_runs:
        length = end - start
        if max_length <= 0 or length <= max_length:
            split_runs.append((start, end))
        else:
            split_runs.extend(_subdivide_run(start, end, max_length))

    # Step 3: create Segment objects, filter by min_length
    exportable: list[Segment] = []
    dropped: list[Segment] = []
    segment_id = 0

    for start, end in split_runs:
        length = end - start
        seg = Segment(
            segment_id=segment_id,
            start_frame=start,
            end_frame=end,
            length=length,
            frame_data=frame_data[start:end],
        )
        segment_id += 1

        if length >= min_length:
            exportable.append(seg)
        else:
            seg.status = "dropped"
            seg.drop_reason = "segment_too_short"
            dropped.append(seg)

    if max_length > 0:
        logger.info(
            "Segmentation: %d raw runs -> %d segments after max_length=%d split",
            len(raw_runs), len(split_runs), max_length,
        )

    return exportable, dropped


def _subdivide_run(start: int, end: int, max_length: int) -> list[tuple[int, int]]:
    """Split a long run into chunks respecting max_length.

    - length > 2 * max_length: cut max_length pieces from the front
    - max_length < length <= 2 * max_length: split into 2 roughly equal halves
    """
    result: list[tuple[int, int]] = []
    pos = start
    remaining = end - pos

    while remaining > 0:
        if remaining <= max_length:
            result.append((pos, end))
            break
        elif remaining <= 2 * max_length:
            # Split into 2 roughly equal halves
            mid = pos + remaining // 2
            result.append((pos, mid))
            result.append((mid, end))
            break
        else:
            # Cut a max_length chunk
            result.append((pos, pos + max_length))
            pos += max_length
            remaining = end - pos

    return result
