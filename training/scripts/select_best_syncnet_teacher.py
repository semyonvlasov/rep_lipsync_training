#!/usr/bin/env python3

"""
Choose the best SyncNet teacher from a comparison JSON.

Selection rule:
  1. highest pairwise_acc_mean
  2. highest margin_mean
  3. highest foreign_pairwise_acc
  4. highest shifted_pairwise_acc
  5. prefer the official teacher on exact ties
"""

import argparse
import json
import os


def metric_key(name, metrics):
    return (
        float(metrics.get("pairwise_acc_mean", float("-inf"))),
        float(metrics.get("margin_mean", float("-inf"))),
        float(metrics.get("foreign_pairwise_acc", float("-inf"))),
        float(metrics.get("shifted_pairwise_acc", float("-inf"))),
        1 if name == "official_syncnet" else 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-json", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.compare_json) as f:
        compare = json.load(f)

    teachers = compare.get("teachers", {})
    if not teachers:
        raise RuntimeError(f"No teacher metrics found in {args.compare_json}")

    checkpoint_by_name = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in args.checkpoints
    }

    winner_name, winner_metrics = max(
        teachers.items(),
        key=lambda item: metric_key(item[0], item[1]),
    )

    if winner_name == "official_syncnet":
        winner_path = args.official_checkpoint
        winner_kind = "official"
    else:
        if winner_name not in checkpoint_by_name:
            raise RuntimeError(
                f"Winner {winner_name} is not mapped to a provided checkpoint: "
                f"known={sorted(checkpoint_by_name)}"
            )
        winner_path = checkpoint_by_name[winner_name]
        winner_kind = "local"

    result = {
        "winner_name": winner_name,
        "winner_kind": winner_kind,
        "winner_path": winner_path,
        "winner_metrics": winner_metrics,
        "official_checkpoint": args.official_checkpoint,
        "candidate_checkpoints": args.checkpoints,
        "teachers": teachers,
        "compare_json": args.compare_json,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"winner_name={winner_name}")
    print(f"winner_kind={winner_kind}")
    print(f"winner_path={winner_path}")
    print(f"pairwise_acc_mean={winner_metrics.get('pairwise_acc_mean')}")
    print(f"margin_mean={winner_metrics.get('margin_mean')}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
