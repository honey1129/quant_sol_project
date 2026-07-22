import argparse
import json
from datetime import datetime

from research.directional_v2 import (
    DEFAULT_HASH_PATH,
    DEFAULT_SPEC_PATH,
    forward_holdout_status,
    load_experiment_spec,
    verify_frozen_spec,
)


def parse_now(value):
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify and inspect the frozen directional-v2 forward experiment.",
    )
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--hash-file", default=DEFAULT_HASH_PATH)
    parser.add_argument("--closed-trades", type=int, default=0)
    parser.add_argument("--now", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    frozen_hash = verify_frozen_spec(args.spec, args.hash_file)
    spec = load_experiment_spec(args.spec)
    status = forward_holdout_status(
        spec,
        now=parse_now(args.now),
        closed_trades=args.closed_trades,
    )
    status["spec_sha256"] = frozen_hash
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
