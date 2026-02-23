from __future__ import annotations

import argparse
import json

import yaml

from avic_gatn.pipeline.end2end import run_end2end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out = run_end2end(cfg)
    print("[DONE] report:", out["report_path"])
    print(json.dumps({"baseline_primary": out["baseline"]["primary"], "attack_success_delta": out["alg2_attack"]["success_delta"]}, indent=2))


if __name__ == "__main__":
    main()
