import argparse
import importlib
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Directory including output/config.json",
    )
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    return parser.parse_args()


def resolve_evaluator_module_name(data_config):
    repo = str(data_config.get("instruction_dataset_repo", "")).lower()

    module_map = {
        "qusum": "qusum_evaluator",
        "pwc": "pwc_evaluator",
        "multifield": "multifield_evaluator",
        "multi_field": "multifield_evaluator",
    }

    for keyword, module_name in module_map.items():
        if keyword in repo:
            return module_name

    return None


def main():
    args = parse_args()

    config_path = os.path.join(args.work_dir, "output", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    module_name = resolve_evaluator_module_name(config.get("data_config", {}))
    if module_name is None:
        raise ValueError(
            "Cannot resolve evaluator module from data_config['instruction_dataset_repo']."
        )

    evaluator_module = importlib.import_module(module_name)
    if not hasattr(evaluator_module, "run_from_dispatcher"):
        raise AttributeError(
            f"{module_name}.run_from_dispatcher(args) is required for routed evaluation."
        )

    evaluator_module.run_from_dispatcher(args)


if __name__ == "__main__":
    main()
