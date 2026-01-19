"""Test that every model listed in the central manifest resolves to an
existing data file.

This test reads `abacus_combined_PyTAAA_status.params.json` from the
repository root and checks each entry in `models.model_choices`.
"""
import json
import os
import pytest


def load_config():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(repo_root, 'abacus_combined_PyTAAA_status.params.json')
    if not os.path.exists(config_path):
        pytest.skip(f"Central manifest not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def test_models_resolve_to_existing_files():
    cfg = load_config()
    models_cfg = cfg.get('models', {})
    base_folder = models_cfg.get('base_folder', '')
    model_choices = models_cfg.get('model_choices', {})

    monte_cfg = cfg.get('monte_carlo', {})
    data_format = monte_cfg.get('data_format', 'backtested')
    data_files = monte_cfg.get('data_files', {
        'actual': 'PyTAAA_status.params',
        'backtested': 'pyTAAAweb_backtestPortfolioValue.params'
    })
    data_file = data_files.get(data_format, list(data_files.values())[0])

    missing = []

    for model_name, template in model_choices.items():
        # "cash" or similar empty-template models are expected to be empty strings
        if not template:
            continue

        try:
            resolved = template.format(base_folder=base_folder, data_file=data_file)
        except Exception:
            # If template is not a format string, try joining with base_folder
            resolved = os.path.join(base_folder, template)

        resolved = os.path.expanduser(resolved)
        resolved = os.path.abspath(resolved)

        if not os.path.exists(resolved):
            missing.append((model_name, resolved))

    if missing:
        lines = [f"{name}: {path}" for name, path in missing]
        pytest.fail("The following model data files were not found:\n" + "\n".join(lines))
