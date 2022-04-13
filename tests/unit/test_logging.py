from __future__ import annotations

from morphoclass.serialization import get_experiment_directory


def test_get_experiment_directory(tmp_path):
    log_dir = get_experiment_directory(str(tmp_path), "asdf")
    print(log_dir)
