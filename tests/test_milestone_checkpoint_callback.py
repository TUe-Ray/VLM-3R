import json
from types import SimpleNamespace

import pytest
from transformers import TrainerControl, TrainerState

from llava.train.llava_trainer import MilestoneCheckpointCallback


def _args(tmp_path, *, save_strategy="no", save_total_limit=3):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
    )


def _state(*, step=0, max_steps=100):
    return TrainerState(
        global_step=step,
        max_steps=max_steps,
        is_world_process_zero=True,
        is_local_process_zero=True,
    )


def test_milestone_callback_requests_and_records_only_requested_steps(tmp_path):
    args = _args(tmp_path)
    state = _state()
    callback = MilestoneCheckpointCallback("0.05,0.25,0.50")
    callback.on_train_begin(args, state, TrainerControl())

    assert callback.milestones == {5: [0.05], 25: [0.25], 50: [0.5]}
    for step in (4, 6, 24, 26, 49, 51):
        state.global_step = step
        assert not callback.on_step_end(args, state, TrainerControl()).should_save

    for step in (5, 25, 50):
        state.global_step = step
        control = callback.on_step_end(args, state, TrainerControl())
        assert control.should_save
        (tmp_path / f"checkpoint-{step}").mkdir()
        callback.on_save(args, state, control)

    callback.on_train_end(args, state, TrainerControl())
    manifest = json.loads((tmp_path / callback.MANIFEST_NAME).read_text())
    assert [entry["step"] for entry in manifest["milestones"]] == [5, 25, 50]
    assert all(entry["saved"] for entry in manifest["milestones"])


def test_milestone_callback_uses_ceil_for_discrete_training_steps(tmp_path):
    callback = MilestoneCheckpointCallback("0.05,0.25,0.50")
    callback.on_train_begin(_args(tmp_path), _state(max_steps=21), TrainerControl())
    assert list(callback.milestones) == [2, 6, 11]


@pytest.mark.parametrize(
    ("save_strategy", "save_total_limit", "message"),
    [("steps", 3, "save_strategy no"), ("no", 2, "cannot preserve")],
)
def test_milestone_callback_rejects_settings_that_can_rotate_milestones(
    tmp_path, save_strategy, save_total_limit, message
):
    callback = MilestoneCheckpointCallback("0.05,0.25,0.50")
    with pytest.raises(ValueError, match=message):
        callback.on_train_begin(
            _args(tmp_path, save_strategy=save_strategy, save_total_limit=save_total_limit),
            _state(),
            TrainerControl(),
        )
