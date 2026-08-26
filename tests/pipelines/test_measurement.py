import pytest

from .cases import discover, run_case


@pytest.mark.parametrize("case", discover("measurement"), ids=lambda case: case.id)
def test_measurement(case, runner, env_dir, request):
    run_case(case, runner, env_dir, request)
