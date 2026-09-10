import pytest

from .cases import discover, run_case


@pytest.mark.parametrize("case", discover("stereo"), ids=lambda case: case.id)
def test_stereo(case, runner, env_dir, request):
    run_case(case, runner, env_dir, request)
