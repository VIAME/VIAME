import pytest

from .cases import discover, run_case


@pytest.mark.parametrize("case", discover("filter"), ids=lambda case: case.id)
def test_filter(case, runner, env_dir, request):
    run_case(case, runner, env_dir, request)
