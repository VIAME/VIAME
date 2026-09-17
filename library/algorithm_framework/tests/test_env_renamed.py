# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`viame.util.env.get_renamed`, the python half of the phase 11 env rename.

The C++ half is `viame::get_env_renamed`, covered by `test_file_system.cxx`.
Both follow the same rule, and the rule is the point: an environment written
for an older VIAME keeps working, and says once that it is doing so.
"""

import logging

import pytest

from viame.util import env as env_mod
from viame.util.env import get_renamed


NEW = "VIAME_ENV_RENAMED_TEST"
OLD = "KWIVER_ENV_RENAMED_TEST"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """No leakage between cases, including the warned-once memory."""
    monkeypatch.delenv(NEW, raising=False)
    monkeypatch.delenv(OLD, raising=False)
    env_mod._WARNED.discard(OLD)
    yield
    env_mod._WARNED.discard(OLD)


def test_the_new_name_is_read(monkeypatch):
    monkeypatch.setenv(NEW, "new value")
    assert get_renamed(NEW, OLD) == "new value"


def test_the_old_name_is_still_read(monkeypatch):
    monkeypatch.setenv(OLD, "old value")
    assert get_renamed(NEW, OLD) == "old value"


def test_the_new_name_wins_over_the_old(monkeypatch):
    """Both set is what a half-migrated environment looks like."""
    monkeypatch.setenv(NEW, "new value")
    monkeypatch.setenv(OLD, "old value")
    assert get_renamed(NEW, OLD) == "new value"


def test_neither_set_gives_the_default(monkeypatch):
    assert get_renamed(NEW, OLD) is None
    assert get_renamed(NEW, OLD, "fallback") == "fallback"


def test_an_empty_new_name_is_a_value(monkeypatch):
    """Explicitly emptied is a setting, not an absence.

    `VIAME_PIPE_INCLUDE_PATH=` is how a caller says "no extra directories",
    and falling through to the old name there would ignore them.
    """
    monkeypatch.setenv(NEW, "")
    monkeypatch.setenv(OLD, "old value")
    assert get_renamed(NEW, OLD) == ""


def test_the_old_name_warns_once(monkeypatch, caplog):
    monkeypatch.setenv(OLD, "old value")

    with caplog.at_level(logging.WARNING, logger=env_mod._LOG.name):
        assert get_renamed(NEW, OLD) == "old value"
        first = len(caplog.records)
        assert get_renamed(NEW, OLD) == "old value"
        second = len(caplog.records)

    assert first == 1, caplog.records
    # Read on every discovery pass; saying it every time would bury it.
    assert second == 1, caplog.records
    assert OLD in caplog.records[0].getMessage()
    assert NEW in caplog.records[0].getMessage()


def test_the_new_name_alone_warns_about_nothing(monkeypatch, caplog):
    monkeypatch.setenv(NEW, "new value")

    with caplog.at_level(logging.WARNING, logger=env_mod._LOG.name):
        assert get_renamed(NEW, OLD) == "new value"

    assert caplog.records == []
