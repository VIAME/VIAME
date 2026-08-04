# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Every tracker trainer must load, publish its options, and survive a
configuration round trip.

This is what kwiver does to a trainer before any data reaches it, so anything
broken here fails the run in its first seconds -- and, since a python plugin
that raises on import is silently dropped rather than reported, sometimes
fails it as "Could not find implementation" pointing at the plugin path
instead of at the actual error.

Three separate faults would have been caught here in the time this takes to
run: an option published in get_configuration but never set in __init__, a
module that could not be imported because a new dependency was not installed,
and a helper referenced in one trainer but only defined in another.
"""

import pytest

TRAINERS = [
    ('viame.core.bytetrack_trainer', 'ByteTrackTrainer'),
    ('viame.core.ocsort_trainer', 'OCSORTTrainer'),
    ('viame.pytorch.deepsort_trainer', 'DeepSORTTrainer'),
    ('viame.pytorch.botsort_trainer', 'BoTSORTTrainer'),
    ('viame.pytorch.srnn_trainer', 'SRNNTrainer'),
    ('viame.pytorch.siammask_trainer', 'SiamMaskTrainer'),
]


def load(module_name, class_name):
    """Import a trainer, skipping only when its subsystem is not built.

    The distinction matters: a build without pytorch should not fail this,
    but a build *with* pytorch whose trainer does not import is exactly the
    failure worth catching, and blanket skipping on ImportError would hide
    it. So the package is imported first, and only that is allowed to skip.
    """
    import importlib

    package = module_name.rsplit('.', 1)[0]

    try:
        importlib.import_module(package)
    except ImportError as error:
        pytest.skip('{} not built: {}'.format(package, error))

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@pytest.mark.parametrize('module_name,class_name', TRAINERS)
def test_imports(module_name, class_name):
    """The module imports and the class is there."""
    assert load(module_name, class_name) is not None


@pytest.mark.parametrize('module_name,class_name', TRAINERS)
def test_constructs(module_name, class_name):
    """__init__ runs without the configuration having been set."""
    load(module_name, class_name)()


@pytest.mark.parametrize('module_name,class_name', TRAINERS)
def test_configuration_round_trip(module_name, class_name):
    """get_configuration then set_configuration then get_configuration.

    The second get is the one that matters: it reads back attributes that
    set_configuration assigned, so an option published without a default in
    __init__ raises AttributeError here rather than in a job.
    """
    trainer = load(module_name, class_name)()

    first = trainer.get_configuration()
    trainer.set_configuration(first)
    second = trainer.get_configuration()

    for key in first.available_values():
        assert second.has_value(key), \
            '{} lost {} across a round trip'.format(class_name, key)


@pytest.mark.parametrize('module_name,class_name', TRAINERS)
def test_published_options_have_defaults(module_name, class_name):
    """Everything get_configuration publishes is backed by an attribute.

    Reaching into the instance rather than only exercising the round trip,
    because an option whose attribute is missing but which is never read back
    would otherwise pass.
    """
    trainer = load(module_name, class_name)()
    missing = []

    for key in trainer.get_configuration().available_values():
        attribute = '_' + key

        if not hasattr(trainer, attribute):
            # Not every option maps to an underscore attribute of the same
            # name; only complain when nothing plausible is there at all.
            alternatives = [attribute, key, '_' + key.replace(':', '_')]

            if not any(hasattr(trainer, a) for a in alternatives):
                missing.append(key)

    assert not missing, \
        '{} publishes {} with no backing attribute'.format(class_name, missing)
