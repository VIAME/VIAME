"""
ckwg +29
Copyright 2020 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Base class for testing the common functionality of all algorithms
"""

from kwiver.vital.config import empty_config
from kwiver.vital.tests.py_helpers import generate_dummy_config
import kwiver.vital.algo
from kwiver.vital import plugin_management
import pytest


def get_algo_list():
    vpm = plugin_management.plugin_manager_instance()
    vpm.load_all_plugins()
    algo_list = []
    for v in kwiver.vital.algo.__dict__.values():
        if isinstance(v, type) and issubclass(v, kwiver.vital.algo.algos._algorithm):
            simple_impl_name = "Simple" + v.__name__
            all_impl_names = [sc.__name__ for sc in v.__subclasses__()]
            # Note that we assume the structure of the name of the wrapper module
            if simple_impl_name in all_impl_names:
                algo_list.append((v, simple_impl_name))
    return algo_list


def _dummy_algorithm_cfg():
    return generate_dummy_config(threshold=0.3)


class TestVitalAlgorithmsCommon(object):
    def get_fresh_instance(self, abstract_algo, implementation_name):
        return abstract_algo.create(implementation_name)

    # Display all the registered implementations of this
    # abstract algorithm
    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_registered_names(self, abstract_algo, simple_impl_name):
        registered_implementation_names = abstract_algo.registered_names()
        print("\nAll registered {}".format(abstract_algo.interface_name()))
        for name in registered_implementation_names:
            print(" " + name)

    # Test create function
    # For an invalid value it raises RuntimeError
    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_bad_create(self, abstract_algo, simple_impl_name):
        with pytest.raises(RuntimeError):
            # Should fail to create an algorithm without a factory
            abstract_algo.create("NonExistantAlgorithm")

    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_create_helper(self, abstract_algo, simple_impl_name):
        assert (
            simple_impl_name in abstract_algo.registered_names()
        ), f"No simple implementation found for {abstract_algo.interface_name()}"
        algo_out = abstract_algo.create(simple_impl_name)

        assert isinstance(algo_out, abstract_algo)

    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_algo_factory_create(self, abstract_algo, simple_impl_name):
        assert abstract_algo.has_algorithm_impl_name(
            simple_impl_name
        ), f"{simple_impl_name} not found by the factory"

        assert (
            simple_impl_name in abstract_algo.registered_names()
        ), f"{simple_impl_name} not in implementations list for {abstract_algo.interface_name()}"

        algo_out = abstract_algo.create(simple_impl_name)

        assert isinstance(algo_out, abstract_algo)

    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_impl(self, abstract_algo, simple_impl_name):
        a = abstract_algo.impl_name
        abstract_algo.impl_name = "example_impl_name"
        assert abstract_algo.impl_name == "example_impl_name"

    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_config_helper(self, abstract_algo, simple_impl_name):
        instance = self.get_fresh_instance(abstract_algo, simple_impl_name)
        instance_cfg = instance.get_configuration()
        # Verify that "threshold" config value is present
        assert instance_cfg.has_value("threshold"), "threshold config value not present"
        # Verify that the value for key "threshold" is 0.0
        threshold_value = instance_cfg.get_value("threshold")
        assert (
            threshold_value == "0.0"
        ), f"threshold config value {threshold_value}, expected 0.0"

        test_cfg = _dummy_algorithm_cfg()
        # Verify that the instance has different configuration before setting to test
        assert not instance.check_configuration(test_cfg)
        instance.set_configuration(test_cfg)
        # Verify that the config value is being set properly
        assert instance.check_configuration(test_cfg)

    @pytest.mark.parametrize("abstract_algo, simple_impl_name", get_algo_list())
    def test_nested_config_helper(self, abstract_algo, simple_impl_name):
        instance = self.get_fresh_instance(abstract_algo, simple_impl_name)
        nested_cfg = empty_config()
        instance.get_nested_algo_configuration("algorithm", nested_cfg, instance)

        assert instance.check_nested_algo_configuration("algorithm", nested_cfg)

        nested_algo = abstract_algo.set_nested_algo_configuration(
            "algorithm", nested_cfg
        )

        # Should have created a concrete algorithm instance
        assert type(instance) is type(nested_algo)

        # Verify that the value for key "threshold" is 0.0
        threshold_value = nested_algo.get_configuration().get_value("threshold")
        assert (
            threshold_value == "0.0"
        ), f"threshold config value {threshold_value}, expected 0.0"

        # Check case where the value for key "type" doesn't match
        # any implementation
        nested_cfg.subblock_view("algorithm").set_value("type", "foo")

        # Check should fail
        assert not (instance.check_nested_algo_configuration("algorithm", nested_cfg))

        # Should get back nullptr
        assert instance.set_nested_algo_configuration("algorithm", nested_cfg) is None
