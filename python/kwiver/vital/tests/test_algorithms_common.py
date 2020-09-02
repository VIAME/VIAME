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
import nose.tools
import os

from kwiver.vital.modules import modules
from kwiver.vital.config import config
from unittest import TestCase
from kwiver.vital.tests.helpers import generate_dummy_config
import kwiver.vital.algo
import kwiver.vital.algo.algos
from kwiver.vital.algo import algorithm_factory


def _dummy_algorithm_cfg():
    return generate_dummy_config(threshold=0.3)


class TestVitalAlgorithmsCommon(object):
    def get_algo_list(self):
        modules.load_known_modules()
        algo_list = []
        for v in kwiver.vital.algo.__dict__.values():
            if isinstance(v, type) and issubclass(
                v, kwiver.vital.algo.algos._algorithm
            ):
                simple_impl_name = "Simple" + v.__name__
                all_impl_names = [sc.__name__ for sc in v.__subclasses__()]
                # Note that we assume the structure of the name of the wrapper module
                if simple_impl_name in all_impl_names:
                    algo_list.append((v, simple_impl_name))
        return algo_list

    def get_fresh_instance(self, abstract_algo, implementation_name):
        return abstract_algo.create(implementation_name)

    def test_registered_names(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            yield self.registered_names_helper, abstract_algo

    def test_bad_create(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            yield self.bad_create_helper, abstract_algo

    def test_create(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            yield self.create_helper, abstract_algo, simple_impl_name

    def test_algo_factory_create(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            yield self.algo_factory_create_helper, abstract_algo, simple_impl_name

    def test_impl_name(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            yield self.impl_helper, abstract_algo

    def test_config(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            instance = self.get_fresh_instance(abstract_algo, simple_impl_name)
            yield self.config_helper, instance

    def test_nested_config(self):
        algo_list = self.get_algo_list()
        for abstract_algo, simple_impl_name in algo_list:
            instance = self.get_fresh_instance(abstract_algo, simple_impl_name)
            yield self.nested_config_helper, instance, abstract_algo

    # Display all the registered implementations of this
    # abstract algorithm
    def registered_names_helper(self, abstract_algo):
        registered_implementation_names = abstract_algo.registered_names()
        print("\nAll registered {}".format(abstract_algo.static_type_name()))
        for name in registered_implementation_names:
            print(" " + name)

    # Test create function
    # For an invalid value it raises RuntimeError
    @nose.tools.raises(RuntimeError)
    def bad_create_helper(self, abstract_algo):
        # Should fail to create an algorithm without a factory
        abstract_algo.create("NonExistantAlgorithm")

    def create_helper(self, abstract_algo, simple_impl_name):
        nose.tools.ok_(
            simple_impl_name in abstract_algo.registered_names(),
            "No simple implementation found for {}".format(
                abstract_algo.static_type_name()
            ),
        )
        algo_out = abstract_algo.create(simple_impl_name)

        nose.tools.ok_(isinstance(algo_out, abstract_algo))

    def algo_factory_create_helper(self, abstract_algo, simple_impl_name):
        nose.tools.ok_(
            algorithm_factory.has_algorithm_impl_name(
                abstract_algo.static_type_name(), simple_impl_name
            ),
            "{} not found by the factory".format(simple_impl_name),
        )

        nose.tools.ok_(
            simple_impl_name
            in algorithm_factory.implementations(abstract_algo.static_type_name()),
            "{} not in implementations list for {}".format(
                simple_impl_name, abstract_algo.static_type_name()
            ),
        )

        algo_out = abstract_algo.create(simple_impl_name)

        nose.tools.ok_(isinstance(algo_out, abstract_algo))

    def config_helper(self, instance):
        instance_cfg = instance.get_configuration()
        # Verify that "threshold" config value is present
        nose.tools.ok_(
            instance_cfg.has_value("threshold"), "threshold config value not present"
        )
        # Verify that the value for key "threshold" is 0.0
        threshold_value = instance_cfg.get_value("threshold")
        nose.tools.ok_(
            threshold_value == "0.0",
            "threshold config value {}, expected 0.0".format(threshold_value),
        )

        test_cfg = _dummy_algorithm_cfg()
        # Verify that the instance has different configuration before setting to test
        nose.tools.ok_(not instance.check_configuration(test_cfg))
        instance.set_configuration(test_cfg)
        # Verify that the config value is being set properly
        nose.tools.ok_(instance.check_configuration(test_cfg))

    def nested_config_helper(self, instance, abstract_algo):
        nested_cfg = config.empty_config()
        instance.get_nested_algo_configuration("algorithm", nested_cfg, instance)

        nose.tools.ok_(
            instance.check_nested_algo_configuration("algorithm", nested_cfg)
        )

        nested_algo = instance.set_nested_algo_configuration("algorithm", nested_cfg)

        # Should have created a concrete algorithm instance
        nose.tools.ok_(type(instance) is type(nested_algo))

        # Verify that the value for key "threshold" is 0.0
        threshold_value = nested_algo.get_configuration().get_value("threshold")
        nose.tools.ok_(
            threshold_value == "0.0",
            "threshold config value {}, expected 0.0".format(threshold_value),
        )

        # Check case where the value for key "type" doesn't match
        # any implementation
        nested_cfg.subblock_view("algorithm").set_value("type", "foo")

        # Check should fail
        nose.tools.assert_false(
            instance.check_nested_algo_configuration("algorithm", nested_cfg)
        )

        # Should get back nullptr
        nose.tools.ok_(
            instance.set_nested_algo_configuration("algorithm", nested_cfg) is None
        )

    def impl_helper(self, instance):
        a = instance.impl_name
        instance.impl_name = "example_impl_name"
        nose.tools.assert_equals(instance.impl_name, "example_impl_name")
