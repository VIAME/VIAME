import kwiver
import os

def get_initial_plugin_path():
    kwiver_module_path = os.path.dirname(os.path.abspath(kwiver.__file__))
    plugin_path = os.path.join(kwiver_module_path, 'lib', 'kwiver', 'modules')
    return plugin_path
