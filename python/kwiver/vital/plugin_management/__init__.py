from ._plugin_management import (
    plugin_manager_instance,
    SayFactory,
)
from kwiver.vital.util.initial_plugin_path import get_initial_plugin_path

# add the default path of plugins, this allows wheels to work without needing
# to set the KWIVER_PLUGIN_PATH environmental. Also, it sets the environmental
# variable free for the user to modify without worrying for its original value.
plugin_manager_instance().add_search_path(get_initial_plugin_path())
