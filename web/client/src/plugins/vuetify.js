import Vue from "vue";
import Vuetify from "vuetify/lib";
import colors from "vuetify/lib/util/colors";
import vuetifyConfig from "@girder/components/src/utils/vuetifyConfig.js";

import "@mdi/font/css/materialdesignicons.css";

Vue.use(Vuetify);

vuetifyConfig.theme.dark = true;
vuetifyConfig.theme.themes.dark = {
  ...vuetifyConfig.theme.themes.dark,
  ...{ accent: colors.blue.lighten3 }
};

export default new Vuetify(vuetifyConfig);
