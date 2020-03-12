import Vue from "vue";
import Girder, { RestClient } from "@girder/components/src";
import NotificationBus from "@girder/components/src/utils/notifications";
import snackbarService from "vue-utilities/snackbar-service";
import promptService from "vue-utilities/prompt-service";
import vMousetrap from "vue-utilities/v-mousetrap";

import vuetify from "@/plugins/vuetify.js";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import girder from "./girder";

Vue.config.productionTip = false;

Vue.use(Girder);
Vue.use(snackbarService(vuetify));
Vue.use(promptService(vuetify));
Vue.use(vMousetrap);

var girderRest = new RestClient({ apiRoot: "api/v1" });
girder.girderRest = girderRest;
const notificationBus = new NotificationBus(girderRest);
notificationBus.connect();

girderRest.fetchUser().then(() => {
  new Vue({
    router,
    store,
    vuetify,
    provide: { girderRest, notificationBus },
    render: h => h(App)
  })
    .$mount("#app")
    .$snackbarAttach()
    .$promptAttach();
});
