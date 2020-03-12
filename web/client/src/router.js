import Vue from "vue";
import Router from "vue-router";

import girder from "./girder";
import Viewer from "./views/Viewer.vue";
import Home from "./views/Home.vue";
import Jobs from "./views/Jobs.vue";
import Login from "./views/Login.vue";
import Settings from "./views/Settings.vue";

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: "/login",
      name: "login",
      component: Login
    },
    {
      path: "/jobs",
      name: "jobs",
      component: Jobs,
      beforeEnter
    },
    {
      path: "/settings",
      name: "settings",
      component: Settings,
      beforeEnter
    },
    {
      path: "/viewer/:datasetId?",
      name: "viewer",
      component: Viewer,
      beforeEnter
    },
    {
      path: "/:_modelType?/:_id?",
      name: "home",
      component: Home,
      beforeEnter
    }
  ]
});

function beforeEnter(to, from, next) {
  if (!girder.girderRest.user) {
    next("/login");
  } else {
    next();
  }
}
