import Vue from "vue";
import Girder, { RestClient } from "@girder/components/src";

// Install the Vue plugin that lets us use the components
Vue.use(Girder);

// This connects to another server if the VUE_APP_API_ROOT
// environment variable is set at build-time
const apiRoot = process.env.VUE_APP_API_ROOT || "http://localhost:8080/api/v1";

// Create the axios-based client to be used for all API requests
const girderRest = new RestClient({
  apiRoot
});

// This is passed to our Vue instance; it will be available in all components
const girderProvider = {
  girderRest
};
export default girderProvider;
