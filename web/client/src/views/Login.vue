<script>
import { Authentication as GirderAuth } from "@girder/components/src/components";

export default {
  name: "Login",
  components: {
    GirderAuth
  },
  inject: ["girderRest"],
  data() {
    return {
      form: "login",
      userDialog: true
    };
  },
  watch: {
    "girderRest.user"(user) {
      if (user) {
        this.$router.push("/");
      }
    }
  }
};
</script>

<template>
  <v-container>
    <v-dialog :value="userDialog" persistent max-width="300px">
      <GirderAuth :register="true" />
    </v-dialog>
  </v-container>
</template>
