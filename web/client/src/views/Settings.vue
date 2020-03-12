<script>
// import { mapState, mapMutations } from "vuex";

import NavigationBar from "@/components/NavigationBar";
import AttributeSettings from "@/components/AttributeSettings";

export default {
  name: "Config",
  components: { NavigationBar, AttributeSettings },
  inject: ["girderRest"],
  data: () => ({}),
  computed: {},
  methods: {}
};
</script>

<template>
  <v-content>
    <NavigationBar />
    <v-container>
      <AttributeSettings />
    </v-container>
  </v-content>
</template>
