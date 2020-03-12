<script>
// import { mapState, mapMutations } from "vuex";

import NavigationBar from "@/components/NavigationBar";
import JobList from "@girder/components/src/components/Job/JobList";

export default {
  name: "Jobs",
  components: { NavigationBar, JobList },
  // inject: ["girderRest"],
  data: () => ({}),
  computed: {},
  methods: {}
};
</script>

<template>
  <v-content>
    <NavigationBar />
    <v-container>
      <JobList />
    </v-container>
  </v-content>
</template>
