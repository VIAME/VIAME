<script>
import { mapState } from "vuex";
import { all } from "@girder/components/src/components/Job/status";

import NavigationTitle from "@/components/NavigationTitle";
import { getPathFromLocation } from "@/utils";

export default {
  name: "GenericNavigationBar",
  components: {
    NavigationTitle
  },
  inject: ["girderRest", "notificationBus"],
  data: () => ({
    runningJobIds: []
  }),
  computed: {
    ...mapState(["location"])
  },
  async created() {
    let jobStatus = all();
    let { data: runningJobs } = await this.girderRest.get("/job", {
      params: {
        statuses: `[${jobStatus.RUNNING.value}]`
      }
    });
    this.runningJobIds = runningJobs.map(job => job._id);

    this.notificationBus.$on("message:job_status", ({ data: job }) => {
      let jobId = job._id;
      switch (job.status) {
        case jobStatus.RUNNING.value:
          if (this.runningJobIds.indexOf(jobId) === -1) {
            this.runningJobIds.push(jobId);
          }
          break;
        case jobStatus.SUCCESS.value:
        case jobStatus.ERROR.value:
          if (this.runningJobIds.indexOf(jobId) !== -1) {
            this.runningJobIds.splice(this.runningJobIds.indexOf(jobId), 1);
          }
          break;
      }
    });
  },
  methods: { getPathFromLocation }
};
</script>

<template>
  <v-app-bar app>
    <NavigationTitle>VIAME</NavigationTitle>
    <v-tabs icons-and-text color="accent">
      <v-tab :to="getPathFromLocation(location)"
        >Data<v-icon>mdi-database</v-icon></v-tab
      >
      <v-tab to="/jobs">
        Jobs
        <v-badge :value="runningJobIds.length" class="my-badge">
          <template slot="badge">
            <v-icon dark class="rotate">mdi-autorenew</v-icon>
          </template>
          <v-icon>mdi-format-list-checks</v-icon>
        </v-badge>
      </v-tab>
      <v-tab to="/settings">Settings<v-icon>mdi-settings</v-icon></v-tab>
    </v-tabs>
    <v-spacer></v-spacer>
    <v-btn text @click="girderRest.logout()">Logout</v-btn>
  </v-app-bar>
</template>

<style lang="scss">
.rotate {
  animation: rotation 1.5s infinite linear;
}

@keyframes rotation {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(359deg);
  }
}

.my-badge .v-badge__badge {
  top: -8px;
}
</style>
