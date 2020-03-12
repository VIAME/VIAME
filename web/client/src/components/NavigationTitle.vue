<script>
export default {
  name: "NavigationTitle",
  computed: {
    version() {
      return process.env.VERSION;
    }
  }
};
</script>

<template>
  <v-toolbar-title class="my-title mx-5">
    <v-tooltip open-delay="1000" bottom>
      <template #activator="{ on }">
        <span v-on="on">VIAME</span>
      </template>
      <span>v{{ version }}</span>
    </v-tooltip>
  </v-toolbar-title>
</template>

<style lang="scss" scoped>
.my-title {
  overflow: unset;
}
</style>
