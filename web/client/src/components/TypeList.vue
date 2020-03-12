<script>
export default {
  name: "TypeList",
  components: {},
  props: {
    types: {
      type: Array
    },
    checkedTypes: {
      type: Array
    },
    colorMap: {
      type: Function,
      required: false
    }
  },
  data: function() {
    return { checkedTypes_: this.checkedTypes };
  },
  watch: {
    checkedTypes(value) {
      this.checkedTypes_ = value;
    },
    checkedTypes_(value) {
      this.$emit("update:checkedTypes", value);
    }
  }
};
</script>

<template>
  <div class="typelist d-flex flex-column">
    <v-subheader>Types</v-subheader>
    <div class="flex-grow-1" style="overflow-y: auto;">
      <div>
        <v-checkbox
          v-for="type of types"
          :key="type"
          class="my-2 ml-3"
          v-model="checkedTypes_"
          :value="type"
          dense
          hide-details
        >
          <template slot="label">
            <div>
              <span class="color" :style="{ backgroundColor: colorMap(type) }"
                >&nbsp;&nbsp;</span
              >&nbsp;
              <span>{{ type }}</span>
            </div>
          </template></v-checkbox
        >
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.typelist {
  overflow-y: auto;
}
</style>
