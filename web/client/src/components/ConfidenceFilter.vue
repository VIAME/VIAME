<script>
import _ from "lodash";

export default {
  name: "ConfidenceFilter",
  props: {
    confidence: {
      type: Number
    }
  },
  created() {
    this.updateConfidence = _.throttle(this.updateConfidence, 100);
  },
  methods: {
    updateConfidence(value) {
      this.$emit("update:confidence", value);
    }
  }
};
</script>

<template>
  <div class="confidence-filter">
    <v-slider
      :min="0"
      :max="1"
      :step="0.01"
      :value="confidence"
      @input="updateConfidence"
      :label="`Confidence Filter: ${confidence.toFixed(2)}`"
      hide-details
    />
  </div>
</template>
