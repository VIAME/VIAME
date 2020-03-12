<script>
export default {
  name: "TimelineWrapper",
  inject: ["annotator"],
  methods: {
    seek(frame) {
      this.annotator.$emit("seek", frame);
    }
  }
};
</script>

<template>
  <div>
    <slot
      :maxFrame="annotator.maxFrame"
      :frame="annotator.frame"
      :seek="seek"
    />
  </div>
</template>
