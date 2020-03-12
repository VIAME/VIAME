<script>
import AttributeInput from "@/components/AttributeInput";

export default {
  name: "AttributesPanel",
  components: {
    AttributeInput
  },
  inject: ["girderRest"],
  props: {
    selectedDetection: {
      type: Object
    },
    selectedTrack: {
      type: Object
    }
  },
  computed: {
    trackAttributes() {
      if (!this.attributes) {
        return [];
      }
      return this.attributes.filter(attribute => attribute.belongs === "track");
    },
    detectionAttributes() {
      if (!this.attributes) {
        return [];
      }
      return this.attributes.filter(
        attribute => attribute.belongs === "detection"
      );
    }
  },
  asyncComputed: {
    async attributes() {
      var { data } = await this.girderRest.get("/viame/attribute");
      return data;
    }
  },
  watch: {}
};
</script>

<template>
  <v-row class="attributes-panel flex-column" no-gutters>
    <v-col class="" style="overflow-y: auto;">
      <v-subheader>Track attributes</v-subheader>
      <div v-if="!selectedTrack" class="ml-4 body-2">No track selected</div>
      <template v-else>
        <div class="mx-4 body-2" style="line-height:22px;">
          <div>Track ID: {{ selectedTrack.trackId }}</div>
          <div>
            Confidence pairs:
            <div
              class="ml-3"
              v-for="(pair, index) in selectedTrack.confidencePairs"
              :key="index"
            >
              {{ pair[0] }}: {{ pair[1].toFixed(2) }}
            </div>
          </div>
          <AttributeInput
            v-for="(attribute, i) of trackAttributes"
            :key="i"
            :datatype="attribute.datatype"
            :name="attribute.name"
            :values="attribute.values ? attribute.values : null"
            :value="
              selectedTrack.trackAttributes
                ? selectedTrack.trackAttributes[attribute.name]
                : undefined
            "
            @change="$emit('change', { ...$event, ...{ type: 'track' } })"
          ></AttributeInput>
        </div>
      </template>
    </v-col>
    <v-col style="overflow-y: auto;">
      <v-subheader>Detection attributes</v-subheader>
      <div v-if="!selectedDetection" class="ml-4 body-2">
        No detection selected
      </div>
      <template v-else>
        <div class="mx-4 body-2" style="line-height:22px;">
          <div>Frame: {{ selectedDetection.frame }}</div>
          <div>Confidence: {{ selectedDetection.confidence }}</div>
          <div>Fish length: {{ selectedDetection.fishLength }}</div>
          <AttributeInput
            v-for="(attribute, i) of detectionAttributes"
            :key="i"
            :datatype="attribute.datatype"
            :name="attribute.name"
            :values="attribute.values ? attribute.values : null"
            :value="
              selectedDetection.attributes
                ? selectedDetection.attributes[attribute.name]
                : undefined
            "
            @change="$emit('change', { ...$event, ...{ type: 'detection' } })"
          ></AttributeInput>
        </div>
      </template>
    </v-col>
  </v-row>
</template>

<style lang="scss" scoped>
.attributes-panel {
}
</style>
