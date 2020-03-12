<script>
import VirtualList from "vue-virtual-scroll-list";
import TrackItem from "./TrackItem";

// A monkey patch
VirtualList.options.props.item.type = [Object, Function];

export default {
  name: "Tracks",
  components: {
    VirtualList
  },
  props: {
    tracks: {
      type: Array
    },
    types: {
      type: Array
    },
    checkedTracks: {
      type: Array
    },
    selectedTrack: {
      type: Number
    },
    editingTrack: {
      type: Number
    }
  },
  data: function() {
    return { checkedTracks_: this.checkedTracks, item: TrackItem };
  },
  watch: {
    checkedTracks(value) {
      this.checkedTracks_ = value;
    },
    checkedTracks_(value) {
      this.$emit("update:checkedTracks", value);
    }
  },
  methods: {
    getItemProps(itemIndex) {
      var track = this.tracks[itemIndex];
      return {
        props: {
          track,
          inputValue: this.checkedTracks_.indexOf(track.trackId) !== -1,
          selectedTrack: this.selectedTrack,
          editingTrack: this.editingTrack,
          types: this.types
        },
        on: {
          change: checked => {
            if (checked) {
              this.checkedTracks_.push(track.trackId);
            } else {
              var index = this.checkedTracks_.indexOf(track.trackId);
              this.checkedTracks_.splice(index, 1);
            }
          },
          "type-change": type => {
            this.$emit("track-type-change", track, type);
          },
          "goto-first-frame": () => {
            this.$emit("goto-track-first-frame", track);
          },
          delete: () => {
            this.$emit("delete-track", track);
          },
          click: () => {
            this.$emit("click-track", track);
          },
          edit: () => {
            this.$emit("edit-track", track);
          }
        }
      };
    }
  }
};
</script>

<template>
  <div class="tracks">
    <v-subheader
      >Tracks<v-spacer /><v-btn icon @click="$emit('add-track')"
        ><v-icon>mdi-plus</v-icon></v-btn
      ></v-subheader
    >
    <virtual-list
      :size="45"
      :remain="9"
      :item="item"
      :itemcount="tracks.length"
      :itemprops="getItemProps"
    >
    </virtual-list>
  </div>
</template>

<style lang="scss">
.tracks {
  overflow-y: auto;
  padding: 4px 0;

  .v-input--checkbox {
    label {
      white-space: pre-wrap;
    }
  }
}
</style>
