<script>
import { cloneDeep } from "lodash";
// import { getType } from '@turf/invariant';
// import { feature } from '@turf/helpers';
import geo from "geojs";

export default {
  name: "PointMarkingLayer",
  inject: ["annotator"],
  props: {
    geojson: {
      type: Object,
      default: null,
      validator(value) {
        return value.type === "Polygon";
      }
    },
    editing: {
      type: Boolean,
      default: true
    },
    featureStyle: {
      type: Object,
      required: false
    }
  },
  data() {
    return {
      changed: false
    };
  },
  watch: {
    geojson() {
      this.reinitialize();
    },
    editing() {
      this.reinitialize();
    }
  },
  mounted() {
    this.initialize();
  },
  beforeDestroy() {
    this.$geojsLayer.mode(null);
    this.annotator.viewer.deleteLayer(this.$geojsLayer);
    delete this.$geojsLayer;
  },
  methods: {
    reinitialize() {
      this.$geojsLayer.geoOff(geo.event.annotation.mode);
      this.$geojsLayer.geoOff(geo.event.annotation.state);
      this.$geojsLayer.geoOff(geo.event.annotation.edit_action);
      this.$geojsLayer.removeAllAnnotations();
      this.$geojsLayer.mode(null);
      this.initialize();
    },
    initialize() {
      if (!this.$geojsLayer) {
        this.$geojsLayer = this.annotator.viewer.createLayer("annotation", {
          clickToEdit: true,
          showLabels: false
        });
        // this.listenLayerClick();
      }
      if (this.geojson) {
        let geojson = cloneDeep(this.geojson);
        if (!("geometry" in geojson)) {
          geojson = { type: "Feature", geometry: geojson, properties: {} };
        }
        // Always is rectangle
        geojson.properties.annotationType = "rectangle";
        this.$geojsLayer.geojson(geojson);
        const annotation = this.$geojsLayer.annotations()[0];
        if (this.featureStyle) {
          annotation.style(this.featureStyle);
        }
        annotation.editHandleStyle({ handles: { rotate: false } });
        if (this.editing) {
          this.$geojsLayer.mode("edit", annotation);
          this.$geojsLayer.draw();
        }
      } else if (this.editing) {
        this.changed = true;
        this.$geojsLayer.mode("rectangle");
      }

      this.$geojsLayer.geoOn(geo.event.annotation.mode, e => {
        this.$emit("update:editing", e.mode === "edit" ? true : e.mode);
      });

      this.$geojsLayer.geoOn(geo.event.annotation.state, e => {
        if (this.changed) {
          this.changed = false;
          const newGeojson = e.annotation.geojson();
          let geojson = cloneDeep(this.geojson);
          if (geojson) {
            if ("geometry" in geojson) {
              geojson.geometry.coordinates = newGeojson.geometry.coordinates;
            } else {
              geojson.coordinates = newGeojson.geometry.coordinates;
            }
          } else {
            geojson = {
              ...newGeojson,
              ...{
                properties: {
                  annotationType: newGeojson.properties.annotationType
                }
              }
            };
          }
          this.$emit("update:geojson", geojson);
        }
      });

      this.$geojsLayer.geoOn(geo.event.annotation.edit_action, e => {
        if (e.action === geo.event.actionmove) {
          if (this.$listeners["being-edited-geojson"]) {
            this.$emit("being-edited-geojson", e.annotation.geojson().geometry);
          }
        }
        if (e.action === geo.event.actionup) {
          this.$emit("being-edited-geojson", null);
          this.changed = true;
        }
      });
    }
  },
  render() {
    return null;
  }
};
</script>
