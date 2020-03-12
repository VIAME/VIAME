<script>
import geo from "geojs";

export default {
  name: "AnnotationLayer",
  inject: ["annotator"],
  props: {
    data: {
      type: Array,
      validator(data) {
        if (!Array.isArray(data)) {
          return false;
        }
        if (data.find(item => !Number.isInteger(item.frame) || !item.polygon)) {
          return false;
        }
        return true;
      }
    },
    annotationStyle: {
      type: Object,
      required: false
    }
  },
  computed: {
    frameMap() {
      var map = new Map();
      this.data.forEach(record => {
        let arr = map.get(record.frame);
        if (!map.has(record.frame)) {
          arr = [];
          map.set(record.frame, arr);
        }
        var coords = record.polygon.coordinates[0];
        arr.push({
          record,
          geometry: {
            outer: [
              { x: coords[0][0], y: coords[0][1] },
              { x: coords[1][0], y: coords[1][1] },
              { x: coords[2][0], y: coords[2][1] },
              { x: coords[3][0], y: coords[3][1] }
            ]
          }
        });
      });
      return map;
    }
  },
  watch: {
    "annotator.syncedFrame": {
      sync: true,
      handler() {
        this.frameChanged();
      }
    },
    annotationStyle() {
      this.updateStyle();
    },
    frameMap() {
      this.frameChanged();
    }
  },
  mounted() {
    // console.log('mounted');
    // console.log(this.annotator.viewer);
    var viewer = this.annotator.viewer;
    this.featureLayer = viewer.createLayer("feature", {
      features: ["point", "line", "polygon"]
    });
    this.polygonFeature = this.featureLayer
      .createFeature("polygon", { selectionAPI: true })
      .geoOn(geo.event.feature.mouseclick, e => {
        if (e.mouse.buttonsDown.left) {
          this.$emit("annotation-click", e.data.record, e);
        } else if (e.mouse.buttonsDown.right) {
          this.$emit("annotation-right-click", e.data.record, e);
        }
      });
    this.polygonFeature.geoOn(
      geo.event.feature.mouseclick_order,
      this.polygonFeature.mouseOverOrderClosestBorder
    );
    this.frameChanged();
    this.updateStyle();
  },
  beforeDestroy() {
    this.annotator.viewer.removeChild(this.featureLayer);
  },
  methods: {
    updateStyle() {
      var style = {
        ...{
          stroke: true,
          uniformPolygon: true,
          strokeColor: "lime",
          strokeWidth: 1,
          fill: false
        },
        ...this.annotationStyle
      };
      this.polygonFeature.style(style).draw();
    },
    frameChanged() {
      var frame = this.annotator.syncedFrame;
      var data = this.frameMap.get(frame);
      data = data ? data : [];
      this.polygonFeature
        .data(data)
        .polygon(data => data.geometry)
        .draw();
    }
  },
  render() {
    return null;
  }
};
</script>
