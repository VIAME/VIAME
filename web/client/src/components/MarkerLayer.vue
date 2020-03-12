<script>
export default {
  name: "MarkerLayer",
  inject: ["annotator"],
  props: {
    data: {
      type: Array
    },
    markerStyle: {
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
        arr.push(record);
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
    markerStyle() {
      this.updateStyle();
    },
    frameMap() {
      this.frameChanged();
    }
  },
  mounted() {
    var viewer = this.annotator.viewer;
    this.featureLayer = viewer.createLayer("feature", {
      features: ["point"]
    });
    this.pointFeature = this.featureLayer.createFeature("point");
    this.frameChanged();
    this.updateStyle();
  },
  methods: {
    frameChanged() {
      var frame = this.annotator.syncedFrame;
      var data = this.frameMap.get(frame);
      data = data ? data : [];
      this.pointFeature.data(data).draw();
    },
    updateStyle() {
      this.pointFeature.style(this.markerStyle).draw();
    }
  },
  render() {
    return null;
  }
};
</script>
