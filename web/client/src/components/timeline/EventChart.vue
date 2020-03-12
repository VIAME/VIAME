<script>
import { throttle, debounce, sortBy } from "lodash";
import * as d3 from "d3";

export default {
  name: "EventChart",
  props: {
    startFrame: {
      type: Number,
      required: true
    },
    endFrame: {
      type: Number,
      required: true
    },
    maxFrame: {
      type: Number,
      required: true
    },
    data: {
      type: Array,
      required: true,
      validator(data) {
        return !data.find(datum => {
          return !datum.range;
        });
      }
    }
  },
  data() {
    return {
      x: null,
      tooltip: null,
      startFrame_: this.startFrame,
      endFrame_: this.endFrame
    };
  },
  computed: {
    barData() {
      var sorted = sortBy(this.data, data => data.range[0]);
      var bars = [];
      sorted.forEach(event => {
        for (let bar of bars) {
          if (bar.endPosition < event.range[0]) {
            bar.events.push(event);
            bar.endPosition = event.range[1];
            return;
          }
        }
        bars.push({
          startPoisiton: event.range[0],
          endPosition: event.range[1],
          events: [event]
        });
      });
      return bars;
    },
    bars() {
      if (!this.x) {
        return [];
      }
      var startFrame_ = this.startFrame_;
      var endFrame_ = this.endFrame_;
      var x = this.x;
      var bars = [];
      this.barData
        .filter(barData =>
          intersect(
            [startFrame_, endFrame_],
            [barData.startPoisiton, barData.endPosition]
          )
        )
        .forEach((barData, i) => {
          barData.events
            .filter(event =>
              intersect(
                [startFrame_, endFrame_],
                [event.range[0], event.range[1]]
              )
            )
            .forEach(event => {
              var left = x(event.range[0]);
              bars.push({
                left: x(event.range[0]),
                width: Math.max(x(event.range[1]) - left, 3),
                top: i * 15,
                color: event.color,
                name: event.name
              });
            });
        });
      return bars;
    }
  },
  watch: {
    startFrame() {
      this.update();
    },
    endFrame() {
      this.update();
    },
    data() {
      this.update();
    }
  },
  created() {
    this.update = throttle(this.update, 20);
    this.detectBarHovering = debounce(this.detectBarHovering, 300);
    this.tooltipTimeoutHandle = null;
  },
  mounted() {
    this.initialize();
    this.update();
  },
  methods: {
    initialize() {
      var width = this.$el.clientWidth;
      var x = d3
        .scaleLinear()
        .domain([this.startFrame_, this.endFrame_])
        .range([0, width]);
      this.x = x;
    },
    update() {
      this.startFrame_ = this.startFrame;
      this.endFrame_ = this.endFrame;
      this.x.domain([this.startFrame_, this.endFrame_]);
      var canvas = this.$refs.canvas;
      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var bars = this.bars;
      if (!bars.length) {
        return;
      }
      canvas.width = this.$el.clientWidth;
      canvas.height = bars.slice(-1)[0].top + 10;
      for (var bar of bars) {
        ctx.fillStyle = bar.color ? bar.color : "#4c9ac2";
        ctx.fillRect(bar.left, bar.top, bar.width, 10);
      }
    },
    mousemove(e) {
      this.tooltip = null;
      this.detectBarHovering(e);
    },
    mouseout() {
      this.detectBarHovering.cancel();
    },
    detectBarHovering(e) {
      var { offsetX, offsetY } = e;
      var remainder = offsetY % 15;
      if (remainder > 10) {
        return;
      }
      var top = offsetY - (offsetY % 15);
      var bar = this.bars
        .filter(bar => bar.top === top)
        .reverse()
        .find(bar => bar.left < offsetX && bar.left + bar.width > offsetX);
      if (!bar) {
        return;
      }
      this.tooltip = {
        left: offsetX,
        top: offsetY,
        content: bar.name
      };
    }
  }
};

function intersect(range1, range2) {
  var min = range1[0] < range2[0] ? range1 : range2;
  var max = min == range1 ? range2 : range1;
  if (min[1] < max[0]) {
    return null;
  }
  return [max[0], min[1] < max[1] ? min[1] : max[1]];
}
</script>

<template>
  <div class="event-chart" @mousewheel.prevent>
    <canvas ref="canvas" @mousemove="mousemove" @mouseout="mouseout" />
    <div
      class="tooltip"
      v-if="tooltip"
      :style="{ left: tooltip.left + 15 + 'px', top: tooltip.top + 5 + 'px' }"
    >
      {{ tooltip.content }}
    </div>
  </div>
</template>

<style lang="scss">
.event-chart {
  position: relative;
  height: calc(100% - 10px);
  margin: 5px 0;
  overflow-y: auto;
  overflow-x: hidden;

  .tooltip {
    position: absolute;
    background: black;
    border: 1px solid white;
    padding: 0px 5px;
    font-size: 14px;
    z-index: 1;
  }
}
</style>
