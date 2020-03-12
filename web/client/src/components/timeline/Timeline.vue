<script>
import { throttle } from "lodash";
import * as d3 from "d3";

export default {
  name: "Timeline",
  props: {
    maxFrame: {
      type: Number,
      default: 0
    },
    frame: {
      type: Number,
      default: 0
    },
    seek: {
      type: Function
    }
  },
  data() {
    return {
      init: this.maxFrame ? true : false,
      mounted: false,
      startFrame: 0,
      endFrame: this.maxFrame,
      timelineScale: null
    };
  },
  computed: {
    minimapFillStyle() {
      return {
        left: (this.startFrame / this.maxFrame) * 100 + "%",
        width: ((this.endFrame - this.startFrame) / this.maxFrame) * 100 + "%"
      };
    },
    handLeftPosition() {
      if (
        !this.mounted ||
        this.frame < this.startFrame ||
        this.frame > this.endFrame
      ) {
        return null;
      }
      return Math.round(
        this.$refs.workarea.clientWidth *
          ((this.frame - this.startFrame) / (this.endFrame - this.startFrame))
      );
    }
  },
  watch: {
    maxFrame(value) {
      this.endFrame = value;
      this.init = true;
      this.update();
    },
    startFrame() {
      this.update();
    },
    endFrame() {
      this.update();
    },
    handLeftPosition(value) {
      this.$refs.hand.style.left = (value ? value : "-10") + "px";
    },
    frame(frame) {
      if (frame > this.endFrame) {
        this.endFrame = Math.min(frame + 200, this.maxFrame);
      } else if (frame < this.startFrame) {
        this.startFrame = Math.max(frame - 100, 0);
      }
    }
  },
  created() {
    this.update = throttle(this.update, 30);
  },
  mounted() {
    var width = this.$refs.workarea.clientWidth;
    var height = this.$refs.workarea.clientHeight;
    var scale = d3
      .scaleLinear()
      .domain([0, this.maxFrame])
      .range([0, width]);
    this.timelineScale = scale;
    var axis = d3
      .axisTop()
      .scale(scale)
      .tickSize(height - 30)
      .tickSizeOuter(0);
    this.axis = axis;
    this.g = d3
      .select(this.$refs.workarea)
      .append("svg")
      .style("display", "block")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(0,${height - 17})`);
    this.updateAxis();
    this.mounted = true;
  },
  methods: {
    onwheel(e) {
      var extend =
        Math.round((this.endFrame - this.startFrame) * 0.2) *
        Math.sign(e.deltaY);
      var ratio = (e.layerX - this.$el.offsetLeft) / this.$el.clientWidth;
      var startFrame = this.startFrame - extend * ratio;
      var endFrame = this.endFrame + extend * (1 - ratio);
      startFrame = Math.max(0, startFrame);
      endFrame = Math.min(this.maxFrame, endFrame);
      if (startFrame >= endFrame - 200) {
        return;
      }
      this.startFrame = startFrame;
      this.endFrame = endFrame;
    },
    updateAxis() {
      this.g.call(this.axis).call(g =>
        g
          .selectAll(".tick text")
          .attr("y", 0)
          .attr("dy", 13)
      );
    },
    update() {
      this.timelineScale.domain([this.startFrame, this.endFrame]);
      this.axis.scale(this.timelineScale);
      this.updateAxis();
    },
    emitSeek(e) {
      var frame = Math.round(
        ((e.clientX - this.$refs.workarea.getBoundingClientRect().left) /
          this.$refs.workarea.clientWidth) *
          (this.endFrame - this.startFrame) +
          this.startFrame
      );
      this.seek(frame);
    },
    workareaMouseup(e) {
      if (this.dragging) {
        this.emitSeek(e);
      }
      this.dragging = false;
    },
    workareaMousedown() {
      this.dragging = true;
      // e.preventDefault();
    },
    workareaMousemove(e) {
      if (this.dragging) {
        this.emitSeek(e);
      }
      e.preventDefault();
    },
    workareaMouseleave() {
      this.dragging = false;
    },
    minimapFillMousedown(e) {
      e.preventDefault();
      this.minimapDragging = true;
      this.minimapDraggingStartClientX = e.clientX;
      this.minimapDraggingStartFrame = this.startFrame;
      this.minimapDraggingEndFrame = this.endFrame;
    },
    containerMousemove(e) {
      e.preventDefault();
      if (!this.minimapDragging) {
        return;
      }
      if (!e.which) {
        this.minimapDragging = false;
        return;
      }
      var delta = this.minimapDraggingStartClientX - e.clientX;
      var frameDelta = (delta / this.$refs.minimap.clientWidth) * this.maxFrame;
      var startFrame = this.minimapDraggingStartFrame - frameDelta;
      if (startFrame < 0) {
        return;
      }
      var endFrame = this.minimapDraggingEndFrame - frameDelta;
      if (endFrame > this.maxFrame) {
        return;
      }
      this.startFrame = startFrame;
      this.endFrame = endFrame;
    },
    containerMouseup() {
      this.minimapDragging = false;
    }
    // rendered() {
    // console.log("timeline rendered");
    // }
  }
};
</script>

<template>
  <div
    class="timeline"
    @wheel="onwheel"
    @mouseup="containerMouseup"
    @mousemove="containerMousemove"
  >
    <div
      class="work-area"
      ref="workarea"
      @mouseup="workareaMouseup"
      @mousedown="workareaMousedown"
      @mousemove="workareaMousemove"
      @mouseleave="workareaMouseleave"
    >
      <div class="hand" ref="hand"></div>
      <div class="child" v-if="init && mounted">
        <slot
          name="child"
          :startFrame="startFrame"
          :endFrame="endFrame"
          :maxFrame="maxFrame"
        />
      </div>
    </div>
    <div class="minimap" ref="minimap">
      <div
        class="fill"
        :style="minimapFillStyle"
        @mousedown="minimapFillMousedown"
      >
        <!-- {{ rendered() }} -->
      </div>
    </div>
    <slot />
  </div>
</template>

<style lang="scss" scoped>
.timeline {
  min-height: 175px;
  position: relative;
  display: flex;
  flex-direction: column;

  .work-area {
    flex: 1;
    position: relative;
    overflow: hidden;

    .hand {
      position: absolute;
      top: 0;
      width: 0;
      height: 100%;
      border-left: 1px solid #299be3;
    }

    .child {
      position: absolute;
      top: 0;
      bottom: 17px;
      left: 0;
      right: 0;
    }
  }

  .minimap {
    height: 10px;

    .fill {
      position: relative;
      height: 100%;
      background-color: #80c6e8;
    }
  }
}
</style>

<style lang="scss">
.timeline {
  .tick {
    shape-rendering: crispEdges;
    font-size: 12px;
    stroke-opacity: 0.5;
    stroke-dasharray: 2, 2;
  }
}
</style>
