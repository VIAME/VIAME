<script>
import { throttle } from "lodash";
import * as d3 from "d3";

export default {
  name: "LineChart",
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
          return !Array.isArray(datum.values);
        });
      }
    }
  },
  computed: {
    lineData() {
      return this.data.map(datum => {
        var lastFrame = -1;
        // var lastPoint = [0, 0];
        var padZero = [];
        datum.values.forEach(point => {
          var frame = point[0];
          if (frame != lastFrame + 1) {
            for (var i = lastFrame + 1; i < frame; i++) {
              padZero.push([i, 0]);
            }
          }
          padZero.push(point);
          lastFrame = frame;
        });
        if (this.maxFrame != lastFrame) {
          for (var i = lastFrame + 1; i <= this.maxFrame; i++) {
            padZero.push([i, 0]);
          }
        }
        var clean = [padZero[0]];
        var lastValue = padZero[0][1];
        for (let i = 1; i < padZero.length; i++) {
          if (padZero[i][1] != lastValue) {
            clean.push(padZero[i - 1]);
            clean.push(padZero[i]);
            lastValue = padZero[i][1];
          }
        }
        if (clean.slice(-1)[0][0] !== this.maxFrame) {
          clean.push(padZero.slice(-1)[0]);
        }
        return { ...datum, values: clean };
      });
    }
  },
  watch: {
    startFrame() {
      this.update();
    },
    endFrame() {
      this.update();
    },
    lineData() {
      this.initialize();
      this.update();
    }
  },
  created() {
    this.update = throttle(this.update, 30);
  },
  mounted() {
    this.initialize();
  },
  methods: {
    initialize() {
      d3.select(this.$el)
        .select("svg")
        .remove();
      var width = this.$el.clientWidth;
      var height = this.$el.clientHeight;
      var x = d3
        .scaleLinear()
        .domain([this.startFrame, this.endFrame])
        .range([0, width]);
      this.x = x;
      var max = d3.max(this.lineData, datum => d3.max(datum.values, d => d[1]));
      var y = d3
        .scaleLinear()
        .domain([0, Math.max(max + max * 0.2, 2)])
        .range([height, 0]);

      var line = d3
        .line()
        .curve(d3.curveStepAfter)
        .x(d => x(d[0]))
        .y(d => y(d[1]));
      this.line = line;

      var svg = d3
        .select(this.$el)
        .append("svg")
        .style("display", "block")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(0,-1)`);

      var axis = d3.axisRight(y).tickSize(width);
      this.axis = axis;
      svg
        .append("g")
        .attr("class", "axis-y")
        .call(axis)
        .call(g =>
          g
            .selectAll(".tick text")
            .attr("x", 0)
            .attr("dx", 13)
        );

      var path = svg
        .selectAll()
        .data(this.lineData)
        .enter()
        .append("path")
        .attr("class", "line")
        .attr("d", d => line(d.values))
        .style("stroke", d => (d.color ? d.color : "#4c9ac2"))
        .on("mouseenter", function(d) {
          var [x, y] = d3.mouse(this);
          tooltipTimeoutHandle = setTimeout(() => {
            tooltip
              .style("left", x + 2 + "px")
              .style("top", y - 25 + "px")
              .text(d.name)
              .style("display", "block");
          }, 200);
        })
        .on("mouseout", function() {
          clearTimeout(tooltipTimeoutHandle);
          tooltip.style("display", "none");
        });
      this.path = path;

      var tooltip = d3
        .select(this.$el)
        .append("div")
        .attr("class", "tooltip")
        .style("display", "none");
      var tooltipTimeoutHandle = null;
    },
    update() {
      this.x.domain([this.startFrame, this.endFrame]);
      this.line.x(d => {
        return this.x(d[0]);
      });
      this.path.attr("d", d => this.line(d.values));
    },
    rendered() {
      console.log("linechart rendered");
    }
  }
};
</script>

<template>
  <div class="line-chart">{{ rendered() }}</div>
</template>

<style lang="scss">
.line-chart {
  height: 100%;

  .line {
    fill: none;
    stroke-width: 1.5px;
  }

  .axis-y {
    font-size: 12px;

    g:first-of-type,
    g:last-of-type {
      display: none;
    }
  }

  .tooltip {
    position: absolute;
    background: black;
    border: 1px solid white;
    padding: 0px 5px;
    font-size: 14px;
  }
}
</style>
