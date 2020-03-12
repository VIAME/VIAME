<script>
import { throttle } from "lodash";
import Vue from "vue";
import geo from "geojs";

// TODO: have a common base class with VideoAnnotator

export default {
  name: "ImageAnnotator",
  props: {
    imageUrls: {
      type: Array,
      required: true
    },
    frameRate: {
      type: Number,
      required: true
    }
  },
  provide() {
    return {
      annotator: this.provided
    };
  },
  data() {
    this.provided = new Vue({
      computed: {
        viewer: () => this.viewer,
        playing: () => this.playing,
        frame: () => this.frame,
        maxFrame: () => this.maxFrame,
        syncedFrame: () => this.syncedFrame
      }
    });
    return {
      ready: false,
      playing: false,
      frame: 0,
      maxFrame: 0,
      syncedFrame: 0
    };
  },
  created() {
    this.provided.$on("play", this.play);
    this.provided.$on("prev-frame", this.prevFrame);
    this.provided.$on("next-frame", this.nextFrame);
    this.provided.$on("pause", this.pause);
    this.provided.$on("seek", this.seek);
    this.emitFrame();
    this.emitFrame = throttle(this.emitFrame, 200);
    this.maxFrame = this.imageUrls.length - 1;
    this.imgs = new Array(this.imageUrls.length);
    this.pendingImgs = new Set();
    this.cacheImage();
    var img = this.imgs[0];
    img.onload = () => {
      img.onload = null;
      this.width = img.naturalWidth;
      this.height = img.naturalHeight;
      this.init();
    };
  },
  methods: {
    init() {
      var params = geo.util.pixelCoordinateParams(
        this.$refs.container,
        this.width,
        this.height,
        this.width,
        this.height
      );
      this.viewer = geo.map(params.map);
      this.viewer.zoomRange({
        min: this.viewer.zoomRange().origMin,
        max: this.viewer.zoomRange().max + 3
      });
      var interactorOpts = this.viewer.interactor().options();
      interactorOpts.keyboard.focusHighlight = false;
      interactorOpts.keyboard.actions = {};
      interactorOpts.actions = [
        interactorOpts.actions[0],
        interactorOpts.actions[2],
        interactorOpts.actions[6],
        interactorOpts.actions[7],
        interactorOpts.actions[8]
      ];
      this.viewer.interactor().options(interactorOpts);

      this.quadFeatureLayer = this.viewer.createLayer("feature", {
        features: ["quad"]
      });
      this.quadFeature = this.quadFeatureLayer
        .createFeature("quad")
        .data([
          {
            ul: { x: 0, y: 0 },
            lr: { x: this.width, y: this.height },
            image: this.imgs[this.frame]
          }
        ])
        .draw();
      this.ready = true;
    },
    async play() {
      try {
        this.playing = true;
        this.syncWithVideo();
      } catch (ex) {
        console.log(ex);
      }
    },
    async seek(frame) {
      this.frame = frame;
      this.syncedFrame = frame;
      this.emitFrame();
      this.cacheImage();
      this.quadFeature
        .data([
          {
            ul: { x: 0, y: 0 },
            lr: { x: this.width, y: this.height },
            image: this.imgs[frame]
          }
        ])
        .draw();
    },
    prevFrame() {
      var targetFrame = this.frame - 1;
      if (targetFrame >= 0) {
        this.seek(targetFrame);
      }
    },
    nextFrame() {
      var targetFrame = this.frame + 1;
      if (targetFrame <= this.maxFrame) {
        this.seek(targetFrame);
      }
    },
    pause() {
      this.playing = false;
    },
    onResize() {
      if (!this.viewer) {
        return;
      }
      const size = this.$refs.container.getBoundingClientRect();
      const mapSize = this.viewer.size();
      if (size.width !== mapSize.width || size.height !== mapSize.height) {
        this.viewer.size(size);
      }
    },
    syncWithVideo() {
      if (this.playing) {
        this.frame++;
        this.syncedFrame++;
        if (this.frame > this.maxFrame) {
          this.pause();
          this.frame = this.maxFrame;
          this.syncedFrame = this.maxFrame;
          return;
        }
        this.seek(this.frame);
        setTimeout(this.syncWithVideo, 1000 / this.frameRate);
      }
    },
    rendered() {
      // console.log("rendered an");
    },
    cacheImage() {
      var frame = this.frame;
      var max = Math.min(frame + 10, this.maxFrame);
      var imgs = this.imgs;
      this.pendingImgs.forEach(imageAndFrame => {
        if (imageAndFrame[1] < frame || imageAndFrame[1] > max) {
          imgs[imageAndFrame[1]] = null;
          imageAndFrame[0].src = "";
          this.pendingImgs.delete(imageAndFrame);
        }
      });
      for (let i = frame; i <= max; i++) {
        if (!imgs[i]) {
          var img = new Image();
          img.crossOrigin = "Anonymous";
          img.src = this.imageUrls[i];
          imgs[i] = img;
          ((img, frame) => {
            var imageAndFrame = [img, frame];
            this.pendingImgs.add(imageAndFrame);
            img.onload = () => {
              this.pendingImgs.delete(imageAndFrame);
              img.onload = null;
            };
          })(img, i);
        }
      }
    },
    emitFrame() {
      this.$emit("frame-update", this.frame);
    }
  }
};
</script>

<template>
  <div class="video-annotator" v-resize="onResize">
    <div class="playback-container" ref="container">{{ rendered() }}</div>
    <slot name="control" />
    <slot v-if="ready" />
  </div>
</template>

<style lang="scss" scoped>
.video-annotator {
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  z-index: 0;

  display: flex;
  flex-direction: column;

  .playback-container {
    flex: 1;

    &.geojs-map:focus {
      outline: none;
    }
  }
}
</style>
