<script>
import { throttle } from "lodash";

import Vue from "vue";
import geo from "geojs";

export default {
  name: "VideoAnnotator",
  props: {
    videoUrl: {
      type: String,
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
  computed: {
    isLastFrame() {
      return this.playing;
    }
  },
  created() {
    this.provided.$on("play", this.play);
    this.provided.$on("prev-frame", this.prevFrame);
    this.provided.$on("next-frame", this.nextFrame);
    this.provided.$on("pause", this.pause);
    this.provided.$on("seek", this.seek);
    this.emitFrame();
    this.emitFrame = throttle(this.emitFrame, 200);
    var video = document.createElement("video");
    this.video = video;
    video.preload = "auto";
    video.src = this.videoUrl;
    video.onloadedmetadata = () => {
      video.onloadedmetadata = null;
      this.width = video.videoWidth;
      this.height = video.videoHeight;
      this.maxFrame = this.frameRate * video.duration;
      this.init();
    };
    video.addEventListener("pause", this.videoPaused);
    // setTimeout(() => {
    //   this.play();
    // }, 2000);
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
        features: ["quad.video"]
      });
      this.quadFeatureLayer
        .createFeature("quad")
        .data([
          {
            ul: { x: 0, y: 0 },
            lr: { x: this.width, y: this.height },
            video: this.video
          }
        ])
        .draw();
      this.ready = true;
    },
    async play() {
      try {
        await this.video.play();
        this.playing = true;
        this.syncWithVideo();
      } catch (ex) {
        console.log(ex);
      }
    },
    async seek(frame) {
      this.video.currentTime = frame / this.frameRate;
      this.frame = Math.round(this.video.currentTime * this.frameRate);
      this.emitFrame();
      this.video.removeEventListener("seeked", this.pendingUpdate);
      this.video.addEventListener("seeked", this.pendingUpdate);
    },
    pendingUpdate() {
      this.syncedFrame = Math.round(this.video.currentTime * this.frameRate);
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
      this.video.pause();
      this.playing = false;
    },
    videoPaused() {
      if (this.video.currentTime === this.video.duration) {
        // console.log("video ended");
        this.frame = 0;
        this.syncedFrame = 0;
        this.pause();
      }
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
        this.frame = Math.round(this.video.currentTime * this.frameRate);
        this.syncedFrame = this.frame;
        this.viewer.scheduleAnimationFrame(this.syncWithVideo);
      }
    },
    emitFrame() {
      this.$emit("frame-update", this.frame);
    },
    rendered() {
      // console.log("rendered an");
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
