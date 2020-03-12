<script>
import _ from "lodash";
import { mapState } from "vuex";
import * as d3 from "d3";
import colors from "vuetify/lib/util/colors";

import NavigationTitle from "@/components/NavigationTitle";
import VideoAnnotator from "@/components/VideoAnnotator";
import ImageAnnotator from "@/components/ImageAnnotator";
import Controls from "@/components/Controls";
import AnnotationLayer from "@/components/AnnotationLayer";
import EditAnnotationLayer from "@/components/EditAnnotationLayer";
import ConfidenceFilter from "@/components/ConfidenceFilter";
import Tracks from "@/components/Tracks";
import TypeList from "@/components/TypeList";
import AttributesPanel from "@/components/AttributesPanel";
import TextLayer from "@/components/TextLayer";
import MarkerLayer from "@/components/MarkerLayer";
import TimelineWrapper from "@/components/TimelineWrapper";
import Timeline from "@/components/timeline/Timeline";
import LineChart from "@/components/timeline/LineChart";
import EventChart from "@/components/timeline/EventChart";
import { getPathFromLocation } from "@/utils";

var typeColors = [
  colors.red.accent2,
  "aqua",
  "fuchsia",
  "yellow",
  colors.purple.lighten2,
  "#0099FF",
  colors.amber.accent3,
  colors.green.accent2,
  colors.lightBlue.accent2
];
var typeColorMap = d3.scaleOrdinal();
typeColorMap.range(typeColors);

export default {
  name: "Viewer",
  inject: ["girderRest"],
  components: {
    NavigationTitle,
    VideoAnnotator,
    ImageAnnotator,
    Controls,
    AnnotationLayer,
    EditAnnotationLayer,
    TextLayer,
    MarkerLayer,
    Timeline,
    TimelineWrapper,
    ConfidenceFilter,
    Tracks,
    TypeList,
    AttributesPanel,
    LineChart,
    EventChart
  },
  data: () => ({
    dataset: null,
    detections: null,
    selectedTrackId: null,
    checkedTracks: [],
    checkedTypes: [],
    confidence: 0.1,
    showTrackView: false,
    editingTrack: null,
    attributeEditing: false,
    frame: null,
    pendingSave: false,
    featurePointing: false,
    featurePointIndex: 0,
    featurePointingGeojson: null
  }),
  computed: {
    ...mapState(["location"]),
    annotatorType() {
      if (!this.dataset) {
        return null;
      }
      if (this.dataset.meta.type === "video") {
        return VideoAnnotator;
      } else if (this.dataset.meta.type === "image-sequence") {
        return ImageAnnotator;
      }
      return null;
    },
    imageUrls() {
      if (!this.items || this.dataset.meta.type !== "image-sequence") {
        return null;
      }
      return this.items
        .filter(item => {
          var name = item.name.toLowerCase();
          return (
            name.endsWith("png") ||
            name.endsWith("jpeg") ||
            name.endsWith("jpg")
          );
        })
        .map(item => {
          return `api/v1/item/${item._id}/download`;
        });
    },
    frameRate() {
      if (!this.dataset) {
        return null;
      }
      return this.dataset.meta.fps;
    },
    filteredDetections() {
      if (!this.detections) {
        return null;
      }
      var checkedTracksSet = new Set(this.checkedTracks);
      var checkedTypesSet = new Set(this.checkedTypes);
      var confidence = this.confidence;
      return this.detections.filter(
        detection =>
          checkedTracksSet.has(detection.track) &&
          (detection.confidencePairs.length === 0 ||
            detection.confidencePairs.find(
              pair => pair[1] > confidence && checkedTypesSet.has(pair[0])
            ))
      );
    },
    annotationData() {
      if (!this.filteredDetections) {
        return null;
      }
      return this.filteredDetections.map(detection => {
        return {
          detection,
          frame: detection.frame,
          polygon: boundToGeojson(detection.bounds)
        };
      });
    },
    annotationStyle() {
      var selectedTrackId = this.selectedTrackId;
      var editingTrack = this.editingTrack;
      return {
        strokeColor: (a, b, data) => {
          if (data.record.detection.track === selectedTrackId) {
            return "lime";
          }
          if (data.record.detection.confidencePairs.length) {
            return typeColorMap(data.record.detection.confidencePairs[0][0]);
          } else {
            return typeColorMap.range()[0];
          }
        },
        strokeOpacity: (a, b, data) => {
          return data.record.detection.track === editingTrack ? 0.5 : 1;
        }
      };
    },
    textData() {
      if (!this.filteredDetections) {
        return null;
      }
      var data = [];
      this.filteredDetections.forEach(detection => {
        var bounds = detection.bounds;
        if (!detection.confidencePairs) {
          return;
        }
        detection.confidencePairs
          .filter(pair => pair[1] >= this.confidence)
          .forEach(([type, confidence], i) => {
            data.push({
              detection,
              frame: detection.frame,
              text: `${type}: ${confidence.toFixed(2)}`,
              x: bounds[1],
              y: bounds[2],
              offsetY: i * 14
            });
          });
      });
      return data;
    },
    textStyle() {
      var selectedTrackId = this.selectedTrackId;
      return {
        color: data => {
          if (data.detection.track === selectedTrackId) {
            return "lime";
          }
          return typeColorMap(data.detection.confidencePairs[0][0]);
        },
        offsetY(data) {
          return data.offsetY;
        }
      };
    },
    markerData() {
      if (!this.filteredDetections) {
        return null;
      }
      var data = [];
      this.filteredDetections.forEach(detection => {
        Object.entries(detection.features).forEach(([key, value]) => {
          data.push({
            detection,
            frame: detection.frame,
            feature: key,
            x: value[0],
            y: value[1]
          });
        });
      });
      return data;
    },
    markerStyle() {
      var selectedTrackId = this.selectedTrackId;
      return {
        fillColor: data => {
          return data.feature === "head" ? "orange" : "blue";
        },
        radius: 4,
        stroke: data => data.detection.track === selectedTrackId,
        strokeColor: "lime"
      };
    },
    lineChartData() {
      if (!this.filteredDetections) {
        return null;
      }
      var types = new Map();
      var total = new Map();
      this.filteredDetections.forEach(detection => {
        var frame = detection.frame;
        total.set(frame, total.get(frame) + 1 || 1);
        if (!detection.confidencePairs.length) {
          return;
        }
        var type = detection.confidencePairs[0][0];
        var typeCounter = types.get(type);
        if (!typeCounter) {
          typeCounter = new Map();
          types.set(type, typeCounter);
        }
        typeCounter.set(frame, typeCounter.get(frame) + 1 || 1);
      });
      return [
        {
          values: Array.from(total.entries()).sort((a, b) => a[0] - b[0]),
          color: "lime",
          name: "Total"
        },
        ...Array.from(types.entries()).map(([type, counter]) => ({
          values: Array.from(counter.entries()).sort((a, b) => a[0] - b[0]),
          name: type,
          color: typeColorMap(type)
        }))
      ];
    },
    eventChartData() {
      if (!this.filteredDetections) {
        return [];
      }
      return Object.entries(
        _.groupBy(this.filteredDetections, detection => detection.track)
      )
        .filter(([, detections]) => {
          return detections[0].confidencePairs.length;
        })
        .map(([name, detections]) => {
          var range = [
            _.minBy(detections, detection => detection.frame).frame,
            _.maxBy(detections, detection => detection.frame).frame
          ];
          return {
            track: detections[0].track,
            name: `Track ${name}`,
            color: typeColorMap(detections[0].confidencePairs[0][0]),
            range
          };
        });
    },
    tracks() {
      if (!this.detections) {
        return [];
      }
      var tracks = Object.entries(
        _.groupBy(this.detections, detection => detection.track)
      ).map(([, detections]) => {
        let confidencePairs = detections[0].confidencePairs;
        let detectionWithTrackAttribute = detections.find(
          detection => detection.trackAttributes
        );

        return {
          trackId: detections[0].track,
          confidencePairs,
          trackAttributes: detectionWithTrackAttribute
            ? detectionWithTrackAttribute.trackAttributes
            : null
        };
      });
      return _.sortBy(tracks, track => track.trackId);
    },
    types() {
      if (!this.tracks) {
        return [];
      }
      var typeSet = new Set();
      for (var { confidencePairs } of this.tracks) {
        for (var pair of confidencePairs) {
          typeSet.add(pair[0]);
        }
      }
      return Array.from(typeSet);
    },
    selectedDetection() {
      if (this.selectedTrackId === null || this.frame === null) {
        return null;
      }
      return this.detections.find(
        detection =>
          detection.track === this.selectedTrackId &&
          detection.frame === this.frame
      );
    },
    selectedTrack() {
      if (this.selectedTrackId === null) {
        return null;
      }
      return this.tracks.find(track => track.trackId === this.selectedTrackId);
    },
    editingDetection() {
      if (this.editingTrack == null || this.frame == null) {
        return null;
      }
      return this.detections.find(
        detection =>
          detection.track === this.editingTrack &&
          detection.frame === this.frame
      );
    },
    editingDetectionGeojson() {
      if (!this.editingDetection) {
        return null;
      }
      return boundToGeojson(this.editingDetection.bounds);
    }
  },
  asyncComputed: {
    async items() {
      if (!this.dataset) {
        return null;
      }
      var { data: items } = await this.girderRest.get(`item/`, {
        params: { folderId: this.dataset._id, limit: 200000 }
      });
      return items;
    },
    async videoUrl() {
      if (!this.dataset || this.dataset.meta.type !== "video") {
        return null;
      }
      var { data: clipMeta } = await this.girderRest.get(
        "viame_detection/clip_meta",
        {
          params: {
            folderId: this.dataset._id
          }
        }
      );
      if (!clipMeta.video) {
        return null;
      }
      var { data: files } = await this.girderRest.get(
        `item/${clipMeta.video._id}/files`
      );
      if (!files[0]) {
        return null;
      }
      return `api/v1/file/${files[0]._id}/download`;
    }
  },
  watch: {
    detections() {
      this.updatecheckedTracksAndTypes();
    }
  },
  async created() {
    var datasetId = this.$route.params.datasetId;
    try {
      await this.loadDataset(datasetId);
      await this.loadDetections();
    } catch (ex) {
      this.$router.replace("/");
    }
  },
  methods: {
    typeColorMap,
    getPathFromLocation,
    async loadDataset(datasetId) {
      var { data: dataset } = await this.girderRest.get(`folder/${datasetId}`);
      if (!dataset || !dataset.meta || !dataset.meta.viame) {
        return null;
      }
      this.dataset = dataset;
    },
    async loadDetections() {
      var { data: detections } = await this.girderRest.get("viame_detection", {
        params: { folderId: this.dataset._id }
      });
      this.detections = detections.map(detection => {
        return Object.freeze(detection);
      });
    },
    annotationClick(data) {
      if (!this.featurePointing) {
        this.selectTrack(data.detection.track);
      }
    },
    clickTrack(track) {
      this.selectTrack(track.trackId);
    },
    selectTrack(track) {
      if (this.editingTrack !== null) {
        this.editingTrack = null;
        return;
      }
      this.selectedTrackId = this.selectedTrackId === track ? null : track;
    },
    updatecheckedTracksAndTypes() {
      if (!this.tracks) {
        return;
      }
      this.checkedTracks = this.tracks.map(track => track.trackId);
      this.checkedTypes = this.types;
    },
    gotoTrackFirstFrame(track) {
      this.selectedTrackId = track.trackId;
      var frame = this.eventChartData.find(d => d.track === track.trackId)
        .range[0];
      this.$refs.playpackComponent.provided.$emit("seek", frame);
    },
    async deleteTrack(track) {
      var result = await this.$prompt({
        title: "Confirm",
        text: `Please confirm to delete track ${track.trackId}`,
        confirm: true
      });
      if (!result) {
        return;
      }
      this.pendingSave = true;
      this.detections
        .filter(detection => detection.track === track.trackId)
        .forEach(detection => {
          this.detections.splice(this.detections.indexOf(detection), 1);
        });
    },
    annotationRightClick(data) {
      this.editTrack(data.detection.track);
    },
    editTrack(track) {
      this.editingTrack = track;
    },
    addTrack() {
      this.editingTrack = this.tracks.slice(-1)[0].trackId + 1;
    },
    toggleFeaturePointing() {
      if (this.featurePointing) {
        this.featurePointing = false;
        this.featurePointIndex = 0;
      } else if (this.selectedTrackId === null) {
        return;
      } else {
        this.featurePointing = true;
      }
    },
    featurePointed(geojson) {
      this.pendingSave = true;
      var [x, y] = geojson.geometry.coordinates;
      var selectedDetection = this.selectedDetection;
      this.detections.splice(this.detections.indexOf(selectedDetection), 1);
      this.detections.push(
        Object.freeze({
          ...selectedDetection,
          ...{
            features: {
              ...selectedDetection.features,
              ...{
                [["head", "tail"][this.featurePointIndex]]: [
                  x.toFixed(0),
                  y.toFixed(0)
                ]
              }
            }
          }
        })
      );
      this.featurePointing = false;
      this.$nextTick(() => {
        if (this.featurePointIndex < 1) {
          this.featurePointIndex++;
          this.featurePointing = true;
        } else {
          this.featurePointIndex = 0;
        }
      });
    },
    deleteFeaturePoints() {
      this.pendingSave = true;
      var selectedDetection = this.selectedDetection;
      this.detections.splice(this.detections.indexOf(selectedDetection), 1);
      this.detections.push(
        Object.freeze({
          ...selectedDetection,
          ...{
            features: {}
          }
        })
      );
    },
    deleteDetection() {
      if (!this.selectedDetection) {
        return;
      }
      this.pendingSave = true;
      this.detections.splice(
        this.detections.indexOf(this.selectedDetection),
        1
      );
    },
    detectionChanged(feature) {
      if (this.editingTrack === null) {
        return;
      }
      this.pendingSave = true;
      var bounds =
        feature.type === "Feature"
          ? geojsonToBound2(feature.geometry)
          : geojsonToBound(feature);
      var confidencePairs = [];
      var trackMeta = this.tracks.find(
        track => track.trackId === this.editingTrack
      );
      if (trackMeta) {
        confidencePairs = trackMeta.confidencePairs;
      }
      if (this.editingDetection) {
        let detectionToChange = this.editingDetection;
        this.detections.splice(this.detections.indexOf(detectionToChange), 1);
        this.detections.push(
          Object.freeze({
            ...detectionToChange,
            ...{
              track: this.editingTrack,
              confidencePairs,
              frame: this.frame,
              features: {},
              bounds
            }
          })
        );
      } else {
        this.detections.push(
          Object.freeze({
            track: this.editingTrack,
            confidencePairs,
            frame: this.frame,
            features: {},
            confidence: 1,
            fishLength: -1,
            attributes: null,
            trackAttributes: null,
            bounds
          })
        );
      }
    },
    trackTypeChange(track, type) {
      var detections = this.detections;
      detections
        .filter(detection => detection.track === track.trackId)
        .forEach(detection => {
          var index = detections.indexOf(detection);
          detections.splice(index, 1);
          detections.push({
            ...detection,
            ...{
              confidence: 1,
              confidencePairs: [[type, 1]]
            }
          });
        });
      this.pendingSave = true;
    },
    attributeChange({ type, name, value }) {
      if (type === "track") {
        this.trackAttributeChange_(name, value);
      } else if (type === "detection") {
        this.detectionAttributeChange_(name, value);
      }
      this.pendingSave = true;
    },
    trackAttributeChange_(name, value) {
      var selectedTrack = this.selectedTrack;
      var detectionToChange = null;
      if (selectedTrack.trackAttributes) {
        detectionToChange = this.detections.find(
          detection =>
            detection.track === selectedTrack.trackId &&
            detection.trackAttributes
        );
      } else {
        detectionToChange = this.detections.find(
          detection => detection.track === selectedTrack.trackId
        );
      }
      var trackAttributes = {
        ...detectionToChange.trackAttributes,
        [name]: value
      };
      this.detections.splice(this.detections.indexOf(detectionToChange), 1);
      this.detections.push(
        Object.freeze({
          ...detectionToChange,
          trackAttributes
        })
      );
    },
    detectionAttributeChange_(name, value) {
      var detectionToChange = this.selectedDetection;
      var attributes = {
        ...detectionToChange.attributes,
        [name]: value
      };
      this.detections.splice(this.detections.indexOf(detectionToChange), 1);
      this.detections.push(
        Object.freeze({
          ...detectionToChange,
          attributes
        })
      );
    },
    async save() {
      await this.girderRest.put(
        `viame_detection?folderId=${this.$route.params.datasetId}`,
        this.detections
      );
      this.pendingSave = false;
    }
  }
};

function boundToGeojson(bounds) {
  return {
    type: "Polygon",
    coordinates: [
      [
        [bounds[0], bounds[2]],
        [bounds[1], bounds[2]],
        [bounds[1], bounds[3]],
        [bounds[0], bounds[3]],
        [bounds[0], bounds[2]]
      ]
    ]
  };
}

function geojsonToBound(geojson) {
  var coords = geojson.coordinates[0];
  return [coords[0][0], coords[1][0], coords[0][1], coords[2][1]];
}

function geojsonToBound2(geojson) {
  var coords = geojson.coordinates[0];
  return [coords[0][0], coords[2][0], coords[1][1], coords[0][1]];
}
</script>

<template>
  <v-content class="viewer">
    <v-app-bar app>
      <NavigationTitle />
      <v-tabs icons-and-text hide-slider style="flex-basis:0; flex-grow:0;">
        <v-tab :to="getPathFromLocation(location)"
          >Data<v-icon>mdi-database</v-icon></v-tab
        >
      </v-tabs>
      <span class="subtitle-1 text-center" style="flex-grow: 1;">{{
        dataset ? dataset.name : ""
      }}</span>
      <ConfidenceFilter :confidence.sync="confidence" />
      <v-btn icon :disabled="!pendingSave" @click="save"
        ><v-icon>mdi-content-save</v-icon></v-btn
      >
    </v-app-bar>
    <v-row no-gutters class="fill-height">
      <v-card width="300" style="z-index:1;">
        <v-btn
          icon
          class="swap-button"
          @click="attributeEditing = !attributeEditing"
          title="A key"
          v-mousetrap="[
            {
              bind: 'a',
              handler: () => {
                attributeEditing = !attributeEditing;
              }
            }
          ]"
          ><v-icon>mdi-swap-horizontal</v-icon></v-btn
        >
        <v-slide-x-transition>
          <div
            class="wrapper d-flex flex-column"
            v-if="!attributeEditing"
            key="type-tracks"
          >
            <TypeList
              class="flex-grow-1"
              :types="types"
              :checkedTypes.sync="checkedTypes"
              :colorMap="typeColorMap"
            />
            <v-divider />
            <Tracks
              :tracks="tracks"
              :types="types"
              :checked-tracks.sync="checkedTracks"
              :selected-track="selectedTrackId"
              :editing-track="editingTrack"
              class="flex-shrink-0"
              @goto-track-first-frame="gotoTrackFirstFrame"
              @delete-track="deleteTrack"
              @edit-track="editTrack($event.trackId)"
              @click-track="clickTrack"
              @add-track="addTrack"
              @track-type-change="trackTypeChange"
            />
          </div>
          <div v-else class="wrapper d-flex" key="attributes">
            <AttributesPanel
              :selectedDetection="selectedDetection"
              :selectedTrack="selectedTrack"
              @change="attributeChange"
            />
          </div>
        </v-slide-x-transition>
      </v-card>
      <v-col style="position: relative; ">
        <component
          class="playback-component"
          ref="playpackComponent"
          v-if="imageUrls || videoUrl"
          :is="annotatorType"
          :image-urls="imageUrls"
          :video-url="videoUrl"
          :frame-rate="frameRate"
          @frame-update="frame = $event"
          v-mousetrap="[
            { bind: 'f', handler: toggleFeaturePointing },
            { bind: 'd', handler: deleteDetection }
          ]"
        >
          <template slot="control">
            <Controls />
            <TimelineWrapper>
              <template #default="{maxFrame, frame, seek}">
                <Timeline :maxFrame="maxFrame" :frame="frame" :seek="seek">
                  <template #child="{startFrame, endFrame, maxFrame}">
                    <LineChart
                      v-if="!showTrackView && lineChartData"
                      :startFrame="startFrame"
                      :endFrame="endFrame"
                      :maxFrame="maxFrame"
                      :data="lineChartData"
                    />
                    <EventChart
                      v-if="showTrackView && eventChartData"
                      :startFrame="startFrame"
                      :endFrame="endFrame"
                      :maxFrame="maxFrame"
                      :data="eventChartData"
                    />
                  </template>
                  <v-btn
                    outlined
                    x-small
                    class="toggle-timeline-button"
                    @click="showTrackView = !showTrackView"
                    tabIndex="-1"
                  >
                    {{ showTrackView ? "Detection" : "Track" }}
                  </v-btn>
                </Timeline>
              </template>
            </TimelineWrapper>
          </template>
          <AnnotationLayer
            v-if="annotationData"
            :data="annotationData"
            :annotationStyle="annotationStyle"
            @annotation-click="annotationClick"
            @annotation-right-click="annotationRightClick"
          />
          <EditAnnotationLayer
            v-if="editingTrack !== null"
            editing="rectangle"
            :geojson="editingDetectionGeojson"
            :feature-style="{ fill: false, strokeColor: 'lime' }"
            @update:geojson="detectionChanged"
          />
          <EditAnnotationLayer
            v-if="featurePointing"
            editing="point"
            @update:geojson="featurePointed"
          />
          <TextLayer v-if="textData" :data="textData" :textStyle="textStyle" />
          <MarkerLayer
            v-if="markerData"
            :data="markerData"
            :markerStyle="markerStyle"
          />
        </component>
        <v-menu offset-y v-if="selectedDetection">
          <template v-slot:activator="{ on }">
            <v-btn class="selection-menu-button" icon v-on="on">
              <v-icon>mdi-dots-horizontal</v-icon>
            </v-btn>
          </template>
          <v-list>
            <v-list-item @click="toggleFeaturePointing">
              <v-list-item-title>Add feature points (F key)</v-list-item-title>
            </v-list-item>
            <v-list-item @click="deleteFeaturePoints">
              <v-list-item-title>Delete feature points</v-list-item-title>
            </v-list-item>
            <v-divider />
            <v-list-item @click="deleteDetection">
              <v-list-item-title>Delete detection (D key)</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-menu>
      </v-col>
    </v-row>
  </v-content>
</template>

<style lang="scss" scoped>
.wrapper {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

.toggle-timeline-button {
  position: absolute;
  top: -24px;
  left: 2px;
}

.confidence-filter {
  flex-basis: 400px;
}

.swap-button {
  position: absolute;
  top: 5px;
  right: 5px;
  z-index: 1;
}

.selection-menu-button {
  position: absolute;
  right: 0;
  top: 0;
  z-index: 1;
}
</style>

<style lang="scss">
.playback-component .playback-container {
  background: black;
}
</style>
