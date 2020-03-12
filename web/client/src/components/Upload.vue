<script>
import Dropzone from "@girder/components/src/components/Presentation/Dropzone.vue";
import { Upload } from "@girder/components/src/utils";

import pipelines from "@/pipelines";

export default {
  name: "Upload",
  components: { Dropzone },
  inject: ["girderRest"],
  props: {
    location: {
      type: Object
    },
    uploading: {
      type: Boolean
    }
  },
  data: () => ({
    pendingUploads: []
  }),
  computed: {
    uploadEnabled() {
      return this.location && this.location._modelType === "folder";
    },
    pipelineItems() {
      return [
        { text: "None", value: null },
        ...pipelines.map(pipeline => ({ text: pipeline, value: pipeline }))
      ];
    }
  },
  methods: {
    async dropped(e) {
      e.preventDefault();
      let [name, files] = await readFilesFromDrop(e);
      this.addPendingUpload(name, files);
    },
    onFileChange(files) {
      var name = files.length === 1 ? files[0].name : "";
      this.addPendingUpload(name, files);
    },
    addPendingUpload(name, allFiles) {
      var [type, files] = prepareFiles(allFiles);
      this.pendingUploads.push({
        name,
        files,
        type,
        fps: null,
        pipeline: null,
        uploading: false
      });
    },
    remove(pendingUpload) {
      var index = this.pendingUploads.indexOf(pendingUpload);
      this.pendingUploads.splice(index, 1);
    },
    async upload() {
      if (this.location._modelType !== "folder") {
        return;
      }
      if (!this.$refs.form.validate()) {
        return;
      }
      var uploaded = [];
      this.$emit("update:uploading", true);
      await Promise.all(
        this.pendingUploads.map(async pendingUpload => {
          var { name, files, fps } = pendingUpload;
          fps = parseInt(fps);
          pendingUpload.uploading = true;
          var { data: folder } = await this.girderRest.post(
            "/folder",
            `metadata=${JSON.stringify({
              viame: true,
              fps,
              type: pendingUpload.type
            })}`,
            {
              params: {
                parentId: this.location._id,
                name
              }
            }
          );
          var pending = files;
          var results = [];
          while (pending.length) {
            results = results.concat(
              await Promise.all(
                pending.splice(0, 500).map(async file => {
                  var uploader = new Upload(file, {
                    $rest: this.girderRest,
                    parent: folder
                  });
                  return await uploader.start();
                })
              )
            );
          }
          uploaded.push({ folder, results, pipeline: pendingUpload.pipeline });
          this.remove(pendingUpload);
        })
      );
      this.$emit("update:uploading", false);
      this.$emit("uploaded", uploaded);
    }
  }
};

async function readFilesFromDrop(e) {
  var item = e.dataTransfer.items[0];
  var firstEntry = item.webkitGetAsEntry();
  if (!firstEntry.isDirectory) {
    return [
      firstEntry.name,
      Array.from(e.dataTransfer.items)
        .filter(item => item.webkitGetAsEntry().isFile)
        .map(item => item.getAsFile())
    ];
  } else {
    let entries = await readDirectoryEntries(firstEntry);
    return [
      firstEntry.name,
      await Promise.all(entries.filter(entry => entry.isFile).map(entryToFile))
    ];
  }
}

async function readDirectoryEntries(entry) {
  let entries = [];
  let reader = entry.createReader();
  let readEntries = await readEntriesPromise(reader);
  while (readEntries.length > 0) {
    entries.push(...readEntries);
    readEntries = await readEntriesPromise(reader);
  }
  return entries;
}

async function readEntriesPromise(directoryReader) {
  try {
    return await new Promise((resolve, reject) => {
      directoryReader.readEntries(resolve, reject);
    });
  } catch (err) {
    console.log(err);
  }
}

function entryToFile(entry) {
  return new Promise(resolve => {
    entry.file(file => {
      resolve(file);
    });
  });
}

function prepareFiles(files) {
  var videoFilter = file => /\.mp4$|\.avi$|\.mov$/i.test(file.name);
  var csvFilter = file => /\.csv$/i.test(file.name);
  var imageFilter = file => /\.jpg$|\.jpeg$|\.png$|\.bmp$/i.test(file.name);

  if (files.find(videoFilter)) {
    return [
      "video",
      files.filter(file => videoFilter(file) || csvFilter(file))
    ];
  } else {
    return [
      "image-sequence",
      files.filter(file => imageFilter(file) || csvFilter(file))
    ];
  }
}
</script>

<template>
  <div class="upload">
    <v-form
      v-if="pendingUploads.length"
      ref="form"
      class="pending-upload-form"
      @submit.prevent="upload"
    >
      <v-toolbar flat color="primary" dark dense>
        <v-toolbar-title>Pending upload</v-toolbar-title>
        <v-spacer />
        <v-btn type="submit" text :disabled="!uploadEnabled">
          Upload
        </v-btn>
      </v-toolbar>
      <v-list class="py-0 pending-uploads">
        <v-list-item v-for="(pendingUpload, i) of pendingUploads" :key="i">
          <v-list-item-content>
            <v-row>
              <v-col>
                <v-text-field
                  class="upload-name"
                  v-model="pendingUpload.name"
                  :rules="[
                    val => (val || '').length > 0 || 'This field is required'
                  ]"
                  required
                  label="Name"
                  hide-details
                  :disabled="pendingUpload.uploading"
                ></v-text-field>
              </v-col>
              <v-col :cols="2" v-if="pendingUpload.type === 'image-sequence'">
                <v-text-field
                  v-model="pendingUpload.fps"
                  type="number"
                  :rules="[
                    val => (val || '').length > 0 || 'This field is required'
                  ]"
                  required
                  label="FPS"
                  hide-details
                  :disabled="pendingUpload.uploading"
                ></v-text-field>
              </v-col>
              <v-col :cols="4">
                <v-select
                  v-model="pendingUpload.pipeline"
                  :items="pipelineItems"
                  label="Run pipeline"
                  :disabled="pendingUpload.uploading"
                />
              </v-col>
            </v-row>
            <v-list-item-subtitle
              v-if="pendingUpload.type === 'image-sequence'"
            >
              {{ pendingUpload.files.length }} images
            </v-list-item-subtitle>
          </v-list-item-content>
          <v-list-item-action>
            <v-btn
              icon
              small
              @click="remove(pendingUpload)"
              :disabled="pendingUpload.uploading"
            >
              <v-icon>mdi-close</v-icon>
            </v-btn>
          </v-list-item-action>
          <v-progress-linear
            :active="pendingUpload.uploading"
            :indeterminate="true"
            absolute
            bottom
          ></v-progress-linear>
        </v-list-item>
      </v-list>
    </v-form>
    <div class="dropzone-container">
      <Dropzone
        class="dropzone"
        multiple
        message="Drag files or directory here"
        @drop.native="dropped"
        @change="onFileChange"
      />
    </div>
  </div>
</template>

<style lang="scss" scoped>
.upload {
  min-height: 350px;
  display: flex;
  flex-direction: column;

  .pending-upload-form {
    max-height: 65%;
    overflow-y: auto;
    display: flex;
    flex-direction: column;

    .pending-uploads {
      overflow-y: auto;
    }
  }

  .dropzone-container {
    flex: 1;
    height: 1px;
  }
}
</style>

<style lang="scss">
.upload {
  .upload-name {
    .v-input__slot {
      padding-left: 0 !important;
    }
  }
}

.v-progress-linear--absolute {
  margin: 0;
}
</style>
