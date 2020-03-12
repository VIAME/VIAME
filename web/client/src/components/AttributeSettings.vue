<script>
export default {
  name: "AttributeSettings",
  inject: ["girderRest"],
  data: () => ({
    selectedIndex: undefined,
    name: "",
    belongs: "",
    datatype: "",
    values: [],
    addNew: false,
    changed: false
  }),
  computed: {
    selectedAttribute() {
      if (
        this.selectedIndex !== undefined &&
        this.attributes[this.selectedIndex]
      ) {
        return this.attributes[this.selectedIndex];
      } else {
        return null;
      }
    },
    disabled() {
      return this.selectedIndex === undefined && this.addNew === false;
    },
    textValues: {
      get() {
        if (this.values) {
          return this.values.join("\n");
        } else {
          return "";
        }
      },
      set(value) {
        this.values = value.split("\n");
      }
    }
  },
  asyncComputed: {
    async attributes() {
      var { data } = await this.girderRest.get("/viame/attribute");
      return data;
    }
  },
  watch: {
    selectedAttribute(attribute) {
      if (attribute) {
        this.addNew = false;
        this.name = attribute.name;
        this.belongs = attribute.belongs;
        this.datatype = attribute.datatype;
        this.values = attribute.values;
      } else {
        this.name = "";
        this.belongs = "track";
        this.datatype = "number";
      }
      this.$nextTick(() => {
        this.changed = false;
      });
    },
    name() {
      this.changed = true;
    },
    belongs() {
      this.changed = true;
    },
    datatype() {
      this.changed = true;
    },
    values() {
      this.changed = true;
    }
  },
  created() {
    this.setDefaultValue();
  },
  methods: {
    setDefaultValue() {
      this.name = "";
      this.belongs = "track";
      this.datatype = "number";
      this.values = [];
      this.$nextTick(() => {
        this.changed = false;
      });
    },
    add() {
      this.setDefaultValue();
      this.addNew = true;
      this.$refs.form.resetValidation();
    },
    async submit() {
      if (!this.$refs.form.validate()) {
        return;
      }

      var content = {
        name: this.name,
        belongs: this.belongs,
        datatype: this.datatype,
        values: this.datatype === "text" ? this.values : []
      };

      if (this.addNew) {
        await this.girderRest.post("/viame/attribute", content);
      } else {
        await this.girderRest.put(
          `/viame/attribute/${this.selectedAttribute._id}`,
          content
        );
      }
      this.$asyncComputed.attributes.update();
      this.changed = false;
    },
    async deleteAttribute() {
      var result = await this.$prompt({
        title: "Confirm",
        text: "Do you want to delete this attribute?",
        confirm: true
      });
      if (!result) {
        return;
      }
      await this.girderRest.delete(
        `/viame/attribute/${this.selectedAttribute._id}`
      );
      this.$asyncComputed.attributes.update();
    }
  }
};
</script>

<template>
  <v-card class="attribute-settings">
    <v-card-title class="pb-0">
      Attributes
    </v-card-title>
    <v-card-text>
      <v-row>
        <v-col cols="2">
          <v-list v-if="attributes" dense>
            <v-list-item-group v-model="selectedIndex" color="primary">
              <v-list-item v-for="attribute in attributes" :key="attribute._id">
                <v-list-item-content>
                  <v-list-item-title>{{ attribute.name }}</v-list-item-title>
                </v-list-item-content>
              </v-list-item>
            </v-list-item-group>
          </v-list>
        </v-col>
        <v-col>
          <v-form ref="form">
            <v-text-field
              style="max-width: 220px;"
              label="Name"
              v-model="name"
              :rules="[v => !!v || 'Name is required']"
              required
              :disabled="disabled"
            ></v-text-field>
            <v-radio-group
              label="Belongs to"
              v-model="belongs"
              :mandatory="true"
              :disabled="disabled"
            >
              <v-radio label="Track" value="track"></v-radio>
              <v-radio label="Detection" value="detection"></v-radio>
            </v-radio-group>
            <v-select
              style="max-width: 220px;"
              v-model="datatype"
              :items="[
                { text: 'Boolean', value: 'boolean' },
                { text: 'Number', value: 'number' },
                { text: 'Text', value: 'text' }
              ]"
              label="Datatype"
              :disabled="disabled"
            ></v-select>
            <v-textarea
              v-if="datatype === 'text'"
              style="max-width: 250px;"
              label="Predefined values"
              v-model="textValues"
              hint="Line separated values"
              auto-grow
              row-height="30"
            ></v-textarea>
            <v-row>
              <v-col>
                <v-btn
                  type="submit"
                  color="primary"
                  @click.prevent="submit"
                  :disabled="disabled || !changed"
                  >{{ addNew ? "Add" : "Save" }}</v-btn
                >
                <v-btn
                  v-if="!addNew && !disabled"
                  text
                  class="ml-3"
                  color="error"
                  @click.prevent="deleteAttribute"
                  >Delete</v-btn
                >
              </v-col>
            </v-row>
          </v-form>
          <v-btn
            absolute
            dark
            class="mb-8"
            fab
            small
            bottom
            right
            color="primary"
            @click="add"
          >
            <v-icon>mdi-plus</v-icon>
          </v-btn>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>
</template>

<style lang="scss">
.attribute-settings {
  .v-textarea textarea {
    line-height: 24px;
  }
}
</style>
