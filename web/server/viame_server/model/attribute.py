###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

from girder.models.model_base import Model


class Attribute(Model):
    def initialize(self):
        self.name = "attribute"

    def validate(self, model):
        return model

    def create(self, name, belongs, datatype, values=None):
        doc = {"name": name, "belongs": belongs, "datatype": datatype}

        if datatype == "text" and values:
            doc["values"] = values

        return self.save(doc)
