import os
from girder.utility.webroot import WebrootBase
from girder import constants


class ClientWebroot(WebrootBase):
    def __init__(self, templatePath=None):
        super(ClientWebroot, self).__init__("")

        self.vars = {
            # 'title' is deprecated use brandName instead
            "title": "Girder"
        }

    def GET(self, **params):
        file = open(os.path.join(constants.STATIC_ROOT_DIR, "viame", "index.html"), "r")
        return file.read()

    def DELETE(self, **params):
        raise Exception(405)

    def PATCH(self, **params):
        raise Exception(405)

    def POST(self, **params):
        raise Exception(405)

    def PUT(self, **params):
        raise Exception(405)
