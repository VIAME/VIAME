import os
import cherrypy
from girder.models.user import User
from girder.models.assetstore import Assetstore
from girder.exceptions import ValidationException

cherrypy.config["database"]["uri"] = os.getenv("MONGO_URI")

ADMIN_USER = os.getenv("GIRDER_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("GIRDER_ADMIN_PASS", "letmein")


def createInitialUser():
    try:
        User().createUser(
            ADMIN_USER, ADMIN_PASS, ADMIN_USER, ADMIN_USER, "admin@admin.com"
        )
    except ValidationException:
        print("Admin user already exists, skipping...")


def createAssetstore():
    try:
        Assetstore().createFilesystemAssetstore("assetstore", "/home/assetstore")
    except ValidationException:
        print("Assetstore already exists, skipping...")


def run_girder_init():
    createInitialUser()
    createAssetstore()


if __name__ == "__main__":
    run_girder_init()
