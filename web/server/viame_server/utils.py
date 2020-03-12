from girder.models.item import Item
from girder.models.folder import Folder


# Ad hoc way to guess the FPS of an Image Sequence based on file names
# Currently not being used, can only be used once you know that all items
# have been imported.
def determine_image_sequence_fps(folder):
    items = Item().find({"folderId": folder["_id"]})

    start = None
    current = None

    item_length = 0
    for item in items:
        item_length += 1
        name = item["name"]

        try:
            _, two, three, four, _ = name.split(".")
            seconds = two[:2] * 3600 + two[2:4] * 60 + two[4:]

            if not start:
                start = int(seconds)
            current = int(seconds)

        except ValueError:
            if "annotations.csv" not in name:
                return None

    total = current - start
    return round(item_length / total)


def get_or_create_auxiliary_folder(folder, user):
    return Folder().createFolder(folder, "auxiliary", reuseExisting=True, creator=user)


def move_existing_result_to_auxiliary_folder(folder, user):
    auxiliary = get_or_create_auxiliary_folder(folder, user)

    existingResultItem = Item().findOne(
        {
            "folderId": folder["_id"],
            "meta.folderId": str(folder["_id"]),
            "meta.pipeline": {"$exists": True},
        }
    )
    if existingResultItem:
        Item().move(existingResultItem, auxiliary)


def check_existing_annotations(event):
    info = event.info

    if "annotations.csv" in info["importPath"]:
        item = Item().findOne({"_id": info["id"]})
        item["meta"].update({"folderId": str(item["folderId"]), "pipeline": None})
        Item().save(item)

        folder = Folder().findOne({"_id": item["folderId"]})

        # FPS is hardcoded for now
        folder["meta"].update({"type": "image-sequence", "viame": True, "fps": 30})
        Folder().save(folder)
