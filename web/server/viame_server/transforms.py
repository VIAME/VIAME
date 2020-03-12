import os
import shutil
import tempfile

from girder_worker_utils.transforms.girder_io import (
    GirderClientTransform,
    GirderClientResultTransform,
)


class GetPathFromItemId(GirderClientTransform):
    """
    This transform downloads a Girder Item to a directory on the local machine
    and passes its local path into the function.
    :param _id: The ID of the item to download.
    :type _id: str
    """

    def __init__(self, _id, **kwargs):
        super(GetPathFromItemId, self).__init__(**kwargs)
        self.item_id = _id

    def _repr_model_(self):
        return "{}('{}')".format(self.__class__.__name__, self.item_id)

    def transform(self):
        temp_dir = tempfile.mkdtemp()
        self.item_path = os.path.join(temp_dir, self.item_id)

        self.gc.downloadItem(self.item_id, temp_dir, self.item_id)

        return self.item_path

    def cleanup(self):
        shutil.rmtree(os.path.dirname(self.item_path), ignore_errors=True)


class GetPathFromFolderId(GirderClientTransform):
    """
    This transform downloads a Girder Item to a directory on the local machine
    and passes its local path into the function.
    :param _id: The ID of the item to download.
    :type _id: str
    """

    def __init__(self, _id, **kwargs):
        super(GetPathFromFolderId, self).__init__(**kwargs)
        self.folder_id = _id

    def _repr_model_(self):
        return "{}('{}')".format(self.__class__.__name__, self.folder_id)

    def transform(self):
        temp_dir = tempfile.mkdtemp()
        self.folder_path = os.path.join(temp_dir, self.folder_id)

        self.gc.downloadFolderRecursive(self.folder_id, self.folder_path)

        return self.folder_path

    # def cleanup(self):
    #     shutil.rmtree(os.path.dirname(self.folder_path),
    #                   ignore_errors=True)


class GirderUploadToFolder(GirderClientResultTransform):
    """
    This is a result hook transform that uploads a file or directory recursively
    to a folder in Girder.
    :param _id: The ID of the folder to upload into.
    :type _id: str
    :param delete_file: Whether to delete the local data afterward
    :type delete_file: bool
    :param upload_kwargs: Additional kwargs to pass to the upload method.
    :type upload_kwargs: dict
    """

    def __init__(
        self, _id, metadata=None, delete_file=False, upload_kwargs=None, **kwargs
    ):
        super(GirderUploadToFolder, self).__init__(**kwargs)
        self.folder_id = _id
        self.metadata = metadata
        self.upload_kwargs = upload_kwargs or {}
        self.delete_file = delete_file

    def _repr_model_(self):
        return "{}('{}')".format(self.__class__.__name__, self.folder_id)

    def _uploadFolder(self, path, folder_id):
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                _file = self.gc.uploadFileToFolder(
                    folder_id, fpath, **self.upload_kwargs
                )
                self.gc.addMetadataToItem(_file["itemId"], self.metadata)

            elif os.path.isdir(fpath) and not os.path.islink(fpath):
                folder = self.gc.createFolder(folder_id, f, reuseExisting=True)
                self._uploadFolder(fpath, folder["_id"])
                self.gc.addMetadataToFolder(folder["_id"], self.metadata)

    def transform(self, path):
        self.output_file_path = path
        if os.path.isdir(path):
            self._uploadFolder(path, self.folder_id)
        else:
            _file = self.gc.uploadFileToFolder(
                self.folder_id, path, **self.upload_kwargs
            )
            self.gc.addMetadataToItem(_file["itemId"], self.metadata)
        return self.folder_id

    def cleanup(self):
        if self.delete_file is True:
            if os.path.isdir(self.output_file_path):
                shutil.rmtree(self.output_file_path)
            else:
                os.remove(self.output_file_path)
