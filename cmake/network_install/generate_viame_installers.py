from argparse import ArgumentParser
import girder_client
import glob
import os
import pdb
import shutil
import sys
from subprocess import run
from tempfile import TemporaryDirectory

run_dir = os.path.dirname(__file__)

cmake_template = """
project({data_name}Data NONE)

if(NOT EXISTS ${{CMAKE_BINARY_DIR}}/{data_name}.zip)
  file(DOWNLOAD https://data.kitware.com/api/v1/item/{data_sha}/download
    ${{CMAKE_BINARY_DIR}}/{data_name}.zip
    SHOW_PROGRESS)
endif()

install(FILES ${{CMAKE_BINARY_DIR}}/{data_name}.zip
  DESTINATION data)

set(CPACK_BINARY_WIX ON CACHE BOOL "Use WIX" FORCE)
set(CPACK_BINARY_NSIS OFF CACHE BOOL "No NSIS" FORCE)
include(CPack)"""

pkg_template = """<MsiPackage Id="{data_name}Data"
            Name="{file_name}"
            Compressed="no"
            DisplayInternalUI="yes"
            DownloadUrl="{girder_url}/item/{file_id}/download"
            DisplayName="Downloading {file_name}"
            InstallCondition="{data_name}ModelsCheckbox=1">
</MsiPackage>
"""

# Before Running:
# Ensure the information below is up to date
gider_parent_folder_id = ""
girder_id = ""
girder_password = ""
girder_url = "https://viame.kitware.com/api/v1"

# Dictionary of descriptive string to 
# ITEM Id of the archive file. 

data_name_dict = {
  "SeaLion1_1": "5f6639dd50a41e3d199a4153",
  "SeaLion1_2": "5fcdb5f150a41e3d198fa856",
  "MOUSS": "5cdec8ac8d777f072bb4457f",
   "MOUSSAlt": "5ce5af728d777f072bd5836d",
  "SEFSC1_1": "5fb69a7450a41e3d1960c6a5",
  "SEFSC1_2": "5fcdb5f150a41e3d198fa856",
  "ArcticSeals": "5e30b8ffaf2e2eed3545bff6"
}

def main(options):
  # End pre-set information 
  file_list = {}
  gc = girder_client.GirderClient(apiUrl=girder_url)
  gc.authenticate(girder_id, girder_password)

  new_folder = gc.createFolder(gider_parent_folder_id,
                              "Supplemental Installers",
                              parentType='collection',
                              reuseExisting=True)

  for data_name in data_name_dict:
    # First, check to see if it exists already
    existing_obj = gc.get("item", {"folderId": new_folder["_id"], "name": data_name+"Data-0.1.1-win32.msi"})
    # if not asked to remake it, skip the
    if options.remake_all or not existing_obj:
      with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "CMakeLists.txt"), "w") as cmake_file:
          text = cmake_template.format(data_name=data_name,
                                      data_sha=data_name_dict[data_name])
          cmake_file.write(text)
        run(["cmake", "."], cwd=tmpdir)
        run(["cmake", "--build", ".", "--target", "PACKAGE"], cwd=tmpdir)
        installer_file = glob.glob(os.path.join(tmpdir, "*.msi"))[0] # Should only ever be one 
        installer_dir, installer_name = os.path.split(installer_file)
        installer_size = os.path.getsize(installer_file)

        with open(installer_file,"rb") as upload_file:
          file_list[installer_name] = gc.uploadFile(new_folder["_id"],
                                                    upload_file,
                                                    installer_name,
                                                    installer_size,
                                                    parentType="folder"
                                                    )

        # Additionally, move to __file__'s location for WiX to capture data of
        shutil.move(installer_file, os.path.join(run_dir, installer_name))
    else:
      file_data = existing_obj[0]
      file_data["itemId"] = file_data["_id"]
      file_list[ data_name+"Data-0.1.1-win32.msi"] = file_data
  with open(os.path.join(run_dir, "VIAME_Chain_File.wxs"), "w") as chain:
    chain.write("<Include>\n")
    for installer_name, installer_data in file_list.items():
      chain.write(pkg_template.format(data_name=installer_name.split("-")[0],
                                      girder_url=girder_url,
                                      file_name=installer_name,
                                      file_id=installer_data["itemId"]))
    chain.write("</Include>")

if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument(
        "--remake_all",
        action="store_true",
        required=False,
        default=False,
    )
    options = arg.parse_args(sys.argv[1:])
    main(options)