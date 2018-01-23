# External Data
include(ExternalData)

set(ExternalData_URL_TEMPLATES "https://data.kitware.com/api/v1/file/hashsum/%(algo)/%(hash)/download")
set(ExternalData_LINK_CONTENT SHA512)
set(ExternalData_TIMEOUT_ABSOLUTE 0)

# Any large examples that we need to pull from Girder
include(darknet.cmake)