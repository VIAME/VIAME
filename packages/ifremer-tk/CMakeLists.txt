cmake_minimum_required(VERSION 3.11)
project(ifremer_tk
  VERSION 0.1
  LANGUAGES CXX
  )

# Install IFREMER configuration files
install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/configs"
  DESTINATION .
  )
