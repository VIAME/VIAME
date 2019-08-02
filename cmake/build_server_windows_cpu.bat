IF EXIST build rmdir /s /q build
git submodule update --init --recursive
"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV
