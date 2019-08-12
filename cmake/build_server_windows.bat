IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\kv1 rmdir /s /q C:\tmp\kv1

git submodule update --init --recursive

"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV
