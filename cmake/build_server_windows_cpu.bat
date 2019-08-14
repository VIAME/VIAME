IF EXIST build rmdir /s /q build

IF NOT EXIST C:\tmp mkdir C:\tmp
IF EXIST C:\tmp\kv2 rmdir /s /q C:\tmp\kv2
IF EXIST C:\tmp\vm2 rmdir /s /q C:\tmp\vm2

SET PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin";"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp";C:\Python36;C:\Python36\Scripts;"C:\Program Files (x86)\Common Files\Oracle\Java\javapath";C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;"C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common";"C:\Program Files\Git\cmd";"C:\Program Files\CMake\bin";C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;"C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR";C:\Users\kitware\AppData\Local\Microsoft\WindowsApps;

git submodule update --init --recursive

"C:\Program Files\CMake\bin\ctest.exe" -S jenkins_dashboard.cmake -VV
