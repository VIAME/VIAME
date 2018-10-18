@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline

python.exe "%VIAME_INSTALL%\configs\process_video.py" --init -d INPUT_DIRECTORY ^
  --detection-plots ^
  -objects pristipomoides_auricilla,pristipomoides_zonatus,pristipomoides_sieboldii,etelis_carbunculus,etelis_coruscans,naso,aphareus_rutilans,seriola,hyporthodus_quernus,caranx_melampygus ^
  -plot-threshold 0.25 -frate 2 -smooth 2 ^
  -p pipelines\index_mouss.res.pipe --build-index --ball-tree

pause
