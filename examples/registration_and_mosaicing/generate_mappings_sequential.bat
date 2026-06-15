@echo off

REM Path to VIAME installation
SET VIAME_INSTALL=.\..\..

REM Core processing options. INPUT is either a single folder of images or a
REM multi-camera rig folder containing PORT/STAR/CENTER subfolders.
SET INPUT=insert_foldername_here
SET OUTPUT=output

REM Setup paths and run command
CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Sequential homography registration + coverage with the recommended settings.
REM Frame-to-frame homographies are chained from an anchor frame (no global
REM bundle adjustment); the cross-camera transform is a robust per-rig consensus.
REM Works for both land-heavy and water-heavy scenes.
python.exe "%VIAME_INSTALL%\configs\reconstruct_3d.py" "%INPUT%" --output "%OUTPUT%" --planar --coverage-class suppressed --visualize --affine --consistency-filter --xcam-robust --xcam-low-drift

pause
