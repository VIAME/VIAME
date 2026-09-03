@echo off
REM Backwards compatibility wrapper: viame_resample_tracks -> viame resample-tracks
REM This script will be removed once all tools are updated.
"%~dp0viame.exe" resample-tracks %*
