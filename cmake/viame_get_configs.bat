@echo off
REM Backwards compatibility wrapper: viame_get_configs -> viame get-configs
REM This script will be removed once all tools are updated.
"%~dp0viame.exe" get-configs %*
