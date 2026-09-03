@echo off
REM Backwards compatibility wrapper: viame_score_results -> viame score
REM This script will be removed once all tools are updated.
"%~dp0viame.exe" score %*
