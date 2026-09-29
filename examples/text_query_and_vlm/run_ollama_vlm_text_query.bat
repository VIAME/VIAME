@echo off

REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

REM Run Pipeline (requires Ollama to be running, and: ollama pull qwen3-vl:8b)

kwiver.exe runner "%VIAME_INSTALL%\configs\pipelines\utility_text_query_ollama_vlm_tracking.pipe" ^
                  -s input:video_filename=input_list.txt ^
                  -s detector:detector:ollama_vlm:text_query="fish"

pause
