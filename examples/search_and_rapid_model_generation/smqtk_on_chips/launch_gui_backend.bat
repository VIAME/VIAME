REM Setup VIAME Paths (no need to set if installed to registry or already set up)

SET VIAME_INSTALL=.\..\..\..

CALL "%VIAME_INSTALL%\setup_viame.bat"

runApplication -a IqrSearchDispatcher -c configs/run_app.iqr_dispatcher.json -tv >runApp.IqrSearchDispatcher.log 2>&1
