
Download the demo data from `https://data.kitware.com/#item/5a8607858d777f068578345e` (Note that this is done for you by cmake)

### Building:

Make sure you build VIAME with `VIAME_ENABLE_PYTHON=True` and
`VIAME_ENABLE_CAMTRAWL=True`.  (For development it is useful to set
`VIAME_SYMLINK_PYTHON=True`)

### Running via installed camtrawl python module 

Remember to source the setup VIAME script

```
source install/setup_viame.sh

# you may also want to set these environment variables
# export KWIVER_DEFAULT_LOG_LEVEL=debug
export KWIVER_DEFAULT_LOG_LEVEL=info
export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes
```


You should be able to run the help command 
```
python -m viame.processes.camtrawl.demo --help
```

The script can be run on the demodata via:
```
python -m viame.processes.camtrawl.demo \
    --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
    --cal=camtrawl_demodata/cal.npz \
    --out=out --draw -f
```


### Running via the standalone script

Alternatively you can run by specifying the path to camtrawl module (if you
have a python environment you should be able to run this without even building
VIAME):


```
python ../../plugins/camtrawl/python/viame/processes/camtrawl \
    --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
    --cal=camtrawl_demodata/cal.npz \
    --out=out --draw -f
```


### Running via the pipeline runner

The above pure python script has been ported into the sprokit C++ pipeline
framework.

To run the process via the pipeline runner use the command:
(Note this method may not be stable and is under development)
```
pipeline_runner -p camtrawl_demo.pipe -S pythread_per_process
```
