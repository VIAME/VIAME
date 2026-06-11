
# Identify root VIAME path
export VIAME_DIR="$PWD/install"

if [ ! -d "$VIAME_DIR" ]; then
  if [ -d "/viame/" ]; then
    export VIAME_DIR="/viame/build/install"
  fi
fi

# Add VIAME paths
export PATH=$PATH:$VIAME_DIR/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIAME_DIR/lib

# Add Python Paths
PY_VERSION="3.10"
if [ -d "$VIAME_DIR/include/python3.10d" ]; then
  PY_VERSION="3.10d"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIAME_DIR/lib/python${PY_VERSION}
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$VIAME_DIR/include/python${PY_VERSION}
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$VIAME_DIR/include/python${PY_VERSION}










