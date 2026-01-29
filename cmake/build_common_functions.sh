#!/bin/bash

# VIAME Build Common Functions
# Consolidated utility functions for all build scripts
#
# This file contains all shared functions for:
#   - OS and package manager detection
#   - Environment setup
#   - Dependency installation (apt/yum)
#   - CUDA/CUDNN library management
#   - Build utilities (cmake, openssl, opencv extras)
#   - Build workflow functions
#
# Usage: source this file in your build script
#   source /path/to/build_common_functions.sh

# ==============================================================================
# OS AND PACKAGE MANAGER DETECTION
# ==============================================================================

# Detect OS type
# Returns: "ubuntu", "rocky", "centos", "mac", or "unknown"
detect_os_type() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "mac"
  elif [ -f /etc/os-release ]; then
    if grep -qi 'ubuntu' /etc/os-release; then
      echo "ubuntu"
    elif grep -qi 'rocky' /etc/os-release; then
      echo "rocky"
    elif grep -qi 'centos' /etc/os-release; then
      echo "centos"
    else
      echo "unknown"
    fi
  else
    echo "unknown"
  fi
}

# Detect package manager
# Returns: "apt", "yum", or "unknown"
detect_package_manager() {
  if command -v apt-get &> /dev/null; then
    echo "apt"
  elif command -v yum &> /dev/null; then
    echo "yum"
  elif command -v dnf &> /dev/null; then
    echo "yum"  # dnf is compatible with yum syntax
  else
    echo "unknown"
  fi
}

# Get default library base path for current OS
# Returns: library path (e.g., /usr/lib/x86_64-linux-gnu or /usr/lib64)
get_default_lib_base() {
  local os_type=$(detect_os_type)
  case "$os_type" in
    ubuntu)
      echo "/usr/lib/x86_64-linux-gnu"
      ;;
    rocky|centos)
      echo "/usr/lib64"
      ;;
    mac)
      echo "/usr/local/lib"
      ;;
    *)
      # Try to detect
      if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
        echo "/usr/lib/x86_64-linux-gnu"
      else
        echo "/usr/lib64"
      fi
      ;;
  esac
}

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Setup build environment paths
# Arguments:
#   $1 = VIAME install directory (required)
#   $2 = CUDA directory (optional, default: /usr/local/cuda)
#   $3 = Python version (optional, default: 3.10)
setup_build_environment() {
  local install_dir="$1"
  local cuda_dir="${2:-/usr/local/cuda}"
  local python_version="${3:-3.10}"

  if [ -z "$install_dir" ]; then
    echo "Error: install_dir is required"
    return 1
  fi

  # Basic paths
  export PATH="$install_dir/bin:$PATH"
  export LD_LIBRARY_PATH="$install_dir/lib:$LD_LIBRARY_PATH"

  # Python-specific paths
  if [ -n "$python_version" ]; then
    export LD_LIBRARY_PATH="$install_dir/lib/python${python_version}:$LD_LIBRARY_PATH"
    export C_INCLUDE_PATH="$install_dir/include/python${python_version}:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="$install_dir/include/python${python_version}:$CPLUS_INCLUDE_PATH"
  fi

  # CUDA paths (if directory exists)
  if [ -d "$cuda_dir" ]; then
    export PATH="$cuda_dir/bin:$PATH"
    export LD_LIBRARY_PATH="$cuda_dir/lib64:$LD_LIBRARY_PATH"
  fi

  echo "Environment configured:"
  echo "  Install dir: $install_dir"
  echo "  CUDA dir: $cuda_dir"
  echo "  Python version: $python_version"
}

# Setup basic build environment (minimal version)
# Arguments:
#   $1 = VIAME install directory (required)
#   $2 = CUDA directory (optional)
setup_basic_build_environment() {
  local install_dir="$1"
  local cuda_dir="${2:-}"

  export PATH="$install_dir/bin:$PATH"
  export LD_LIBRARY_PATH="$install_dir/lib:$LD_LIBRARY_PATH"

  if [ -n "$cuda_dir" ] && [ -d "$cuda_dir" ]; then
    export PATH="$cuda_dir/bin:$PATH"
    export LD_LIBRARY_PATH="$cuda_dir/lib64:$LD_LIBRARY_PATH"
  fi
}

# Export CUDA paths for cmake
# Arguments:
#   $1 = CUDA directory (required)
# Sets: CUDA_TOOLKIT_ROOT_DIR, CUDA_NVCC_EXECUTABLE
export_cuda_paths() {
  local cuda_dir="$1"

  if [ -z "$cuda_dir" ]; then
    echo "Error: cuda_dir is required"
    return 1
  fi

  if [ -d "$cuda_dir" ]; then
    export CUDA_TOOLKIT_ROOT_DIR="$cuda_dir"
    export CUDA_NVCC_EXECUTABLE="$cuda_dir/bin/nvcc"
    echo "CUDA paths exported from: $cuda_dir"
  else
    echo "Warning: CUDA directory not found: $cuda_dir"
    return 1
  fi
}

# ==============================================================================
# DEPENDENCY INSTALLATION
# ==============================================================================

# Install dependencies using apt-get (Debian/Ubuntu)
# Arguments:
#   $1 = pip target directory (optional, for installing Python packages)
install_deps_apt() {
  local pip_target="${1:-}"

  echo "Installing dependencies via apt-get..."

  apt-get update -y

  # Core build tools and libraries
  apt-get install -y \
    zip \
    git \
    wget \
    tar \
    libgl1-mesa-dev \
    libexpat1-dev \
    libgtk2.0-dev \
    libxt-dev \
    libxml2-dev \
    liblapack-dev \
    openssl \
    libssl-dev \
    curl \
    libcurl4-openssl-dev \
    gcc \
    g++ \
    gfortran \
    zlib1g-dev \
    bzip2 \
    libbz2-dev \
    liblzma-dev

  # Python system packages
  apt-get install -y \
    python3-dev \
    python3-pip \
    python-is-python3

  # Install numpy via pip (to target directory if specified, otherwise system)
  if [ -n "$pip_target" ]; then
    python -m pip install --target "$pip_target" numpy==1.25.2
  else
    python -m pip install numpy==1.25.2
  fi

  echo "apt-get dependency installation complete"
}

# Install dependencies using yum (CentOS/Rocky/RHEL)
install_deps_yum() {
  echo "Installing dependencies via yum..."

  yum -y update

  # Development tools group
  yum -y groupinstall 'Development Tools'

  # Core libraries and tools
  yum install -y \
    zip \
    git \
    wget \
    zlib \
    zlib-devel \
    zstd \
    freeglut-devel \
    freetype-devel \
    mesa-libGLU-devel \
    libffi-devel \
    libXt-devel \
    libXmu-devel \
    libXi-devel \
    expat-devel \
    readline-devel \
    curl-devel \
    atlas-devel \
    file \
    which \
    bzip2 \
    bzip2-devel \
    xz-devel \
    vim \
    perl \
    perl-IPC-Cmd

  echo "yum dependency installation complete"
}

# Install system dependencies (auto-detects package manager)
# Arguments: $1 = package manager override (optional: "apt" or "yum")
install_system_deps() {
  local pkg_manager="${1:-$(detect_package_manager)}"

  echo "Using package manager: $pkg_manager"

  case "$pkg_manager" in
    apt)
      install_deps_apt
      ;;
    yum)
      install_deps_yum
      ;;
    *)
      echo "Error: Unsupported or unknown package manager: $pkg_manager"
      echo "Supported package managers: apt, yum"
      return 1
      ;;
  esac

  echo "System dependency installation complete"
}

# ==============================================================================
# GCC TOOLSET (RHEL/Rocky/CentOS)
# ==============================================================================

# Setup modern GCC toolset for RHEL-based systems
# Rocky/CentOS 9+ has GCC 11+ by default, Rocky/CentOS 8 needs gcc-toolset
# This function installs and enables the toolset if needed
setup_gcc_toolset() {
  local toolset_version="${1:-11}"

  # Only applicable for RHEL-based systems
  if [ ! -f /etc/redhat-release ]; then
    echo "Not a RHEL-based system, skipping gcc-toolset setup"
    return 0
  fi

  if grep -q "release 8" /etc/redhat-release 2>/dev/null; then
    echo "Rocky/CentOS 8 detected, installing gcc-toolset-${toolset_version}..."
    yum install -y "gcc-toolset-${toolset_version}"
    source "/opt/rh/gcc-toolset-${toolset_version}/enable"
    echo "GCC toolset ${toolset_version} enabled"
    gcc --version
  else
    echo "Rocky/CentOS 9+ detected, using default GCC..."
    gcc --version
  fi
}

# ==============================================================================
# PYTHON PACKAGE MANAGEMENT
# ==============================================================================

# Upgrade pip and setuptools to ensure Python 3.12 compatibility
# This fixes the "pkgutil.ImpImporter" error with old setuptools
# Arguments:
#   $1 = Python executable (default: python)
#   $2 = pip target directory (optional, for installing to specific location)
upgrade_pip_setuptools() {
  local python_exec="${1:-python}"
  local pip_target="${2:-}"

  echo "Upgrading pip and setuptools for $python_exec..."

  if [ -n "$pip_target" ]; then
    # Install to target directory (pip itself can't be upgraded to a target, skip it)
    # Upgrade setuptools to a version compatible with Python 3.12+
    # setuptools >= 67.0.0 removed pkg_resources.ImpImporter usage
    # wheel >= 0.45.0 requires setuptools >= 70.1 for bdist_wheel compatibility
    "$python_exec" -m pip install --target "$pip_target" --upgrade "setuptools>=75.3.0" "wheel>=0.45.0" 2>/dev/null || true
  else
    # System-wide upgrade (for Docker/container builds)
    # Upgrade pip first
    "$python_exec" -m pip install --upgrade pip 2>/dev/null || true

    # Upgrade setuptools to a version compatible with Python 3.12+
    # setuptools >= 67.0.0 removed pkg_resources.ImpImporter usage
    # wheel >= 0.45.0 requires setuptools >= 70.1 for bdist_wheel compatibility
    "$python_exec" -m pip install --upgrade "setuptools>=75.3.0" "wheel>=0.45.0" 2>/dev/null || true
  fi

  echo "pip and setuptools upgrade complete"
}

# ==============================================================================
# NODE.JS AND YARN INSTALLATION (for DIVE desktop builds)
# ==============================================================================

# Install Node.js 18+ and yarn for building DIVE from source
# Arguments:
#   $1 = Node.js major version (default: 18)
install_nodejs_and_yarn() {
  local node_version="${1:-18}"

  echo "Installing Node.js ${node_version}.x and yarn..."

  local pkg_manager=$(detect_package_manager)

  case "$pkg_manager" in
    apt)
      # Install Node.js via NodeSource repository
      curl -fsSL "https://deb.nodesource.com/setup_${node_version}.x" | bash -
      apt-get install -y nodejs

      # Install yarn via npm
      npm install -g yarn
      ;;
    yum)
      # Install Node.js via NodeSource repository
      curl -fsSL "https://rpm.nodesource.com/setup_${node_version}.x" | bash -
      yum install -y nodejs

      # Install yarn via npm
      npm install -g yarn
      ;;
    *)
      echo "Error: Unsupported package manager for Node.js installation: $pkg_manager"
      return 1
      ;;
  esac

  # Verify installation
  echo "Node.js version: $(node --version)"
  echo "yarn version: $(yarn --version)"

  echo "Node.js and yarn installation complete"
}

# ==============================================================================
# CMAKE INSTALLATION
# ==============================================================================

# Install CMake from source
# Arguments:
#   $1 = CMake version (default: 4.2.0)
#   $2 = Install prefix (default: /usr/local)
install_cmake() {
  local cmake_version="${1:-4.2.0}"
  local cmake_major_minor=$(echo "$cmake_version" | cut -d. -f1,2)
  local prefix="${2:-/usr/local}"

  echo "Installing CMake $cmake_version..."

  wget "https://cmake.org/files/v${cmake_major_minor}/cmake-${cmake_version}.tar.gz"
  tar zxvf "cmake-${cmake_version}.tar.gz"
  cd "cmake-${cmake_version}"
  ./bootstrap --prefix="$prefix" --system-curl
  make -j$(nproc)
  make install
  cd ..
  rm -rf "cmake-${cmake_version}.tar.gz" "cmake-${cmake_version}"

  echo "CMake $cmake_version installation complete"
}

# ==============================================================================
# OPENSSL INSTALLATION
# ==============================================================================

# Install OpenSSL from source
# Arguments:
#   $1 = OpenSSL version (default: 3.4.0)
#   $2 = Install prefix (default: /usr)
install_openssl() {
  local openssl_version="${1:-3.4.0}"
  local prefix="${2:-/usr}"

  echo "Installing OpenSSL $openssl_version..."

  wget "https://github.com/openssl/openssl/releases/download/openssl-${openssl_version}/openssl-${openssl_version}.tar.gz"
  tar -xzvf "openssl-${openssl_version}.tar.gz"
  cd "openssl-${openssl_version}"
  ./config --prefix="$prefix" --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic
  make -j$(nproc)
  make install
  cd ..
  rm -rf "openssl-${openssl_version}.tar.gz" "openssl-${openssl_version}"

  echo "OpenSSL $openssl_version installation complete"
}

# ==============================================================================
# OPENCV EXTRAS
# ==============================================================================

# Download OpenCV extra files from Kitware data server
# (Downloads locally instead of from notoriously failing opencv repo)
download_opencv_extras() {
  echo "Downloading OpenCV extras..."
  curl https://data.kitware.com/api/v1/item/682bf0110dcd2dfb445a5404/download --output tmp.tar.gz
  tar -xvf tmp.tar.gz
  rm tmp.tar.gz
  echo "OpenCV extras download complete"
}

# ==============================================================================
# CUDNN PATCHING
# ==============================================================================

# Patch CUDNN headers for certain OS versions
# Creates symlinks from cudnn_v9.h files to expected cudnn.h names
patch_cudnn_headers() {
  echo "Checking CUDNN headers..."

  if [ -f /usr/include/cudnn_v9.h ] && [ ! -f /usr/include/cudnn.h ]; then
    echo "Patching CUDNN v9 headers..."
    ln -s /usr/include/cudnn_v9.h /usr/include/cudnn.h
    ln -s /usr/include/cudnn_adv_v9.h /usr/include/cudnn_adv.h
    ln -s /usr/include/cudnn_cnn_v9.h /usr/include/cudnn_cnn.h
    ln -s /usr/include/cudnn_ops_v9.h /usr/include/cudnn_ops.h
    ln -s /usr/include/cudnn_version_v9.h /usr/include/cudnn_version.h
    ln -s /usr/include/cudnn_backend_v9.h /usr/include/cudnn_backend.h
    ln -s /usr/include/cudnn_graph_v9.h /usr/include/cudnn_graph.h
    echo "CUDNN header patching complete"
  else
    echo "CUDNN headers OK (no patching needed)"
  fi
}

# ==============================================================================
# CUDA/CUDNN LIBRARY COPYING
# ==============================================================================

# Copy CUDA 12 runtime libraries
# Arguments:
#   $1 = CUDA base directory (e.g., /usr/local/cuda)
#   $2 = destination lib directory (e.g., install/lib)
copy_cuda_libraries() {
  local cuda_base="$1"
  local dest_lib="$2"

  if [ ! -d "$cuda_base" ]; then
    echo "CUDA directory not found: $cuda_base, skipping CUDA library copy"
    return 0
  fi

  echo "Copying CUDA libraries from $cuda_base to $dest_lib"

  # Core CUDA libraries
  cp -P "$cuda_base/lib64/libcudart.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcusparse.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcufft.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcusolver.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcublas.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcublasLt.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcupti.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libcurand.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libnvjpeg.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libnvJitLink.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libnvrtc"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/lib64/libnvToolsExt.so"* "$dest_lib" 2>/dev/null || true

  # Target-specific libraries
  cp -P "$cuda_base/targets/x86_64-linux/lib/libcupti.so"* "$dest_lib" 2>/dev/null || true
  cp -P "$cuda_base/targets/x86_64-linux/lib/libcufile.so"* "$dest_lib" 2>/dev/null || true

  # Ubuntu-specific NPP libraries
  local os_type=$(detect_os_type)
  if [ "$os_type" = "ubuntu" ]; then
    cp -P "$cuda_base/lib64/libnppi"* "$dest_lib" 2>/dev/null || true
    cp -P "$cuda_base/lib64/libnppc"* "$dest_lib" 2>/dev/null || true
  fi

  echo "CUDA library copy complete"
}

# Create PyTorch CUDA symlinks if needed
# Arguments:
#   $1 = install lib directory (e.g., install/lib)
#   $2 = Python version (default: 3.10)
create_pytorch_cuda_symlinks() {
  local install_lib="$1"
  local python_version="${2:-3.10}"
  local torch_base="$install_lib/python${python_version}/site-packages/torch"

  if [ -d "$torch_base" ]; then
    ln -sf ../../../../libcublas.so.12 "$torch_base/lib/libcublas.so.12" 2>/dev/null || true
    echo "Created PyTorch CUDA symlinks"
  fi
}

# Copy CUDNN 9 libraries and create symlinks
# Arguments:
#   $1 = CUDNN base directory (e.g., /usr/lib/x86_64-linux-gnu or /usr/lib64)
#   $2 = destination lib directory (e.g., install/lib)
copy_cudnn_libraries() {
  local cudnn_base="$1"
  local dest_lib="$2"

  if [ ! -d "$cudnn_base" ]; then
    echo "CUDNN directory not found: $cudnn_base, skipping CUDNN library copy"
    return 0
  fi

  echo "Copying CUDNN libraries from $cudnn_base to $dest_lib"

  # Copy CUDNN 9 libraries
  local cudnn_libs=(
    "libcudnn.so.9"
    "libcudnn_adv.so.9"
    "libcudnn_cnn.so.9"
    "libcudnn_ops.so.9"
    "libcudnn_engines_precompiled.so.9"
    "libcudnn_engines_runtime_compiled.so.9"
    "libcudnn_graph.so.9"
    "libcudnn_heuristic.so.9"
  )

  for lib in "${cudnn_libs[@]}"; do
    cp -P "$cudnn_base/${lib}"* "$dest_lib" 2>/dev/null || true
  done

  # Create clean symlinks (remove existing, then create)
  local cudnn_symlinks=(
    "libcudnn"
    "libcudnn_adv"
    "libcudnn_cnn"
    "libcudnn_ops"
    "libcudnn_engines_precompiled"
    "libcudnn_engines_runtime_compiled"
    "libcudnn_graph"
    "libcudnn_heuristic"
  )

  for lib in "${cudnn_symlinks[@]}"; do
    rm -f "$dest_lib/${lib}.so" 2>/dev/null || true
    ln -s "${lib}.so.9" "$dest_lib/${lib}.so" 2>/dev/null || true
  done

  echo "CUDNN library copy and symlink creation complete"
}

# Copy system runtime libraries for portability
# Arguments:
#   $1 = system lib base directory
#   $2 = destination lib directory
copy_system_runtime_libraries() {
  local lib_base="$1"
  local dest_lib="$2"

  echo "Copying system runtime libraries from $lib_base to $dest_lib"

  # Common system libraries needed for portability
  local system_libs=(
    "libcrypt.so.2"
    "libcrypt.so.2.0.0"
    "libffi.so.6"
    "libffi.so.7"
    "libffi.so.8"
    "libva.so.1"
    "libssl.so.10"
    "libssl.so.1.1"
    "libssl.so.3"
    "libreadline.so.6"
    "libreadline.so.7"
    "libreadline.so.8"
    "libdc1394.so.22"
    "libcrypto.so.10"
    "libcrypto.so.1.1"
    "libcrypto.so.3"
    "libpcre.so.1"
    "libpcre.so.3"
    "libgomp.so.1"
    "libSM.so.6"
    "libICE.so.6"
    "libblas.so.3"
    "liblapack.so.3"
    "libgfortran.so.4"
    "libgfortran.so.5"
    "libquadmath.so.0"
    "libpng15.so.15"
    "libxcb.so.1"
    "libXau.so.6"
  )

  for lib in "${system_libs[@]}"; do
    cp "$lib_base/$lib" "$dest_lib" 2>/dev/null || true
  done

  echo "System runtime library copy complete"
}

# Copy Ubuntu-specific runtime libraries
# Arguments:
#   $1 = system lib base directory (e.g., /usr/lib/x86_64-linux-gnu)
#   $2 = destination lib directory
copy_ubuntu_specific_libraries() {
  local lib_base="$1"
  local dest_lib="$2"

  echo "Copying Ubuntu-specific libraries to $dest_lib"

  # Libraries from /lib/x86_64-linux-gnu
  cp /lib/x86_64-linux-gnu/libreadline.so.6 "$dest_lib" 2>/dev/null || true
  cp /lib/x86_64-linux-gnu/libreadline.so.7 "$dest_lib" 2>/dev/null || true
  cp /lib/x86_64-linux-gnu/libpcre.so.3 "$dest_lib" 2>/dev/null || true
  cp /lib/x86_64-linux-gnu/libexpat.so.1 "$dest_lib" 2>/dev/null || true

  # Additional libraries from lib_base
  local ubuntu_libs=(
    "libcrypto.so"
    "libcrypto.so.1.1"
    "libfreetype.so.6"
    "libharfbuzz.so.0"
    "libpng16.so.16"
    "libglib-2.0.so.0"
    "libgraphite2.so.3"
  )

  for lib in "${ubuntu_libs[@]}"; do
    cp "$lib_base/$lib" "$dest_lib" 2>/dev/null || true
  done

  echo "Ubuntu-specific library copy complete"
}

# Full library setup - copies all needed libraries
# Arguments:
#   $1 = CUDA base directory (optional, default: /usr/local/cuda)
#   $2 = destination install lib directory (default: install/lib)
#   $3 = CUDNN base directory (optional, auto-detected if not provided)
#   $4 = system lib base directory (optional, auto-detected if not provided)
setup_all_libraries() {
  local cuda_base="${1:-/usr/local/cuda}"
  local dest_lib="${2:-install/lib}"
  local cudnn_base="$3"
  local lib_base="$4"

  # Auto-detect directories if not provided
  if [ -z "$cudnn_base" ]; then
    cudnn_base=$(get_default_lib_base)
  fi
  if [ -z "$lib_base" ]; then
    lib_base=$(get_default_lib_base)
  fi

  echo "Setting up libraries:"
  echo "  CUDA base: $cuda_base"
  echo "  CUDNN base: $cudnn_base"
  echo "  Lib base: $lib_base"
  echo "  Destination: $dest_lib"

  # Copy all library types
  if [ -d "$cuda_base" ]; then
    copy_cuda_libraries "$cuda_base" "$dest_lib"
    copy_cudnn_libraries "$cudnn_base" "$dest_lib"
    create_pytorch_cuda_symlinks "$(dirname $dest_lib)"
  fi

  copy_system_runtime_libraries "$lib_base" "$dest_lib"

  # Ubuntu-specific
  local os_type=$(detect_os_type)
  if [ "$os_type" = "ubuntu" ]; then
    copy_ubuntu_specific_libraries "$lib_base" "$dest_lib"
  fi

  echo "All library setup complete"
}

# ==============================================================================
# BUILD UTILITIES
# ==============================================================================

# Extract VIAME version from RELEASE_NOTES.md
# Arguments: $1 = path to VIAME source directory
# Sets: VIAME_VERSION environment variable
extract_viame_version() {
  local source_dir="${1:-/viame}"
  if [ -f "$source_dir/RELEASE_NOTES.md" ]; then
    export VIAME_VERSION=$(head -n 1 "$source_dir/RELEASE_NOTES.md" | awk '{print $1}')
    echo "Extracted VIAME version: $VIAME_VERSION"
  else
    echo "Warning: RELEASE_NOTES.md not found at $source_dir"
    export VIAME_VERSION="unknown"
  fi
}

# Verify build success by checking log file
# Arguments: $1 = path to build log file
# Returns: 0 on success, 1 on failure
verify_build_success() {
  local log_file="${1:-build_log.txt}"

  if [ ! -f "$log_file" ]; then
    echo "Error: Build log file not found: $log_file"
    return 1
  fi

  if grep -q "Built target viame" "$log_file"; then
    echo "VIAME Build Succeeded"
  else
    echo "VIAME Build Failed"
    return 1
  fi

  if grep -q "fixup_bundle: preparing..." "$log_file"; then
    echo "Fixup Bundle Called Successfully"
  else
    echo "Warning: Fixup Bundle Not Called (may be expected for some configurations)"
  fi

  return 0
}

# Fix libsvm symlink issue
# Arguments: $1 = path to install directory (default: install)
fix_libsvm_symlink() {
  local install_dir="${1:-install}"

  if [ -f "$install_dir/lib/libsvm.so.2" ]; then
    rm -f "$install_dir/lib/libsvm.so"
    cp "$install_dir/lib/libsvm.so.2" "$install_dir/lib/libsvm.so"
    echo "Fixed libsvm symlink"
  fi
}

# Prepare Linux desktop install by removing unneeded directories and adding LICENSE
# Arguments:
#   $1 = install directory (default: install)
#   $2 = source directory containing LICENSE.txt (default: ..)
prepare_linux_desktop_install() {
  local install_dir="${1:-install}"
  local source_dir="${2:-..}"

  echo "Preparing Linux desktop install..."

  # Remove directories not needed for desktop distribution
  local dirs_to_remove=(
    "sbin"
    "qml"
    "include"
    "mkspecs"
    "etc"
    "doc"
  )

  for dir in "${dirs_to_remove[@]}"; do
    if [ -d "$install_dir/$dir" ]; then
      rm -rf "$install_dir/$dir"
      echo "  Removed $dir"
    fi
  done

  # Remove share directory but preserve share/postgresql
  if [ -d "$install_dir/share" ]; then
    if [ -d "$install_dir/share/postgresql" ]; then
      mv "$install_dir/share/postgresql" "$install_dir/postgresql_temp"
    fi
    rm -rf "$install_dir/share"
    if [ -d "$install_dir/postgresql_temp" ]; then
      mkdir -p "$install_dir/share"
      mv "$install_dir/postgresql_temp" "$install_dir/share/postgresql"
    fi
    echo "  Removed share (preserved share/postgresql)"
  fi

  # Copy LICENSE.txt to install root
  if [ -f "$source_dir/LICENSE.txt" ]; then
    cp "$source_dir/LICENSE.txt" "$install_dir/"
    echo "  Copied LICENSE.txt to install root"
  else
    echo "  Warning: LICENSE.txt not found at $source_dir/LICENSE.txt"
  fi

  echo "Linux desktop install preparation complete"
}

# Create tarball of install directory
# Arguments:
#   $1 = version string
#   $2 = platform suffix (e.g., "Linux-64Bit", "Ubuntu-64Bit")
#   $3 = install directory (default: install)
create_install_tarball() {
  local version="$1"
  local platform="$2"
  local install_dir="${3:-install}"
  local tarball_name="VIAME-${version}-${platform}.tar.gz"

  if [ -d "$install_dir" ]; then
    mv "$install_dir" viame
    rm -f "$tarball_name" 2>/dev/null || true
    tar -zcvf "$tarball_name" viame
    mv viame "$install_dir"
    echo "Created tarball: $tarball_name"
  else
    echo "Error: Install directory not found: $install_dir"
    return 1
  fi
}

# Finalize Docker/server install by relocating to /opt/noaa
# Arguments:
#   $1 = build directory (default: /viame/build)
#   $2 = cleanup source (default: true)
finalize_docker_install() {
  local build_dir="${1:-/viame/build}"
  local cleanup="${2:-true}"

  if [ -f "$build_dir/install/setup_viame.sh" ]; then
    cd "$build_dir"
    rm -f build_log.txt
    mkdir -p /opt/noaa
    mv install viame
    mv viame /opt/noaa
    if [ "$cleanup" = "true" ]; then
      cd /
      rm -rf /viame
    fi
    chown -R 1099:1099 /opt/noaa/viame
    echo "Finalized install to /opt/noaa/viame"
  else
    echo "Warning: setup_viame.sh not found, skipping finalization"
  fi
}

# Update git submodules
# Arguments: $1 = source directory
update_git_submodules() {
  local source_dir="${1:-/viame}"

  echo "Checking out VIAME submodules"
  cd "$source_dir"
  git config --global --add safe.directory "$source_dir" 2>/dev/null || true
  git submodule update --init --recursive
}

# Create and enter build directory
# Arguments: $1 = source directory
setup_build_directory() {
  local source_dir="${1:-/viame}"

  cd "$source_dir"
  mkdir -p build
  cd build
  echo "Build directory: $(pwd)"
}

# Run multi-threaded build
# Arguments:
#   $1 = log file (default: build_log.txt)
#   $2 = continue on error (default: false)
run_build() {
  local log_file="${1:-build_log.txt}"
  local continue_on_error="${2:-false}"

  echo "Beginning build, routing output to $log_file"
  if [ "$continue_on_error" = "true" ]; then
    make -j$(nproc) > "$log_file" 2>&1 || true
  else
    make -j$(nproc) > "$log_file" 2>&1
  fi
}

# ==============================================================================
# HIGH-LEVEL BUILD WORKFLOW
# ==============================================================================

# Run full build and library setup
# Performs multi-threaded build and copies required runtime libraries
# Arguments:
#   $1 = CUDA base directory (default: /usr/local/cuda)
#   $2 = CUDNN base directory (auto-detected based on OS)
#   $3 = System lib base directory (auto-detected based on OS)
run_build_and_setup_libraries() {
  local cuda_base="${1:-/usr/local/cuda}"
  local cudnn_base="${2:-$(get_default_lib_base)}"
  local lib_base="${3:-$(get_default_lib_base)}"

  # Perform multi-threaded build
  make -j$(nproc)

  # Below be krakens
  # (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

  export CUDA_BASE="$cuda_base"
  export CUDNN_BASE="$cudnn_base"
  export LIB_BASE="$lib_base"

  # Fix libsvm symlink issue
  fix_libsvm_symlink install

  # Copy CUDA libraries if available
  copy_cuda_libraries "$CUDA_BASE" install/lib

  # Create PyTorch CUDA symlinks
  create_pytorch_cuda_symlinks install/lib

  # Copy CUDNN libraries and create symlinks
  if [ -d "$CUDA_BASE" ]; then
    copy_cudnn_libraries "$CUDNN_BASE" install/lib
  fi

  # Copy system runtime libraries
  copy_system_runtime_libraries "$LIB_BASE" install/lib

  # Ubuntu-specific library copies
  local os_type=$(detect_os_type)
  if [ "$os_type" = "ubuntu" ]; then
    copy_ubuntu_specific_libraries "$LIB_BASE" install/lib
  fi
}
