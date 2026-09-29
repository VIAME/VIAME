<#
.SYNOPSIS
Build a VIAME wheel for each python version, on Windows.

.DESCRIPTION
The Windows half of cmake/wheel/build_matrix.sh. One wheel per interpreter,
for the same reason: the extension modules use the full CPython API and link
python3X.dll, so there is no abi3 wheel to build instead.

Platform as well as version. A wheel carries its platform in its filename, so
win_amd64 and manylinux wheels are separate artifacts that have to be built on
the machine they are for -- this script produces the Windows column of that
matrix and build_matrix.sh produces the Linux one.

By default each version is a pinned python-build-standalone CPython that the
configure downloads and unpacks into that tree's install prefix. That way the
matrix needs no python installed but the one running CMake -- five side-by-side
Windows pythons is a machine setup step that mostly has not happened -- and
every machine builds against the same five interpreters.

-SystemPython instead finds interpreters with the `py` launcher when it is
present, since that is how side-by-side pythons are normally installed on
Windows, and by name on PATH otherwise.

**CUDA resolution differs from Linux and is worth knowing about.** Windows has
no RUNPATH, so nothing is patched into the binaries. `_add_windows_dll_
directories` in the viame package's __init__ calls `os.add_dll_directory` on
the `nvidia/*/bin` folders pip installed, at import time. That covers `import
viame`; a bare `viame.exe` run outside python may still need the CUDA DLLs on
PATH.

A version that fails does not stop the others.

.EXAMPLE
  .\cmake\wheel\build_matrix.ps1 -Build C:\wheels

.EXAMPLE
  .\cmake\wheel\build_matrix.ps1 -Build C:\wheels -Versions 3.10,3.11 -Jobs 8

.EXAMPLE
  .\cmake\wheel\build_matrix.ps1 -Build C:\wheels -SystemPython
#>
[CmdletBinding()]
param(
    [string]   $Source   = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [Parameter(Mandatory = $true)]
    [string]   $Build,
    [string[]] $Versions = @('3.10', '3.11', '3.12', '3.13', '3.14'),
    [int]      $Jobs     = 0,
    [string]   $Generator = 'Ninja',
    [switch]   $SystemPython,
    [string[]] $CMakeArg = @()
)

$ErrorActionPreference = 'Continue'

if (-not (Test-Path (Join-Path $Source 'CMakeLists.txt'))) {
    Write-Error "not a VIAME source tree: $Source"; exit 2
}

if ($Jobs -le 0) {
    $cores = [int]$env:NUMBER_OF_PROCESSORS
    if (-not $cores) { $cores = 2 }
    $Jobs = [Math]::Max(1, [int]($cores / 2))
}

function Find-Python([string] $version) {
    # The py launcher first: it is how side-by-side versions are installed.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $found = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { return $found.Trim() }
    }
    $cmd = Get-Command "python$version" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

New-Item -ItemType Directory -Force -Path $Build | Out-Null
$wheelhouse = Join-Path $Build 'wheelhouse'
New-Item -ItemType Directory -Force -Path $wheelhouse | Out-Null

Write-Host "source     : $Source"
Write-Host "build root : $Build"
Write-Host "versions   : $($Versions -join ' ')"
Write-Host "jobs       : $Jobs"
Write-Host ""

$results = @()

foreach ($version in $Versions) {
    $python = $null

    if ($SystemPython) {
        $python = Find-Python $version

        if (-not $python) {
            Write-Host "== python $version : not installed, skipping"
            $results += [pscustomobject]@{ Version = $version; Result = 'skipped'
                                           Detail = 'no interpreter found' }
            continue
        }
    }

    $tree   = Join-Path $Build "build-py$version"
    $log    = Join-Path $Build "build-py$version.log"
    $prefix = Join-Path $tree 'install'

    $origin = if ($python) { $python } else { 'python-build-standalone' }
    Write-Host "== python $version ($origin)"
    Write-Host "   tree $tree"
    Write-Host "   log  $log"

    New-Item -ItemType Directory -Force -Path $tree | Out-Null
    $started = Get-Date

    # The standalone module downloads the pinned CPython for this series into
    # the tree's install prefix, points FindPython at it and puts setuptools
    # in it -- `install_egg_info` needs that and a fresh CPython has none.
    #
    # With a system python, FindPython's default strategy is VERSION, which
    # takes the highest interpreter it can see rather than the one asked for,
    # so the location has to be pinned as well as named.
    $pythonArgs = if ($SystemPython) {
        @( "-DPython_EXECUTABLE=$python",
           '-DPython_FIND_STRATEGY=LOCATION',
           "-DPython3_EXECUTABLE=$python",
           "-DPYTHON_EXECUTABLE=$python" )
    } else {
        @( '-DVIAME_PYTHON_STANDALONE=ON',
           "-DVIAME_PYTHON_STANDALONE_VERSION=$version",
           '-DVIAME_BUILD_PYTHON_FROM_SOURCE=OFF' )
    }

    $configure = @(
        '-S', $Source, '-B', $tree,
        '-G', $Generator,
        '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_INSTALL_PREFIX=$prefix",
        '-DVIAME_ENABLE_PYTHON=ON'
    ) + $pythonArgs + $CMakeArg

    & cmake @configure               *> $log
    $ok = $LASTEXITCODE -eq 0
    if ($ok) { & cmake --build $tree --parallel $Jobs          *>> $log; $ok = $LASTEXITCODE -eq 0 }
    if ($ok) { & cmake --build $tree --target install          *>> $log; $ok = $LASTEXITCODE -eq 0 }
    if ($ok) { & cmake --build $tree --target wheel            *>> $log; $ok = $LASTEXITCODE -eq 0 }

    $elapsed = [int]((Get-Date) - $started).TotalSeconds

    if (-not $ok) {
        Write-Host "   FAILED after ${elapsed}s -- see $log"
        $results += [pscustomobject]@{ Version = $version; Result = 'failed'
                                       Detail = "${elapsed}s, see log" }
        continue
    }

    $wheel = Get-ChildItem (Join-Path $tree 'wheel\*.whl') -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if (-not $wheel) {
        Write-Host "   built, but produced no wheel -- see $log"
        $results += [pscustomobject]@{ Version = $version; Result = 'failed'
                                       Detail = 'built but no .whl' }
        continue
    }

    Copy-Item $wheel.FullName $wheelhouse
    Write-Host "   ok in ${elapsed}s -> $($wheel.Name)"
    $results += [pscustomobject]@{ Version = $version; Result = 'ok'
                                   Detail = $wheel.Name }
}

Write-Host ""
Write-Host "================================ summary ================================"
$results | Format-Table -AutoSize Version, Result, Detail | Out-String | Write-Host

$built = @($results | Where-Object { $_.Result -eq 'ok' }).Count
Write-Host "  $built wheel(s) in $wheelhouse"

# Non-zero only when nothing was produced: a version failing is usually that
# version not being supported yet, not the tree being broken.
if ($built -eq 0) { exit 1 } else { exit 0 }
