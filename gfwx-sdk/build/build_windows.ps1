# Build script for GFWX SDK wrapper on Windows
# Automatically finds CMake and Visual Studio

param(
    [string]$Configuration = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== GFWX SDK Build Script (Windows) ===" -ForegroundColor Cyan

# Function to find CMake
function Find-CMake {
    # Check if cmake is in PATH
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmake) {
        return $cmake.Source
    }

    # Common CMake locations
    $searchPaths = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2019\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )

    foreach ($pattern in $searchPaths) {
        $found = Get-Item $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    return $null
}

# Function to find Visual Studio
function Find-VSGenerator {
    # Check for VS 2022
    $vs2022Paths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\*",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*"
    )
    foreach ($pattern in $vs2022Paths) {
        if (Get-Item $pattern -ErrorAction SilentlyContinue) {
            return "Visual Studio 17 2022"
        }
    }

    # Check for VS 2019
    $vs2019Paths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2019\*",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\*"
    )
    foreach ($pattern in $vs2019Paths) {
        if (Get-Item $pattern -ErrorAction SilentlyContinue) {
            return "Visual Studio 16 2019"
        }
    }

    # Fall back to Ninja if available
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        return "Ninja"
    }

    return $null
}

# Find CMake
$cmake = Find-CMake
if (-not $cmake) {
    Write-Host "ERROR: CMake not found!" -ForegroundColor Red
    Write-Host "Please install CMake from https://cmake.org/download/" -ForegroundColor Yellow
    Write-Host "Or install Visual Studio with C++ workload (includes CMake)" -ForegroundColor Yellow
    exit 1
}
Write-Host "Found CMake: $cmake" -ForegroundColor Green

# Find generator
$generator = Find-VSGenerator
if (-not $generator) {
    Write-Host "ERROR: No suitable build system found!" -ForegroundColor Red
    Write-Host "Please install Visual Studio with C++ workload" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using generator: $generator" -ForegroundColor Green

# Set up build directory
$buildDir = Join-Path $ScriptDir "out"
$outputDir = Join-Path $ScriptDir $Configuration

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}

# Configure
Write-Host "`nConfiguring..." -ForegroundColor Cyan
$configArgs = @("-B", $buildDir, "-S", $ScriptDir, "-G", $generator)
if ($generator -like "Visual Studio*") {
    $configArgs += @("-A", "x64")
}
& $cmake @configArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed!" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "`nBuilding ($Configuration)..." -ForegroundColor Cyan
& $cmake --build $buildDir --config $Configuration
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Copy output
Write-Host "`nCopying output..." -ForegroundColor Cyan
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Find and copy the DLL
$dllPath = Get-ChildItem -Path $buildDir -Recurse -Filter "gfwx.dll" | Select-Object -First 1
if ($dllPath) {
    Copy-Item $dllPath.FullName -Destination $outputDir -Force
    Write-Host "Output: $(Join-Path $outputDir 'gfwx.dll')" -ForegroundColor Green
} else {
    Write-Host "WARNING: gfwx.dll not found in build output" -ForegroundColor Yellow
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Cyan
