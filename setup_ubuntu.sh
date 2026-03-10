#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ASSET_DIR="$SCRIPT_DIR/assets"

echo "=========================================="
echo " VulkanRenderVB - Ubuntu Setup Script"
echo "=========================================="

# ── 1. Install dependencies ──────────────────────────────────────────────

echo ""
echo "[1/3] Installing required packages..."
echo ""

sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libwayland-dev wayland-protocols libxkbcommon-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    vulkan-tools libvulkan-dev glslang-tools \
    mesa-vulkan-drivers

# Ubuntu 24.04+ renamed the validation layers package
if apt-cache show vulkan-validationlayers-dev &>/dev/null; then
    sudo apt-get install -y vulkan-validationlayers-dev
else
    sudo apt-get install -y vulkan-utility-libraries-dev vulkan-validationlayers 2>/dev/null || true
fi

echo ""
echo "[OK] All packages installed."

# ── 2. Download Sponza assets ────────────────────────────────────────────

echo ""
echo "[2/3] Downloading Sponza assets..."
echo ""

if [ -f "$ASSET_DIR/Sponza/Sponza.gltf" ]; then
    echo "[OK] Sponza assets already present, skipping download."
else
    mkdir -p "$ASSET_DIR"
    TEMP_CLONE="$ASSET_DIR/glTF-Sample-Assets"
    git clone --depth 1 https://github.com/KhronosGroup/glTF-Sample-Assets.git "$TEMP_CLONE"
    cp -r "$TEMP_CLONE/Models/Sponza/glTF" "$ASSET_DIR/Sponza"
    rm -rf "$TEMP_CLONE"
    echo "[OK] Sponza assets downloaded to $ASSET_DIR/Sponza/"
fi

# ── 3. Configure and build ───────────────────────────────────────────────

echo ""
echo "[3/3] Building VulkanRenderVB..."
echo ""

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Symlink assets into build directory so the binary can find them
ln -sfn "$ASSET_DIR" "$BUILD_DIR/assets"

echo ""
echo "=========================================="
echo " Build complete!"
echo ""
echo " To run:"
echo "   cd $BUILD_DIR"
echo "   ./VulkanRenderVB"
echo "=========================================="
