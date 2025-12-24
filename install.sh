#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$HOME/.local/share/rembg"
BIN_DIR="$HOME/.local/bin"

echo "Installing rembg to $INSTALL_DIR..."

# Create installation directory
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy scripts
cp -v src/*.py "$INSTALL_DIR/"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"

# Install dependencies
echo "Installing dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r requirements.txt

# Create wrapper script
cat > "$BIN_DIR/rembg" << 'EOF'
#!/usr/bin/env bash
REMBG_DIR="$HOME/.local/share/rembg"
exec "$REMBG_DIR/venv/bin/python" "$REMBG_DIR/remove-bg.py" "$@"
EOF

chmod +x "$BIN_DIR/rembg"

echo "✓ Installation complete!"
echo "  Scripts: $INSTALL_DIR"
echo "  Command: $BIN_DIR/rembg (ensure ~/.local/bin is in PATH)"
echo ""
echo "Usage: rembg [options] <files...>"
echo "       fish function 'rembg' provides extended functionality"
