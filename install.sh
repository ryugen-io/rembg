#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== rembg Installation ===${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# XDG-compliant directories
INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/rembg"
BIN_DIR="$HOME/.local/bin"

echo -e "${BLUE}Installing to $INSTALL_DIR${NC}"

# Create installation directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy pipeline package and CLI
cp -rv src/pipeline "$INSTALL_DIR/"
cp -v src/remove-bg-pipeline.py "$INSTALL_DIR/"

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
python3 -m venv "$INSTALL_DIR/venv"

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q

# Create wrapper script
echo -e "${GREEN}Creating wrapper script...${NC}"
cat > "$BIN_DIR/rembg" << 'EOF'
#!/usr/bin/env bash
REMBG_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/rembg"
exec "$REMBG_DIR/venv/bin/python" "$REMBG_DIR/remove-bg-pipeline.py" "$@"
EOF

chmod +x "$BIN_DIR/rembg"

echo ""
echo -e "${GREEN}✓ Installation complete!${NC}"
echo ""
echo -e "  Pipeline: ${BLUE}$INSTALL_DIR${NC}"
echo -e "  Command:  ${BLUE}$BIN_DIR/rembg${NC}"
echo ""
echo "Usage: rembg [options] <files...>"
echo "       fish function 'rembg' provides extended functionality"
echo ""
echo -e "${BLUE}Note:${NC} Make sure ~/.local/bin is in your PATH"
echo "  Add to your shell config: export PATH=\"\$HOME/.local/bin:\$PATH\""
