#!/bin/bash

set -e

# Configuration
ENV="production"
BINARY_NAME="camber"
INSTALL_DIR="${HOME}/.camber/bin"
DOWNLOAD_URL="https://cli.dev.camber.cloud"
# Override download URL for dev environment
if [ "$ENV" == "staging" ]; then
    DOWNLOAD_URL="https://cli.staging.camber.cloud"
elif [ "$ENV" == "production" ]; then
    DOWNLOAD_URL="https://cli.cambercloud.com"
fi
DEFAULT_VERSION=$(curl -s "$DOWNLOAD_URL/latest.txt")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [-v version] [-e environment]" 1>&2
    echo "  -v    Specify version to install (default: ${DEFAULT_VERSION})" 1>&2
    echo "  -h    Show this help message" 1>&2
    exit 1
}

# Parse command line arguments
while getopts "v:e:h" opt; do
    case "${opt}" in
        v)
            INSTALL_VERSION="${OPTARG}"
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# Set version - use default if not specified
INSTALL_VERSION="${INSTALL_VERSION:-$DEFAULT_VERSION}"

# Detect OS and architecture
detect_platform() {
    local os arch

    os=$(uname -s)
    arch=$(uname -m)

    # Normalize arch names
    case "$arch" in
        x86_64)  arch="x86_64" ;;
        aarch64) arch="arm64" ;;
        arm64)   arch="arm64" ;;
        *)       echo -e "${RED}Unsupported architecture: $arch${NC}" && exit 1 ;;
    esac

    # Check supported platforms
    case "$os" in
        Darwin|Linux) : ;; # Supported platforms
        *) echo -e "${RED}Unsupported operating system: $os${NC}" && exit 1 ;;
    esac

    echo "$os" "$arch"
}

# Download and install binary
install_binary() {
    local os=$1
    local arch=$2
    local tmp_dir

    echo -e "${BLUE}Installing ${BINARY_NAME} version ${YELLOW}${INSTALL_VERSION}${BLUE} for ${os}_${arch}...${NC}"

    # Create temp directory
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT

    # Download tarball
    echo -e "${BLUE}Downloading release...${NC}"
    local download_url="${DOWNLOAD_URL}/releases/${INSTALL_VERSION}/${BINARY_NAME}cli_${INSTALL_VERSION}_${os}_${arch}.tar.gz"
    if ! curl -fsSL "${download_url}" -o "${tmp_dir}/${BINARY_NAME}.tar.gz"; then
        echo -e "${RED}Failed to download ${BINARY_NAME}${NC}"
        exit 1
    fi

    # Extract tarball
    echo -e "${BLUE}Extracting archive...${NC}"
    tar -xzf "${tmp_dir}/${BINARY_NAME}.tar.gz" -C "$tmp_dir"

    # Create install directory
    mkdir -p "$INSTALL_DIR"

    echo -e "${BLUE}Installing binary to ${INSTALL_DIR}...${NC}"
    # Install to temporary name first, then atomically rename
    cp "${tmp_dir}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}.new"
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}.new"
    # Atomic rename - this should work even if the old binary is running
    mv "${INSTALL_DIR}/${BINARY_NAME}.new" "${INSTALL_DIR}/${BINARY_NAME}"
}

# Update shell RC file
update_rc() {
    path_export="export PATH=\"\$HOME/.camber/bin:\$PATH\""
    shell_name=$(basename "$SHELL")

    case "$shell_name" in
        zsh)
            rc_file="$HOME/.zshrc"
            ;;
        bash)
            rc_file="$HOME/.bashrc"
            # For macOS bash
            if [[ "$OSTYPE" == "darwin"* ]]; then
                rc_file="$HOME/.bash_profile"
            fi
            ;;
        *)
            echo -e "${RED}Unsupported shell: $shell_name. Please add ${INSTALL_DIR} to your PATH manually${NC}"
            return 1
            ;;
    esac

    if [ -f "$rc_file" ]; then
        if ! grep -q "$path_export" "$rc_file" 2>/dev/null; then
            echo -e "\n# Added by ${BINARY_NAME} installer" >> "$rc_file"
            echo "$path_export" >> "$rc_file"
            echo -e "${GREEN}Added ${INSTALL_DIR} to PATH in $(basename "$rc_file")${NC}"
        fi
    else
        echo -e "${YELLOW}RC file $rc_file not found. Creating it...${NC}"
        echo -e "# Added by ${BINARY_NAME} installer" > "$rc_file"
        echo "$path_export" >> "$rc_file"
        echo -e "${GREEN}Created and updated $rc_file${NC}"
    fi
}

main() {
    # Check for curl
    if ! command -v curl >/dev/null 2>&1; then
        echo -e "${RED}curl is required but not installed. Please install it first.${NC}"
        exit 1
    fi

    # Detect platform
    read -r os arch < <(detect_platform)
    
    # Install binary
    install_binary "$os" "$arch"
    
    # Update RC file
    update_rc
    
    echo -e "${GREEN}Installation complete!${NC}"
    echo -e "${BLUE}Please restart your shell or run:${NC}"
    echo -e "${BLUE}    source ~/.$(basename $SHELL)rc${NC}"
    echo -e "${GREEN}You can now use '${BINARY_NAME}' command.${NC}"
}

main "$@"
