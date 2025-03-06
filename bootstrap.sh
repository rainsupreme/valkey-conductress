#!/bin/bash

# Get config values
SERVER=$(python3 -c 'import config; print(config.server)')
SSH_KEY_FILE=$(python3 -c 'import config; print(config.sshkeyfile)')

REQUIRED_PYTHON_PACKAGES=(
    "plotext"
    "numerize"
)

# These files must be manually provided by the user - we ensure they exist
REQUIRED_FILES=(
    "./amz_valkey-benchmark"
    "./valkey-benchmark"
    "$SSH_KEY_FILE"
)

# list of repositories to clone on the server so they are available for testing
# format of each line is "url.git||folderName"
REPOSITORIES=(
    "https://github.com/valkey-io/valkey.git||valkey"
    "https://github.com/SoftlyRaining/valkey.git||SoftlyRaining"
    "https://github.com/valkey-io/valkey.git||zuiderkwast"
    "https://github.com/JimB123/valkey.git||JimB123"
)

# Exit on any error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Starting server bootstrap process..."

# Ensure pip is installed
if ! command_exists pip; then
    echo "pip not found. Installing pip..."
    
    # Install pip
    sudo yum install -y python3-pip
    
    # Sometimes pip might need upgrading
    sudo python3 -m pip install --upgrade pip
fi

echo "Installing required Python packages..."
for package in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    echo "Installing $package..."
    python3 -m pip install "$package"
done

echo "Checking for required files..."
missing_files=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Missing: $file"
        ((missing_files++)) || true
    fi
done
if [ $missing_files -gt 0 ]; then
    echo "$missing_files files missing"
    exit 1
fi

echo "Ensuring keyfile permissions"
chmod 600 "$SSH_KEY_FILE" || {
    echo "Failed to set permissions on $SSH_KEY_FILE"
    exit 1
}

echo "ensure server is in known-hosts"
if ssh-keygen -F "$SERVER" >/dev/null 2>&1; then
    echo "Fingerprint for $SERVER already exists in known_hosts"
else
    echo "Adding new fingerprint for $SERVER to known_hosts..."
    mkdir -p ~/.ssh
    ssh-keyscan -H "$SERVER" >> ~/.ssh/known_hosts 2>/dev/null
    echo "Fingerprint added"
fi

echo "Setting up packages on server ($SERVER)..."
ssh -i "$SSH_KEY_FILE" "ec2-user@$SERVER" << 'EOF'
    set -e  # Exit on error

    echo "Installing required packages..."
    sudo yum update -y
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake \
        cmake3 \
        git \
EOF

echo "Cloning repositories on server ($SERVER)..."
for repo_info in "${REPOSITORIES[@]}"; do
    repo_url=${repo_info%%||*}
    target_dir=${repo_info##*||}
    echo "Checking $target_dir..."
    
    ssh -T -i "$SSH_KEY_FILE" "ec2-user@$SERVER" << EOF
        set -e  # Exit on error
        test -d "$target_dir" || {
            echo "Cloning $repo_url into $target_dir..."
            git clone "$repo_url" "$target_dir"
        }
EOF
done

echo "Bootstrap complete!"