#!/bin/bash
YUM_OPTS="--setopt=subscription-manager.disable=1 --disableplugin=subscription-manager"

# Get config values
SERVERS=$(python3 -c 'import config; print(" ".join(config.servers))')
SSH_KEY_FILE=$(python3 -c 'import config; print(config.sshkeyfile)')

REQUIRED_PYTHON_PACKAGES=(
    "plotext"
    "numerize"
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

echo "⊹˚₊‧───Starting bootstrap/update───‧₊˚⊹"

# remove default motd spam
if [ -f /etc/motd.d/insights-client ]; then
    echo "Removing insights-client motd"
    sudo rm -f /etc/motd.d/insights-client
fi

echo "Installing/updating required distro packages..."
sudo yum $YUM_OPTS update -y
sudo yum $YUM_OPTS groupinstall -y "Development Tools"
sudo yum $YUM_OPTS install -y \
    cmake \
    cmake3 \
    git \
    python3-pip \
    perf \
    js-d3-flame-graph
python3 -m pip install --upgrade pip

echo "Installing required Python packages..."
for package in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    echo "Installing $package..."
    python3 -m pip install "$package"
done

echo "Checking for required files..."
# SSH keyfile
if [ ! -f "$SSH_KEY_FILE" ]; then
    echo "Missing SSH keyfile: '$SSH_KEY_FILE'"
    echo "This must be manually copied to the server."
    exit 1
fi

echo "Ensuring keyfile permissions"
chmod 600 "$SSH_KEY_FILE" || {
    echo "Failed to set permissions on $SSH_KEY_FILE"
    exit 1
}

echo "Checking for required binaries..."
BUILDABLE_FILES=(
    "./valkey-cli"
    "./valkey-benchmark"
)

missing_files=0
for file in "${BUILDABLE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Missing: $file"
        ((missing_files++)) || true
    fi
done
if [ $missing_files -gt 0 ]; then
    echo "retrieving and building needed binaries"
    test -d "$target_dir" || {
        git clone https://github.com/valkey-io/valkey.git valkey
    }

    cd valkey
    git pull
    make distclean
    make -j
    cd ..

    echo "copying needed binaries"
    cp ./valkey/src/valkey-cli .
    cp ./valkey/src/valkey-benchmark .
fi

for SERVER in $SERVERS; do
    echo "ensure server $SERVER is in known-hosts"
    if ssh-keygen -F "$SERVER" >/dev/null 2>&1; then
        echo "Fingerprint for $SERVER already exists in known_hosts"
    else
        echo "Adding new fingerprint for $SERVER to known_hosts..."
        mkdir -p ~/.ssh
        ssh-keyscan -H "$SERVER" -T 10 >> ~/.ssh/known_hosts 2>/dev/null
        echo "Fingerprint added"
    fi

    if ! ssh -i "$SSH_KEY_FILE" -q "ec2-user@$SERVER" exit 2>/dev/null; then
        echo "Error: Cannot connect to $SERVER" >&2
        exit 1
    fi

    echo "Setting up packages on server $SERVER..."
    ssh -i "$SSH_KEY_FILE" "ec2-user@$SERVER" << EOF
        set -e  # Exit on error

        # remove default motd spam
        if [ -f /etc/motd.d/insights-client ]; then
            echo "Removing insights-client motd"
            sudo rm -f /etc/motd.d/insights-client
        fi

        echo "Installing required packages..."
        sudo yum $YUM_OPTS update -y
        sudo yum $YUM_OPTS groupinstall -y "Development Tools"
        sudo yum $YUM_OPTS install -y \
            cmake \
            cmake3 \
            git \
            perf
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
done

echo "Bootstrap complete!"