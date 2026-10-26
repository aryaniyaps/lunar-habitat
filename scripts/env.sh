# Source me: shared tooling paths (see ~/robotics-tooling/README.md)
tooling_dir="${ROBOTICS_TOOLING_DIR:-$HOME/robotics-tooling}"
export UV_CACHE_DIR="$tooling_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$tooling_dir/.python"
export UV_PROJECT_ENVIRONMENT="$tooling_dir/project-envs/lunar-habitat"
export PATH="$tooling_dir/bin:$PATH"
