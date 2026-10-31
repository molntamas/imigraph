#!/usr/bin/env bash
# Entrypoint of the dev/web services in compose.yaml. Prepares the container for the bind-mounted
# repository, then runs the given command.
set -euo pipefail

# The bind mount belongs to another uid as far as git is concerned, and Windows mounts report
# every file as executable: ignore both, as git on Windows does.
git config --global --add safe.directory /home/agent/workspace  # compose.yaml also sets it via GIT_CONFIG_*
git config --global core.fileMode false

# Commit identity: copied from the host's ~/.gitconfig if compose mounted it (read-only).
if [ -f /home/agent/host.gitconfig ]; then
  for key in user.name user.email; do
    value="$(git config -f /home/agent/host.gitconfig --get "$key" || true)"
    [ -n "$value" ] && git config --global "$key" "$value"
  done
fi

# The image has the locked dependencies; this adds the project itself (editable) and anything
# added to uv.lock since the image was built.
uv sync --frozen --quiet

# Commits made in the container run the hooks from docker/git-hooks (LF, Linux interpreter);
# the hook the host installed into .git/hooks is left alone.
git config --global core.hooksPath docker/git-hooks

exec "$@"
