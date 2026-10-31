# Sandbox image for sandcastle agents working on imigraph.
#
# Build context is the REPO ROOT (see .dockerignore), so the image can pre-install the locked
# Python dependencies and the pre-commit hook environments:
#
#   npm run sandbox:build
#   (= sandcastle docker build-image --dockerfile .sandcastle/Dockerfile --image-name sandcastle:imigraph)
#
# What is inside: Node 22 (Claude Code, npm tooling for the Pyodide/PyScript GUI), git, curl, jq,
# gh (GitHub CLI), uv + Python 3.14 with the project's locked dependencies (numpy, scipy, networkx,
# pytest, ruff), pre-commit (+ betterleaks), Claude Code CLI. The agent runs as the non-root user
# "agent" (uid 1000). Sandcastle bind-mounts the repo worktree at /home/agent/workspace.
#
# Rebuild after changing .pre-commit-config.yaml, and after larger dependency changes in uv.lock
# (the `uv sync` hook at sandbox start installs the difference, but that costs time on every run).

FROM node:22-bookworm

ARG GH_VERSION=2.103.3
ARG UV_VERSION=0.13.6

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# System libraries of headless Chromium (docker/check-pages.py verifies the generated pages and
# the Pyodide GUI with Playwright). The browser itself is installed below, as the agent user,
# in the version that matches the locked `playwright` package.
ARG PLAYWRIGHT_VERSION=1.64.1
RUN npx --yes "playwright@${PLAYWRIGHT_VERSION}" install-deps chromium \
    && rm -rf /root/.npm /var/lib/apt/lists/*

# GitHub CLI: the agent uses it to read issues, open pull requests, comment and relabel.
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${arch}.tar.gz" \
      | tar -xz -C /tmp \
    && install -m 0755 "/tmp/gh_${GH_VERSION}_linux_${arch}/bin/gh" /usr/local/bin/gh \
    && rm -rf /tmp/* \
    && gh --version

# Build-args for UID/GID alignment (sandcastle passes the host uid/gid on Linux/macOS;
# on Windows hosts the default 1000 is used).
ARG AGENT_UID=1000
ARG AGENT_GID=1000

# Rename the base image's "node" user to "agent" and align UID/GID.
RUN groupmod -o -g $AGENT_GID node \
    && usermod -o -u $AGENT_UID -g $AGENT_GID -d /home/agent -m -l agent node
USER ${AGENT_UID}:${AGENT_GID}
ENV HOME=/home/agent
ENV PATH="/home/agent/.local/bin:$PATH"
# gh/git never wait for input: without GH_TOKEN they fail instead of hanging the agent.
ENV GH_PROMPT_DISABLED=1 GH_NO_UPDATE_NOTIFIER=1 GH_SPINNER_DISABLED=1 GIT_TERMINAL_PROMPT=0 NO_COLOR=1
# Linux venv outside the bind-mounted repo, so a host .venv in the same checkout stays intact.
ENV UV_PROJECT_ENVIRONMENT=/home/agent/venv UV_LINK_MODE=copy

# Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# uv (pinned)
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && uv --version

# Python (version from .python-version) and the locked dependencies, pre-installed into
# /home/agent/venv. The `uv sync --frozen` hook at sandbox start then only adds the project itself.
COPY --chown=${AGENT_UID}:${AGENT_GID} pyproject.toml uv.lock .python-version /tmp/prewarm/
RUN cd /tmp/prewarm && uv sync --frozen --no-install-project && rm -rf /tmp/prewarm

# Headless Chromium for the locked playwright package (dev dependency group), into
# /home/agent/.cache/ms-playwright. Rebuild after bumping playwright in uv.lock.
RUN /home/agent/venv/bin/playwright install chromium

# pre-commit, with the hook environments pre-warmed. betterleaks is built from source with a Go
# toolchain that pre-commit downloads; doing it once here keeps the agent's first commit fast.
RUN uv tool install pre-commit
COPY --chown=${AGENT_UID}:${AGENT_GID} .pre-commit-config.yaml /tmp/prewarm/.pre-commit-config.yaml
RUN cd /tmp/prewarm && git init -q && pre-commit install-hooks && rm -rf /tmp/prewarm

WORKDIR /home/agent

# sandcastle bind-mounts the git worktree at /home/agent/workspace and overrides the
# working directory at container start. The container just idles; sandcastle execs into it.
ENTRYPOINT ["sleep", "infinity"]
