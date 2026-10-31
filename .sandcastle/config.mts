/**
 * Shared settings for the sandcastle scripts (main, interactive, smoke, setup-github).
 *
 * Secrets and per-developer overrides come from .sandcastle/.env (gitignored). Sandcastle forwards
 * only the keys listed in that file into the container.
 */
import { docker } from "@ai-hero/sandcastle/sandboxes/docker";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const ENV_FILE = resolve(".sandcastle/.env");
if (existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

export const GITHUB_REPO = process.env.GITHUB_REPO || "molntamas/imigraph";
export const GH_TOKEN = process.env.GH_TOKEN || undefined;
export const MODEL = process.env.SANDCASTLE_MODEL || "claude-sonnet-5-5";
export const IMAGE_NAME = "sandcastle:imigraph";
export const BASE_BRANCH = "main";

// The repository is public: anyone can open an issue or write a comment. Agents only act on
// issues and comments written by these GitHub logins (default: the repository owner).
const TRUSTED_USERS = new Set(
  (process.env.GITHUB_TRUSTED_USERS || GITHUB_REPO.split("/")[0])
    .split(",")
    .map((login) => login.trim().toLowerCase())
    .filter(Boolean),
);
export const isTrusted = (login: string | undefined) =>
  login !== undefined && TRUSTED_USERS.has(login.toLowerCase());

// GitHub has no scoped labels, so the scripts swap these explicitly (see setStatus in main.mts).
export const LABELS = {
  todo: { name: "sandcastle:todo", color: "1D76DB", description: "Ready for a sandcastle agent run" },
  inProgress: { name: "sandcastle:in-progress", color: "FBCA04", description: "A sandcastle agent is working on it" },
  inReview: { name: "sandcastle:in-review", color: "0E8A16", description: "Agent opened a pull request, waiting for review" },
  blocked: { name: "sandcastle:blocked", color: "D93F0B", description: "Agent could not finish, see the issue comments" },
} as const;

// GH_REPO makes every gh command inside the container target this repository.
// GIT_CONFIG_*: git settings for the container process only (nothing is written to the .git
// directory, which the worktree shares with the host checkout):
//   - safe.directory=*: the bind-mounted worktree belongs to another uid as far as git is
//     concerned; sandcastle adds it to the global config, but betterleaks (pre-commit hook) runs
//     git without the global config, so the environment form is needed too;
//   - core.hooksPath=docker/git-hooks: commits run the LF hook of the repository instead of the
//     hook a Windows host installed into the shared .git/hooks (CRLF, Windows interpreter path,
//     cannot run on Linux).
export const sandbox = () =>
  docker({
    imageName: IMAGE_NAME,
    env: {
      GH_REPO: GITHUB_REPO,
      GIT_CONFIG_COUNT: "2",
      GIT_CONFIG_KEY_0: "safe.directory",
      GIT_CONFIG_VALUE_0: "*",
      GIT_CONFIG_KEY_1: "core.hooksPath",
      GIT_CONFIG_VALUE_1: "docker/git-hooks",
    },
  });

export const hooks = {
  // git push/fetch authenticate with GH_TOKEN through a credential helper that reads the variable
  // when git asks, so the token is never written to disk. Container-local git config only.
  gitAuth: {
    command:
      `[ -z "$GH_TOKEN" ] || git config --global credential.https://github.com.helper ` +
      `'!f() { [ "$1" = get ] && printf "username=x-access-token\\npassword=%s\\n" "$GH_TOKEN"; }; f'`,
  },
  // Project + any dependencies added since the image was built (the image pre-installs uv.lock).
  uvSync: { command: "uv sync --frozen", timeoutMs: 300_000 },
  // Commit hooks need no installation: core.hooksPath (see sandbox()) points at docker/git-hooks.
};

// ---------- GitHub REST, host side ----------
export class GitHubError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export const requireGitHubToken = () => {
  if (GH_TOKEN) return;
  console.error("GH_TOKEN is not set. Copy .sandcastle/.env.example to .sandcastle/.env and fill it in.");
  process.exit(1);
};

export async function github<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
  if (!GH_TOKEN) throw new Error("GH_TOKEN is not set");
  const response = await fetch(`https://api.github.com/repos/${GITHUB_REPO}${path}`, {
    method,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${GH_TOKEN}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new GitHubError(response.status, `GitHub ${method} ${path} -> HTTP ${response.status}: ${text.slice(0, 300)}`);
  }
  return (text ? JSON.parse(text) : undefined) as T;
}
