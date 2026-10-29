/**
 * One-time GitHub setup for the sandcastle workflow: creates (or updates) the sandcastle:* labels.
 *
 *   npm run sandbox:setup-github
 *
 * Idempotent. Uses GH_TOKEN and GITHUB_REPO from .sandcastle/.env. Once the labels exist, the
 * "Agent task" issue form (.github/ISSUE_TEMPLATE/agent-task.yml) applies sandcastle:todo itself.
 */
import { GITHUB_REPO, GitHubError, LABELS, github, requireGitHubToken } from "./config.mts";

requireGitHubToken();

for (const label of Object.values(LABELS)) {
  try {
    await github("POST", "/labels", label);
    console.log(`label created: ${label.name}`);
  } catch (error) {
    // 422 = a label with this name exists already; bring colour and description up to date.
    if (!(error instanceof GitHubError && error.status === 422)) throw error;
    await github("PATCH", `/labels/${encodeURIComponent(label.name)}`, {
      color: label.color,
      description: label.description,
    });
    console.log(`label updated: ${label.name}`);
  }
}

console.log(`\nDone. Create agent tasks at https://github.com/${GITHUB_REPO}/issues/new/choose`);
