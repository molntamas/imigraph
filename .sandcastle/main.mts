/**
 * Sandcastle orchestrator: GitHub issues -> sandboxed Claude Code -> pull request.
 *
 *   npm run sandbox:run                      # open issues labelled sandcastle:todo (max 3), one after another
 *   npm run sandbox:run -- --issue 7         # one specific issue, whatever its labels
 *   npm run sandbox:run -- --parallel 2      # up to 2 agents at the same time (own branch, worktree, container)
 *   npm run sandbox:run -- --dry-run         # only list what would run
 *
 * Per issue:
 *   1. the issue is claimed: its sandcastle:* label becomes sandcastle:in-progress (host side, REST)
 *   2. sandcastle creates a worktree on branch ai/issue-<n> from origin/main under .sandcastle/worktrees/
 *   3. a container from the sandcastle:imigraph image runs Claude Code with .sandcastle/prompt.md
 *   4. the agent commits, pushes the branch, opens a pull request and relabels the issue (via gh)
 *   5. if the agent did not hand over, the issue is labelled sandcastle:blocked with a comment
 *
 * Only issues opened by trusted users (GITHUB_TRUSTED_USERS, default: the repo owner) are picked
 * up, and only their comments reach the prompt. Secrets come from .sandcastle/.env.
 */
import { run, claudeCode } from "@ai-hero/sandcastle";
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import {
  BASE_BRANCH,
  GITHUB_REPO,
  LABELS,
  MODEL,
  github,
  hooks,
  isTrusted,
  requireGitHubToken,
  sandbox,
} from "./config.mts";

requireGitHubToken();

const { values: args } = parseArgs({
  options: {
    issue: { type: "string" },
    parallel: { type: "string", default: "1" },
    "max-issues": { type: "string", default: "3" },
    "dry-run": { type: "boolean", default: false },
  },
});

interface Issue {
  number: number;
  title: string;
  body: string | null;
  html_url: string;
  user: { login: string };
  labels: { name: string }[];
  pull_request?: unknown;
}

interface Comment {
  body: string;
  user: { login: string };
  created_at: string;
}

// ---------- GitHub ----------
const fetchIssues = async (): Promise<Issue[]> => {
  if (args.issue) {
    const issue = await github<Issue>("GET", `/issues/${args.issue}`);
    if (issue.pull_request) throw new Error(`#${issue.number} is a pull request, not an issue.`);
    if (!isTrusted(issue.user.login)) {
      console.warn(`Warning: #${issue.number} was opened by @${issue.user.login}, not a trusted user. Running it because you named it.`);
    }
    return [issue];
  }
  const open = await github<Issue[]>(
    "GET",
    `/issues?state=open&labels=${encodeURIComponent(LABELS.todo.name)}&sort=created&direction=asc&per_page=100`,
  );
  const issues = open.filter((issue) => !issue.pull_request);
  for (const issue of issues.filter((i) => !isTrusted(i.user.login))) {
    console.warn(`Skipping #${issue.number}: opened by @${issue.user.login}, not a trusted user.`);
  }
  return issues.filter((issue) => isTrusted(issue.user.login)).slice(0, Number(args["max-issues"]));
};

const trustedComments = async (issue: Issue): Promise<string> => {
  const comments = await github<Comment[]>("GET", `/issues/${issue.number}/comments?per_page=100`);
  const kept = comments.filter((comment) => isTrusted(comment.user.login));
  if (kept.length === 0) return "(none)";
  return kept
    .map((comment) => `@${comment.user.login} on ${comment.created_at.slice(0, 10)}:\n\n${comment.body.trim()}`)
    .join("\n\n---\n\n");
};

// Replaces whatever sandcastle:* label the issue has with `label`; other labels stay.
const setStatus = async (issueNumber: number, label: string) => {
  const issue = await github<Issue>("GET", `/issues/${issueNumber}`);
  const labels = issue.labels.map((l) => l.name).filter((name) => !name.startsWith("sandcastle:"));
  await github("PUT", `/issues/${issueNumber}/labels`, { labels: [...labels, label] });
};

const comment = (issueNumber: number, body: string) =>
  github("POST", `/issues/${issueNumber}/comments`, { body });

// ---------- one sandboxed agent run per issue ----------
const workOnIssue = async (issue: Issue) => {
  const branch = `ai/issue-${issue.number}`;
  console.log(`\n> #${issue.number} "${issue.title}" -> branch ${branch}`);
  await setStatus(issue.number, LABELS.inProgress.name);

  const result = await run({
    name: `issue-${issue.number}`,
    sandbox: sandbox(),
    agent: claudeCode(MODEL),

    promptFile: "./.sandcastle/prompt.md",
    promptArgs: {
      ISSUE_NUMBER: String(issue.number),
      ISSUE_TITLE: issue.title,
      ISSUE_URL: issue.html_url,
      ISSUE_BODY: issue.body?.trim() || "(the issue has no description)",
      ISSUE_COMMENTS: await trustedComments(issue),
      BRANCH: branch,
      BASE_BRANCH,
      GITHUB_REPO,
    },

    // One agent invocation per issue. The agent itself pushes and opens the pull request.
    maxIterations: 1,

    // Dedicated branch + worktree per issue, forked from the freshly fetched origin/main (not
    // from whatever you have checked out). An existing ai/issue-<n> branch is reused as is, so a
    // rerun continues where the last run stopped.
    branchStrategy: { type: "branch", branch, baseBranch: `origin/${BASE_BRANCH}` },

    hooks: { sandbox: { onSandboxReady: [hooks.gitAuth, hooks.uvSync] } },

    // Give up if the agent produces no output for 15 minutes.
    idleTimeoutSeconds: 900,
    logging: { type: "file", path: `.sandcastle/logs/issue-${issue.number}.log` },
  });

  console.log(
    `OK #${issue.number}: ${result.commits.length} commit(s) on ${result.branch}, ` +
      `completion signal ${result.completionSignal ? "received" : "NOT received"}, log: ${result.logFilePath}`,
  );
  return result.completionSignal ? "the agent finished but did not relabel the issue" : "no completion signal";
};

// If the agent neither opened a PR nor reported itself blocked, the issue would sit in
// in-progress forever. Make it visible on GitHub instead.
const handOverIfStuck = async (issueNumber: number, reason: string) => {
  const issue = await github<Issue>("GET", `/issues/${issueNumber}`);
  if (!issue.labels.some((l) => l.name === LABELS.inProgress.name)) return;
  await setStatus(issueNumber, LABELS.blocked.name);
  await comment(
    issueNumber,
    `The sandcastle run ended without handing over (${reason}). ` +
      `See \`.sandcastle/logs/issue-${issueNumber}.log\` on the machine that ran it. ` +
      `Relabel \`${LABELS.todo.name}\` to retry; the branch \`ai/issue-${issueNumber}\` is reused.`,
  );
};

// ---------- main ----------
const issues = await fetchIssues();
if (issues.length === 0) {
  console.log(`No open issues labelled "${LABELS.todo.name}" from trusted users in ${GITHUB_REPO}. Nothing to do.`);
  process.exit(0);
}

console.log(`Found ${issues.length} issue(s):`);
for (const issue of issues) console.log(`  #${issue.number} ${issue.title}`);

if (args["dry-run"]) process.exit(0);

// New issue branches fork from origin/main; make sure it is current.
execFileSync("git", ["fetch", "origin", BASE_BRANCH], { stdio: "inherit" });

const parallel = Math.max(1, Number(args.parallel));
const queue = [...issues];
const failures: string[] = [];
await Promise.all(
  Array.from({ length: Math.min(parallel, queue.length) }, async () => {
    for (let issue = queue.shift(); issue; issue = queue.shift()) {
      let reason: string;
      try {
        reason = await workOnIssue(issue);
      } catch (error) {
        reason = error instanceof Error ? error.message : String(error);
        failures.push(`#${issue.number}: ${reason}`);
        console.error(`FAILED #${issue.number}:`, error);
      }
      await handOverIfStuck(issue.number, reason).catch((error) =>
        console.error(`Could not update #${issue.number} on GitHub:`, error),
      );
    }
  }),
);

if (failures.length > 0) {
  console.error(`\n${failures.length} run(s) failed:\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
console.log(`\nAll runs finished. Review the pull requests: https://github.com/${GITHUB_REPO}/pulls`);
