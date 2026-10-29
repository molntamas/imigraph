/**
 * A normal Claude Code session from your terminal, but running inside the sandbox.
 *
 *   npm run sandbox:shell                                # Claude works on your current checkout (bind-mounted)
 *   npm run sandbox:shell -- --branch ai/experiment      # isolated worktree + branch instead of your checkout
 *   npm run sandbox:shell -- --permission-mode default   # ask before every tool call (default here: acceptEdits)
 *   npm run sandbox:shell -- --model claude-opus-5-5     # override SANDCASTLE_MODEL for this session
 *   npm run sandbox:shell -- --prompt "explain the code" # start the session with an initial prompt
 *
 * No GitHub issues involved. What Claude sees: the repository files, the container, and the
 * variables from .sandcastle/.env (with GH_TOKEN set it can `git push` and use `gh`). What it does
 * not see: your home directory, SSH keys, other repos, your shell. Exit Claude (/exit or Ctrl+C)
 * and the container is removed.
 *
 * With the default (head) mode, edits land directly in your working directory and commits on
 * your current branch, exactly like a local session. With --branch, sandcastle creates a worktree
 * under .sandcastle/worktrees/ and keeps your checkout untouched.
 */
import { interactive, claudeCode } from "@ai-hero/sandcastle";
import { parseArgs } from "node:util";
import { MODEL, hooks, sandbox } from "./config.mts";

const { values: args } = parseArgs({
  options: {
    branch: { type: "string" },
    "permission-mode": { type: "string", default: "acceptEdits" },
    model: { type: "string", default: MODEL },
    prompt: { type: "string" },
  },
});

type PermissionMode = "default" | "acceptEdits" | "plan" | "auto" | "dontAsk" | "bypassPermissions";

const result = await interactive({
  name: "shell",
  sandbox: sandbox(),
  agent: claudeCode(args.model, { permissionMode: args["permission-mode"] as PermissionMode }),
  prompt: args.prompt,
  branchStrategy: args.branch ? { type: "branch", branch: args.branch } : { type: "head" },
  hooks: { sandbox: { onSandboxReady: [hooks.gitAuth, hooks.uvSync] } },
});

console.log(
  `Session ended (exit ${result.exitCode}). ${result.commits.length} commit(s) on ${result.branch}.` +
    (result.preservedWorktreePath ? ` Uncommitted changes kept in ${result.preservedWorktreePath}.` : ""),
);
