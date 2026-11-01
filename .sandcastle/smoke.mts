/**
 * Smoke test / "doctor" for the sandbox setup. No GitHub writes.
 *
 *   npm run sandbox:smoke              # worktree + container + toolchain + tests (+ GitHub auth if GH_TOKEN is set), no AI
 *   npm run sandbox:smoke -- --claude  # additionally one minimal Claude Code turn (verifies the Claude token)
 *
 * The worktree is created from your committed HEAD (uncommitted files are not in it) on a fresh
 * ai/smoke-<timestamp> branch. Container, worktree and branch are removed afterwards.
 */
import { createSandbox, run, claudeCode } from "@ai-hero/sandcastle";
import { execFileSync } from "node:child_process";
import { rmSync } from "node:fs";
import { GH_TOKEN, GITHUB_REPO, MODEL, github, hooks, sandbox } from "./config.mts";

const withClaude = process.argv.includes("--claude");

const cleanup = (branch: string) => {
  // close() unregisters the worktree; directory and branch stay behind (so real runs can resume).
  // On Windows the bind mount is released a moment after the container stops, so retry.
  const worktreeDir = `.sandcastle/worktrees/${branch.replace("/", "-")}`;
  try {
    rmSync(worktreeDir, { recursive: true, force: true, maxRetries: 20, retryDelay: 500 });
  } catch (error) {
    console.warn(`Could not remove ${worktreeDir} yet (${(error as Error).message}); delete it manually.`);
  }
  execFileSync("git", ["worktree", "prune"], { stdio: "ignore" });
  execFileSync("git", ["branch", "-D", branch], { stdio: "ignore" });
};

let failed = 0;

// ---- 1. host: GitHub API with the token ----
if (GH_TOKEN) {
  try {
    const repo = await github<{ full_name: string; open_issues_count: number }>("GET", "");
    console.log(`$ GitHub API (host)\n${repo.full_name}: reachable with GH_TOKEN, ${repo.open_issues_count} open issue(s)/PR(s)`);
  } catch (error) {
    failed += 1;
    console.error(`GitHub API (host) failed: ${(error as Error).message}`);
  }
} else {
  console.log(`GH_TOKEN is not set: skipping the GitHub checks for ${GITHUB_REPO}.`);
}

// ---- 2. container + toolchain + repo, no AI ----
{
  const branch = `ai/smoke-${Date.now()}`;
  const sb = await createSandbox({
    branch,
    sandbox: sandbox(),
    hooks: { sandbox: { onSandboxReady: [hooks.gitAuth, hooks.uvSync] } },
  });
  const checks = [
    "pwd && whoami && git branch --show-current && git log --oneline -1",
    "git status --short | head -5",
    "claude --version && gh --version | head -1 && uv --version && pre-commit --version && node --version",
    `uv run python -c "import sys, numpy, scipy, networkx; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| networkx', networkx.__version__)"`,
    "uv run pytest",
    "uv run ruff check . && uv run ruff format --check . && uv run ty check",
    // A commit inside the sandbox runs the pre-commit hooks through docker/git-hooks
    // (core.hooksPath from the container environment, see config.mts); then undone.
    "echo smoke > smoke.txt && git add smoke.txt && git commit -q -m 'smoke: hooks run in the sandbox' " +
      "&& git log --oneline -1 && git reset -q --hard HEAD~1 && git status --short | head -3",
    // The generated pages and the Pyodide GUI, checked with the headless Chromium of the image.
    "uv run python -m imigraph report --out output/smoke-report.html > /dev/null && " +
      "uv run python -m imigraph gas-report --out output/gas_smoke.html --steps 500 --lattice-steps 80 > /dev/null && " +
      "uv run python docker/check-pages.py --report output/smoke-report.html --serve --out output/smoke-screenshots",
  ];
  if (GH_TOKEN) {
    checks.push(
      "gh auth status",
      "gh issue list --limit 3",
      // GitHub requires write access for the push handshake, even with --dry-run.
      `git push --dry-run origin HEAD:refs/heads/${branch}`,
    );
  }
  try {
    for (const command of checks) {
      const result = await sb.exec(command);
      console.log(`\n$ ${command}\n${(result.stdout + result.stderr).trim()}`);
      if (result.exitCode !== 0) {
        failed += 1;
        console.log(`(exit code ${result.exitCode})`);
      }
    }
  } finally {
    await sb.close();
    cleanup(branch);
  }
}

// ---- 3. optional: one trivial Claude Code turn inside the sandbox ----
if (withClaude) {
  const branch = `ai/smoke-claude-${Date.now()}`;
  console.log(`\n$ claude (model ${MODEL}) inside the sandbox ...`);
  try {
    const result = await run({
      name: "smoke-claude",
      sandbox: sandbox(),
      agent: claudeCode(MODEL),
      prompt:
        "This is a connectivity test. Do not read or modify any files. " +
        "Reply with exactly <promise>COMPLETE</promise> and nothing else.",
      maxIterations: 1,
      branchStrategy: { type: "branch", branch },
      idleTimeoutSeconds: 180,
      logging: { type: "stdout", verbose: false },
    });
    console.log(result.completionSignal ? "Claude answered with the completion signal." : "No completion signal received.");
    if (!result.completionSignal) failed += 1;
  } catch (error) {
    failed += 1;
    console.error("Claude turn failed:", error instanceof Error ? error.message : error);
  } finally {
    cleanup(branch);
  }
}

console.log(failed === 0 ? "\nSmoke test OK." : `\n${failed} check(s) failed.`);
process.exit(failed === 0 ? 0 : 1);
