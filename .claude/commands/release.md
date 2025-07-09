# release

Execute a complete release workflow including git cleanup, PR creation, version bumping, and GitHub release.

## Usage
```
/release [version_bump] [branch]
```

## Parameters
- `version_bump`: Type of version bump - `patch` (default), `minor`, or `major`
- `branch`: Target branch for PR - `dev` (default) or `main`

## Examples
```
/release                    # patch bump, PR to dev
/release minor             # minor version bump, PR to dev
/release major main        # major version bump, PR to main
```

---

# Release Workflow Automation

I'll help you execute a complete release workflow. Let me start by checking the current state of your repository.

```bash
# Check current branch and status
git branch --show-current
git status --porcelain | wc -l
```

Based on the parameters provided:
- Version bump type: ${1:-patch}
- Target branch: ${2:-dev}

I'll now:
1. Check for uncommitted changes and commit them if needed
2. Create a pull request to the target branch
3. Bump the version appropriately
4. Generate release notes
5. Create a GitHub release after merge

Let me begin the release process...