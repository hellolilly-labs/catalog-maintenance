# /release

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

## Workflow Steps

1. **Pre-flight Checks**
   - Ensure on a feature branch
   - Check for uncommitted changes
   - Verify remote connectivity

2. **Git Cleanup**
   - Stage and commit any outstanding changes
   - Organize commits by category (feat, fix, docs, etc.)
   - Push all commits to remote

3. **Documentation**
   - Generate release notes from commits
   - Update CHANGELOG.md
   - Create GitHub issue documentation
   - Generate feature documentation if needed

4. **Version Management**
   - Bump version according to parameter
   - Update VERSION file
   - Update CHANGELOG.md with new version
   - Commit version changes

5. **Pull Request**
   - Create PR to target branch (default: dev)
   - Add comprehensive description
   - Link related issues
   - Add labels and reviewers if available

6. **GitHub Release** (after PR merge)
   - Create git tag
   - Generate GitHub release
   - Extract release notes from CHANGELOG
   - Create draft release for review

7. **Post-Release**
   - Provide merge instructions
   - Show next steps
   - Cleanup local branch (optional)

## Detailed Workflow

### 1. Initial State Assessment
```bash
# Check current branch
git branch --show-current

# Check for uncommitted changes
git status --porcelain

# Check remote status
git remote -v
```

### 2. Commit Organization
Group and commit changes by type:
- feat: New features
- fix: Bug fixes
- docs: Documentation
- chore: Maintenance tasks
- refactor: Code refactoring
- test: Test additions/changes

### 3. PR Creation
Create comprehensive PR with:
- Summary of changes
- Testing instructions
- Breaking changes (if any)
- Migration guide
- Related issues

### 4. Version Bump Logic
- **patch** (0.0.X): Bug fixes, small changes
- **minor** (0.X.0): New features, backwards compatible
- **major** (X.0.0): Breaking changes

### 5. Release Creation
After PR is merged:
- Create annotated git tag
- Generate GitHub release
- Include changelog entries
- Add any additional notes

## Interactive Process

The command will:
1. Show current state and ask for confirmation
2. Display what will be committed
3. Preview PR description before creating
4. Confirm version bump
5. Show final summary with next steps

## Error Handling

- Uncommitted changes: Prompt to commit or stash
- Conflicts: Provide resolution guidance
- Failed PR creation: Save description locally
- Network issues: Provide offline alternatives

## Configuration

Optional configuration via `.claude/release.config.json`:
```json
{
  "defaultBranch": "dev",
  "defaultBump": "patch",
  "autoMerge": false,
  "deleteBranchAfterMerge": false,
  "reviewers": ["team-lead"],
  "labels": ["release"]
}
```

## Safety Features

- Dry run mode available
- Confirmation prompts at each major step
- Rollback instructions if needed
- Local backup of all operations

## Example Full Flow

```
> /release minor

🚀 Starting release process...

📋 Current state:
- Branch: feature/price-search
- Version: 0.2.0 → 0.3.0 (minor bump)
- Target: dev
- Uncommitted: 5 files

✅ Committing changes...
✅ Pushing to remote...
✅ Creating PR #23...
✅ Version bumped to 0.3.0...

📝 Next steps:
1. Review PR: https://github.com/org/repo/pull/23
2. After merge: Run './scripts/create_release.sh'
3. Pull latest dev: 'git checkout dev && git pull'

Done! 🎉
```