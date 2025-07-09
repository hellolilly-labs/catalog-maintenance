# Release Notes - v0.2.1

## 🐛 Bug Fixes
- **ProductManager Deadlock Fix**: Resolved potential deadlock in ProductManager when `get_products()` is called while lock is held
  - Introduced `_load_internal()` method to handle loading without acquiring lock
  - Prevents concurrent access issues in voice assistant operations

## 📚 Documentation
- Added `/release` command documentation for automated release workflow

## 🔧 Technical Details
- Refactored ProductManager to use internal loading method for nested calls
- Improved thread safety in asynchronous product loading operations

## 📦 Dependencies
No dependency changes in this release.

---
Released: 2025-01-09