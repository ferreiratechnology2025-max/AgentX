# Coding Style

## Core Principles
- **Immutability**: Always create new objects, never mutate existing ones
- **KISS**: Prefer simplest solution; avoid premature optimization
- **DRY**: Extract repeated logic; avoid copy-paste drift
- **YAGNI**: Don't build features before needed

## File Organization
- Many small files > few large files
- 200-400 lines typical, 800 max
- Organize by feature/domain

## Error Handling
- Handle errors explicitly at every level
- Never silently swallow errors
- Provide clear error messages

## Input Validation
- Validate all input at system boundaries
- Fail fast with clear messages
- Never trust external data

## Code Quality Checklist
- [ ] Code is readable and well-named
- [ ] Functions are small (<50 lines)
- [ ] No deep nesting (>4 levels)
- [ ] Proper error handling
- [ ] No hardcoded values (use constants/config)
- [ ] Immutable patterns used
