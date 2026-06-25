# Security Guidelines

## Mandatory Checks
- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All user inputs validated
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (sanitized output)
- [ ] Auth/authorization verified on all endpoints
- [ ] Error messages don't leak sensitive data

## Secret Management
- NEVER hardcode secrets
- ALWAYS use environment variables
- Validate required secrets at startup

## Python Security
- Use `os.environ["KEY"]` (not `.get()`) to fail fast on missing secrets
- Use **bandit** for static analysis: `bandit -r src/`

## Security Response
If security issue found: STOP, fix CRITICAL issues before continuing, rotate exposed secrets.
