# VeriLogic Security Launch Checklist

VeriLogic is currently a local Streamlit application with no user database, login endpoint,
API routes, or external database. Controls that require those surfaces are recorded here so
that a future deployment does not accidentally treat the demo as production-ready.

The legacy wellness dashboard's GitHub integration is optional. Configure `GITHUB_TOKEN` through Streamlit
secrets for authenticated API access; never place a token in the repository, UI, query string,
or logs. Demo mode uses synthetic activity and does not contact GitHub.

## Implemented in this repository

1. **Hide API keys**: `.env`, virtual environments, model weights, and local audit output are ignored.
2. **Purge Git secrets**: CI runs Gitleaks against the repository history presented to the job.
5. **Encrypt sensitive data**: uploaded images are processed in temporary files; audit records store minimized evidence, not OCR text or EXIF values. Use encrypted storage and managed keys for production.
13. **Parameterize queries**: no database queries exist.
14. **Validate all input**: uploaded files require an allowed image suffix, valid image bytes, and a 10 MB limit.
15. **Escape user content**: OCR and filenames are rendered through Streamlit widgets, not interpolated into HTML.
16. **Restrict file uploads**: Streamlit's global limit is configured and the application enforces a 10 MB image limit.
17. **Trim API responses**: no public API exists; audit output is minimized.
18. **Security headers**: XSRF protection is enabled in Streamlit. Add HSTS, CSP, frame, MIME, and referrer headers at the TLS reverse proxy.
20. **Scan dependencies**: CI runs `pip-audit` against the locked environment.

## Backend or hosting controls still required

3. **Public DB key**: only expose an anonymous client key; never expose service-role credentials.
4. **Row-level security**: enable RLS and test every tenant/user policy before adding a database.
6. **Server-side auth**: add authentication and authorization in a backend before handling accounts.
7. **Lock record access**: scope document and audit reads by authenticated user or tenant.
8. **Block field tampering**: derive risk and audit hashes server-side; never trust client-submitted scores.
9. **Secure session cookies**: use Secure, HttpOnly, SameSite cookies from the production auth layer.
10. **Hash passwords**: use Argon2id or a managed identity provider; never store plaintext passwords.
11. **Rate limit login**: apply account/IP limits and exponential backoff at the auth gateway.
12. **Bot protection**: add CAPTCHA or an equivalent risk-based challenge to public login/upload endpoints.
19. **Force HTTPS**: terminate TLS and redirect HTTP to HTTPS at the reverse proxy or hosting platform.

## Local checks

```powershell
uv run --with pip-audit pip-audit
uv run --with pytest pytest -q
uv run streamlit run app.py
```

Do not commit `.env`, credentials, uploaded documents, `.verilogic/audit.jsonl`, or model weights.
