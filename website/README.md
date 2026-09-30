# BNN website

Independent static website for the estimator, with interactive statistical explanations,
theory, algorithms, examples, software, and full paper references.

Requires Node.js 22+. No npm dependencies or install step.

    node build.mjs
    node server.mjs

Local server: http://localhost:8747 (or set PORT).
Set SITE_URL when building, or edit src/site.json for the production canonical origin.

    node --test tests/*.test.mjs
    node scripts/check-site.mjs

Run mathematical checks and Python fixture generation on a compute host:

    PYTHONPATH=../src python tests/generate_oracle.py

The browser math module is independent of the DOM and is checked against binomial
identities, exhaustive subsets, endpoints, stable ties, and the Python estimators.
Source/result mapping: src/sources.json. The final appendix uses theorem numbers C.3–C.11.

Figures and timing records are sourced from the existing repository. The appendix is
the authors' final online appendix dated July 7, 2026; its source blob is recorded.
The site summarizes source statements and conditions; it does not present a new proof
or turn the illustration into a coverage study.

Railway: one Docker service serving static output with a health endpoint /healthz.
Use the service's assigned PORT. Build context is website/.
Production deployment and rollback details are recorded below when deployed.

Production: https://bnn-production-9654.up.railway.app/
Railway project: 3a92b9d3-1700-4496-b6bc-bfa45933df36
Service: 7e24c050-0ce5-4f99-a3aa-9978289fe14a
Environment: f0c39420-b3c8-4790-9856-6a55b5a05cc0

Deploy from this directory after the GitHub Check workflow passes. A manual verified
deployment keeps package-only commits from rebuilding the website. Use `railway up`
with these explicit project/environment/service IDs. For rollback, redeploy the
previous successful deployment through Railway. No secrets are needed by the site.

Service settings are persisted directly in Railway and documented in
`railway-service.json`. Review `railway environment config --json` before changing
settings. The legacy Railway Config File override is cleared. The static service
has no database, volume, or application secrets.
