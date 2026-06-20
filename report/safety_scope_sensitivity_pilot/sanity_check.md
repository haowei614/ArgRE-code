# Sanity Check Note

Variant A is run with the original QUARE SafetyAgent scope, AD case, seed 101, gpt-4o-mini-2024-07-18, temperature 0.7, and theta_eff 0.85.
Variants B/C patch only `QUARE_AGENT_SYSTEM_SCOPES['SafetyAgent']`; the prompt template and JSON schema are unchanged.

Existing formal mirror run record found: `/root/ArgRE-code/report/blind_requirement_quality_formal/runs/argre_af_preferred/argre_af_preferred-ad-s101/run_record.json`.
Existing run metadata: model=gpt-4o-mini, setting=negotiation_integration_verification, system=quare, seed=101.
The complete original artifact directory referenced by that formal mirror is not present in this checkout, so metric-level equality could not be checked against it.
