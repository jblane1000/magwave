# Security policy

MagWave is an offline desktop application. It does not open sockets at runtime (NFR-6) and does not
process untrusted case files with elevated privileges — but it does parse CAD and case files, so the usual
parser risks apply.

## Reporting

Email jblane1000@gmail.com with steps to reproduce. Please do not open a public issue for a security
problem. Acknowledgement within 7 days; a fix or a stated position within 30.

## In scope

- Case-file (JSON) parser: crashes, resource exhaustion, path traversal via CAD references.
- CAD import path: malformed STEP/IGES/STL causing crashes or writes outside the case directory.
- Any accidental network access.

## Out of scope

- Third-party engine binaries (gmsh, Elmer) — report upstream as well.
- Results being physically wrong: that is a validation issue, not a security one. See
  [docs/CHECKLISTS/validation.md](docs/CHECKLISTS/validation.md).

## Supported versions

Pre-release (0.x): only the latest tagged version.
