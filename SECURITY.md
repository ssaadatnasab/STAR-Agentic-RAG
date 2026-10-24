# Security notes

- Do not commit API keys, private endpoints, benchmark datasets, copyrighted corpus files, generated answers, or local paths containing personal information.
- The original uploaded script included a hardcoded API credential. It was intentionally omitted from this repository. Revoke/rotate that credential before publishing because it has been exposed in source text.
- The generated-code execution helper uses a subprocess, timeout, import checks, and simple blocked-call patterns. This is **not a secure sandbox** against hostile code. For any use with untrusted model output, run execution in a separately isolated container or VM with no credentials, restricted networking, read-only filesystems where possible, resource limits, and a non-privileged user.
- Review any retrieved web content and source material for usage rights before redistribution.
