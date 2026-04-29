# LLMinus

### Disclaimer

**This is a read-only, archived fork of the `LLMinus` tool, created for reference purposes only.**

The tool was originally created and proposed by **Sasha Levin**. This repository is **not** actively maintained.

---

### What is LLMinus?

`LLMinus` is a tool designed to assist Linux kernel maintainers with merge conflict resolution using Large Language Models (LLMs). The core idea is to learn from the vast history of manual merge resolutions in the kernel's git history to help automate and inform future conflict resolutions.

The tool works by:
1.  Extracting historical cases where manual conflict resolution was required.
2.  Creating a searchable knowledge base of these past resolutions.
3.  When a new conflict arises, it finds semantically similar historical resolutions and uses them to construct a prompt for an LLM to suggest a resolution.

### Original Proposal (RFC)

The original concept, goals, and implementation details were introduced by Sasha Levin on the Linux Kernel Mailing List (LKML).

*   **[RFC 0/5] LLMinus: LLM-Assisted Merge Conflict Resolution:** [https://lkml.org/lkml/2025/12/19/1353](https://lkml.org/lkml/2025/12/19/1353)
