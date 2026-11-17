# TangoPosture documentation

Updated 17 November 2026 against the local working tree.

- **Using the app:** [Beginner manual and metric examples](TANGO_POSTURE_GUIDE.md). Start with “Start here”; the couple tutorial covers all five result groups, signs, units, coverage and limitations.
- **Understanding the code:** [Architecture](ARCHITECTURE_DUAL_ENGINE_REENGINEERING.md), then [class diagrams](UML_CLASS_DIAGRAMS_FOR_DUMMIES.md).
- **Planning improvements:** [Technical debt audit and redesign proposal](TECHNICAL_DEBT_AND_REDESIGN.md). Findings distinguish observed code behavior from risks and proposed changes.

A metric is an observation, not a grade. “Unavailable” is not zero, “user confirmed” is not physical ground truth, and a display-only estimated joint should not be treated as independent measurement evidence.

The manual is also packaged as the desktop help document. Keep its core explanations self-contained. Detailed implementation proposals belong in the audit; they describe future work, not features already delivered.
