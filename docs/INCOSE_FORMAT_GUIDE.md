# INCOSE Requirements Template Integration Guide

## 1. INCOSE Requirements Template

The **INCOSE (International Council on Systems Engineering)** requirements template is a standardized format for writing requirements, widely used in safety-critical domains such as aerospace, automotive, and defense.

### Standard Format

```
[Who] shall [What] [How Well] under [Condition]
```

| Component | Description | Example |
|---|---|---|
| **[Who]** | Responsible system/subsystem/component | The braking system |
| **shall** | Obligation level (shall / should / may) | shall (mandatory) |
| **[What]** | Desired function or behavior (active voice) | detect obstacles |
| **[How Well]** | Quantifiable performance criterion | within 100ms with 99.9% accuracy |
| **[Condition]** | Operational scenario or constraint | when speed > 50 km/h |

### Example

```
The emergency braking system shall detect obstacles within 100ms
with 99.9% accuracy when the vehicle speed exceeds 30 km/h.
```

## 2. Motivation: KAOS-to-INCOSE Conversion

KAOS goal models capture hierarchical objectives and relationships, but industry practice often requires requirements in a standardized statement format. INCOSE formatting adds:

- **Clear accountability** — Each requirement names the responsible entity
- **Quantifiable criteria** — Performance targets are explicit and testable
- **Operational context** — Conditions under which the requirement applies are stated
- **Standards compliance** — Conforms to ISO/IEC/IEEE 29148:2018

### Comparison

| Aspect | KAOS Format | INCOSE Format |
|---|---|---|
| Form | Goal name + description | Structured sentence |
| Accountability | Implicit in `stakeholder` field | Explicit `[Who]` component |
| Measurability | In `measurable_criteria` field | Inline `[How Well]` component |
| Testability | Requires interpretation | Directly testable from statement |

## 3. Usage

### Single-Element Conversion

```python
from src.utils.incose_formatter import INCOSEFormatter

formatter = INCOSEFormatter()

kaos_element = {
    "name": "Prevent Hazardous Collisions",
    "description": "Ensure Active Safety functions respond within 100 ms",
    "quality_attribute": "Safety",
    "priority": 1,
    "stakeholder": "Safety System",
    "measurable_criteria": "Response time <= 100 ms"
}

incose_req = formatter.format_kaos_element(kaos_element)
print(incose_req.to_sentence())
# The safety system shall respond within 100 ms under all safety scenarios.
```

### Batch Conversion from Phase 3 Output

```python
import json

with open('phase3_integrated_kaos_model.json', 'r') as f:
    data = json.load(f)

incose_reqs = formatter.format_batch(data['gsn_elements'])
document = formatter.generate_requirements_document(
    data['gsn_elements'],
    title="System Requirements Specification"
)

with open('requirements_incose_format.md', 'w') as f:
    f.write(document)
```

## 4. Integration with Agent Prompts

To generate INCOSE-formatted requirements directly from the LLM, add format guidance to agent prompts:

```python
def conduct_initial_analysis(self, requirement: str) -> tuple[str, List[KAOSElement]]:
    user_input = f"""Analyze the following requirement from a SAFETY perspective.

Use INCOSE Requirements Template Format:
[Who] shall [What] [How Well] under [Condition]

Requirements:
- [Who]: Specify the responsible system/subsystem/component
- shall: Use 'shall' for mandatory requirements (priority 1-2)
- [What]: Use active voice verbs (detect, activate, maintain)
- [How Well]: Include quantifiable metrics (time, accuracy, rate)
- [Condition]: Specify operational scenario or constraint

Example:
"The emergency braking system shall activate within 100ms with
deceleration >= 6m/s^2 when an obstacle is detected within 30m."

Requirement: {requirement}

Structure each element's description field as an INCOSE requirement statement.
"""
```

## 5. Data Model Extension

INCOSE component fields can be added to `KAOSElement`:

```python
@dataclass
class KAOSElement:
    # ... existing fields ...

    # INCOSE template components
    incose_who: Optional[str] = None
    incose_shall: str = "shall"        # shall / should / may
    incose_what: Optional[str] = None
    incose_how_well: Optional[str] = None
    incose_condition: Optional[str] = None

    def to_incose_statement(self) -> str:
        """Generate INCOSE requirement statement."""
        parts = filter(None, [
            self.incose_who,
            self.incose_shall,
            self.incose_what,
            self.incose_how_well,
            self.incose_condition,
        ])
        sentence = " ".join(parts)
        return (sentence + ".") if sentence else self.description
```

## 6. Recommended Layering Strategy

| KAOS Hierarchy Level | INCOSE Formatting | Rationale |
|---|---|---|
| Level 1 (Strategic) | Optional | Abstract goals may not map cleanly |
| Level 2 (Tactical) | Recommended | Specific enough for structured statements |
| Level 3 (Operational) | Mandatory | Directly testable; maximum benefit |

## 7. Relationship to KAOS

INCOSE formatting is a **complement**, not a replacement for KAOS modeling:

- **KAOS** provides hierarchical goal decomposition, obstacle analysis, and relationship networks
- **INCOSE** provides standardized, testable requirement statements
- **Dual output** enables integration with both goal-modeling tools (Astah) and requirements management systems (DOORS)

## References

- INCOSE Systems Engineering Handbook, 5th Edition (2023)
- INCOSE Guide to Writing Requirements (2019)
- ISO/IEC/IEEE 29148:2018 — Systems and Software Engineering: Requirements Engineering
