# Agent4Reqs: Multi-Agent Requirements Engineering Methodology

## 1. Overview

Agent4Reqs is a multi-agent requirements engineering platform designed for automated requirements negotiation and hierarchical KAOS (Keep All Objectives Satisfied) goal modeling. Five quality-specialized agents collaborate to analyze requirements from distinct quality perspectives, negotiate conflicts through structured protocols, and produce a unified, standards-compliant KAOS model.

## 2. Multi-Agent Architecture

### 2.1 Agent Roles

| Agent | Quality Concern | Focus Areas |
|---|---|---|
| **SafetyAgent** | Safety | Hazard analysis, fault detection, risk mitigation |
| **EfficiencyAgent** | Performance | Response time, throughput, resource optimization |
| **GreenAgent** | Sustainability | Energy efficiency, environmental impact, resource conservation |
| **TrustworthinessAgent** | Reliability | Data integrity, system availability, audit trails |
| **ResponsibilityAgent** | Compliance | Regulatory adherence, responsibility allocation, governance |

A **ManagerAgent** orchestrates all five agents, managing workflow phases, negotiation rounds, conflict resolution, and model integration.

### 2.2 Supporting Components

- **NegotiationManager** — Manages structured negotiation rounds between agents
- **PairwiseDialecticManager** — Implements pairwise testing and dialectic protocols
- **KAOSIntegrator** — Merges per-agent KAOS models into a unified hierarchy
- **ReportGenerator** — Produces JSON, GSN XML, and Markdown outputs
- **GSNXMLExporter** — Exports models in SACM ARM format for Astah System Safety

## 3. KAOS Modeling Framework

### 3.1 Element Types

| Type | Description |
|---|---|
| **Goal** | Desired objective to be achieved |
| **Softgoal** | Non-functional quality attribute requirement |
| **Task** | Concrete operational action |
| **Resource** | Required system or environmental resource |
| **Obstacle** | Risk or impediment to goal achievement |
| **Agent** | Responsible entity (human or system component) |

### 3.2 Relationship Types

| Relationship | Semantics |
|---|---|
| AND-refinement | All sub-goals must be satisfied |
| OR-refinement | At least one sub-goal must be satisfied |
| Contribution | Positive or negative influence on a softgoal |
| Dependency | Goal depends on another goal or resource |
| Conflict | Contradictory relationship between goals |
| Operationalization | Goal is realized through a specific task |

### 3.3 Three-Level Hierarchy

```
Level 1 — Strategic
├── System-level safety goals
├── Performance and efficiency objectives
└── Regulatory compliance mandates

Level 2 — Tactical
├── Specific safety requirements
├── Quantified performance metrics
└── Concrete compliance measures

Level 3 — Operational
├── Implementation tasks and procedures
├── Resource allocation plans
└── Monitoring and verification actions
```

## 4. Pipeline Phases

### Phase 1: Initial Hierarchical KAOS Generation

Each of the five agents independently analyzes the input requirements from its quality perspective and generates a three-level KAOS goal model. This phase runs in parallel for maximum coverage.

### Phase 2: Dialectic Negotiation

Agents engage in structured multi-round negotiation using four message types:

| Message | Direction | Purpose |
|---|---|---|
| `NegotiationInit` | Manager → Agents | Initiate negotiation round |
| `NegotiationForward` | Focus Agent → Reviewers | Share current KAOS elements |
| `NegotiationBackward` | Reviewers → Focus Agent | Provide feedback and suggestions |
| `NegotiationFinish` | Manager → All | Record consensus and close round |

The negotiation strategy combines:
- **Sequential focus rotation** — Each agent takes turns as the focus agent
- **Pairwise testing** — Combinatorial pairing for thorough cross-agent coverage
- **Dialectic protocol** — Thesis–antithesis–synthesis argumentation to resolve conflicts

### Phase 3: Model Integration and Topological Verification

The ManagerAgent merges all per-agent KAOS elements into a unified model, resolving remaining conflicts through priority-based rules (e.g., safety over efficiency). The integrated model is validated for:
- DAG topology (no cycles, no orphan nodes)
- Refinement completeness (all leaf goals have operationalizations)
- Cross-agent consistency

### Phase 4: Compliance Verification

RAG-augmented verification checks the integrated model against relevant industry standards (ISO 26262, ISO 27001, IEEE 29148, etc.). Detected violations are corrected automatically where possible, with manual review flags for ambiguous cases.

### Phase 5: Output Generation

The system produces:
- **Structured data** — JSON phase artifacts (`phase1_initial_models.json` through `phase4_verification_report.json`)
- **GSN XML** — SACM ARM format for import into Astah System Safety
- **Reports** — Markdown-formatted analysis reports with traceability information

## 5. Data Models

### KAOSElement

```python
@dataclass
class KAOSElement:
    id: str                        # Unique identifier
    name: str                      # Human-readable name
    description: str               # Detailed description
    element_type: ElementType      # Goal, Softgoal, Task, Resource, Obstacle, Agent
    quality_attribute: str         # Safety, Efficiency, Green, Trustworthiness, Responsibility
    priority: int                  # 1 (highest) to 5 (lowest)
    stakeholder: str               # Responsible entity
    measurable_criteria: str       # Quantifiable acceptance criteria
    hierarchy_level: int           # 1 = Strategic, 2 = Tactical, 3 = Operational
    parent_goal_id: Optional[str]  # Reference to parent in the goal tree
```

### KAOSRelation

```python
@dataclass
class KAOSRelation:
    id: str                        # Unique identifier
    source_id: str                 # Source element ID
    target_id: str                 # Target element ID
    relation_type: RelationType    # AND, OR, Contribution, Dependency, Conflict, Operationalization
    description: str               # Relationship description
    strength: Optional[str]        # Weak, Medium, Strong (for contributions)
```

## 6. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM Client | litellm (routing to OpenAI) |
| LLM Model | gpt-4o-mini (configurable) |
| Package Management | uv, pyproject.toml (hatchling) |
| Data Formats | JSON, GSN XML (SACM ARM), Markdown |
| Dev Tools | ruff, pytest |

## 7. Application Domains

| Domain | Key Quality Concerns |
|---|---|
| **Autonomous Driving** | ISO 26262 functional safety, real-time performance, environmental sustainability |
| **Medical Devices** | FDA/IEC 62304 compliance, patient safety, data privacy (HIPAA) |
| **Financial Systems** | Transaction security, anti-money laundering, audit trails |
| **Aerospace** | DO-178C compliance, fault tolerance, mission reliability |
