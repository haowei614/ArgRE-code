# Orchestrator Agent (ManagerAgent)

## Overview

The Orchestrator Agent (ManagerAgent) is the central coordination component of the QUARE system. It manages the end-to-end hierarchical KAOS goal modeling workflow without directly performing requirements analysis. Instead, it orchestrates the five specialized quality agents — Safety, Efficiency, Green, Trustworthiness, and Responsibility — ensuring they collaborate effectively to produce a unified, high-quality KAOS model.

## Core Roles

| Role | Description |
|---|---|
| **Workflow Orchestration** | Coordinates the modeling process through predefined phases (0–5), ensuring sequential execution and expected outputs |
| **Multi-Agent Coordination** | Manages communication, negotiation, and collaboration among the five specialized agents |
| **Conflict Resolution** | Resolves inter-agent conflicts using prioritization rules (e.g., Safety over Efficiency) |
| **Model Integration** | Merges per-agent KAOS elements into a unified model with valid topology (DAG) |
| **Quality Assurance** | Ensures the final model meets quality standards through validation, compliance checking, and fact-checking |

## Responsibilities

### 1. Requirements Reception and Distribution
- Receive initial requirements descriptions
- Distribute requirements to all specialized agents for parallel analysis
- Manage external specification document processing

### 2. Negotiation Process Management
- Initiate and coordinate multi-round dialectic negotiation
- Manage message flow: `NegotiationInit` → `NegotiationForward` → `NegotiationBackward` → `NegotiationFinish`
- Dynamically adjust negotiation rounds based on quality improvement trajectory
- Record complete negotiation history for traceability

### 3. Conflict Detection and Resolution

Resolution follows a strict priority hierarchy:

| Priority | Strategy | Description |
|---|---|---|
| 1 | Hard constraint validation | DeterministicValidator checks |
| 2 | Safety-first principle | Safety > Efficiency precedence |
| 3 | Traceability stamping | Update `conflict_resolved_by` field |

### 4. Model Integration and Validation
- Merge all per-agent KAOS elements
- Validate model topology: detect orphan nodes, cycles, and invalid leaf nodes
- Automatically repair simple topological issues
- Ensure the final model forms a valid DAG

### 5. Quality Verification
- Universal consistency verification
- Fact-checking to detect LLM hallucinations
- Deterministic rule validation
- Regulatory compliance verification (ISO 26262, ISO 27001, IEEE 29148)
- Automated violation correction

### 6. Output Generation
- Coordinate downstream artifact generation (requirements documents, test cases)
- Manage file organization and storage
- Generate complete session logs and traceability reports

## Relationship with Specialized Agents

The Orchestrator and specialized agents have a **coordinator–participant** relationship:

**Orchestrator Agent:**
- Does not perform requirements analysis directly
- Orchestrates workflow, allocates tasks, and resolves conflicts
- Receives per-agent outputs and performs integration
- Makes final decisions on unresolved conflicts

**Specialized Agents** (Safety, Efficiency, Green, Trustworthiness, Responsibility):
- Analyze requirements from their quality perspective
- Exchange views through structured negotiation protocols
- Submit final outputs to the Orchestrator for integration
- Accept the Orchestrator's coordination and conflict resolution decisions

## Workflow Phases

| Phase | Name | Description |
|---|---|---|
| 0 | External Specification Processing | Parse and extract rules from external standards (optional) |
| 1 | Initial KAOS Generation | Five agents independently generate three-level KAOS models |
| 2 | Dialectic Negotiation | Multi-round structured negotiation between agents |
| 2.5 | Conflict Resolution | Chief Architect logic applies prioritization rules |
| 2.6 | Pairwise Dialectic Negotiation | Combinatorial pairing for thorough cross-agent coverage |
| 3 | Model Integration | Merge elements, validate topology, generate relationships |
| 4 | Verification | Consistency checking, fact-checking, deterministic validation |
| 4.5 | Violation Correction | Automated compliance fixes and re-verification |
| 5 | Output Generation | Generate requirements documents, test cases, reports |

## Technical Characteristics

- **Modular architecture**: Independent components for negotiation, logging, integration, and verification
- **Dynamic negotiation**: Adjusts round count based on quality improvement trajectory
- **Multi-level verification**: Fact-checking, deterministic rules, and topological validation
- **Complete traceability**: All decisions and agent interactions are logged for audit
