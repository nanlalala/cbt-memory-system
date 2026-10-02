# FHIR alignment notes

Memory Schema v1 is an application schema, not a certified electronic health-record format. Its field names are aligned where practical with HL7 FHIR R5 concepts so that later export or institutional review does not begin from an entirely project-specific model.

| Memory concept | Closest FHIR resource or element | Project use |
|---|---|---|
| Emotion or reported symptom | `Observation` | Time-stamped user-reported state and intensity |
| User goal | `Goal` | Description, lifecycle status and target date |
| CBT assignment | `Task` or `CarePlan.activity` | Assigned action, status and outcome link |
| Significant event | `Observation` or application event | User-reported event with effective time |
| Provenance | `Provenance` | Source session, turn IDs, supporting quote and update actor |
| Safety-relevant information | `Flag` / restricted `Observation` | Restricted context; safety routing remains outside ordinary retrieval |
| Correction and superseding | Resource version plus `Provenance` | New record points to the superseded memory; history remains auditable |
| User deletion | Application privacy operation | Content is hard-deleted; a content-free audit event remains |

Primary specification references:

- FHIR R5 index: <https://hl7.org/fhir/>
- Observation: <https://hl7.org/fhir/observation.html>
- Goal: <https://hl7.org/fhir/goal.html>
- Task: <https://hl7.org/fhir/task.html>
- Provenance: <https://hl7.org/fhir/provenance.html>

This alignment does not make the prototype a clinical record system. Before using real patient data, the schema, consent model, access control, retention policy and institutional approvals require specialist review.

