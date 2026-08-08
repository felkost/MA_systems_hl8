# MA_systems_hl8

A multi-agent research system: a Supervisor coordinates three specialised
sub-agents — Planner, Researcher, Critic — in a Plan → Research → Critique
loop, with human approval gating the final report write. Built with
LangChain, LangGraph and LangSmith.

## Status

Under construction. This README is a stub — the full version, with
installation steps, an architecture diagram and a usage transcript, lands
when the project closes out.

- [x] Stage 0 — repository, CI, agent documentation contract
- [x] Stage 1 — RAG foundation (Chroma, hybrid retrieval, tools)
- [x] Stage 2 — Planner agent
- [x] Stage 3 — Researcher agent
- [x] Stage 4 — Critic agent
- [x] Stage 5 — Supervisor (agent-as-tool coordination)
- [x] Stage 6 — human-in-the-loop approval on report writes
- [x] Stage 7 — explicit `StateGraph` orchestration path
- [x] Stage 8 — telemetry and reports
- [ ] Stage 9 — LangSmith evaluation
- [ ] Stage 10 — knowledge graph (Neo4j)
- [ ] Stage 11 — experiments
- [ ] Stage 12 — documentation and final report

## License

[MIT](LICENSE)
