"""Global configuration for AgenticMemo."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    # Case bank
    max_cases: int = 10_000
    case_ttl_days: int | None = None          # None = no expiry
    # Failures (reward -0.2 by default) are stored as anti-cases so the
    # FailureMiner (CFM) can learn from them; raise this to discard them.
    min_reward_to_store: float = -1.0

    # Hierarchical memory
    enable_hierarchy: bool = True
    domain_auto_classify: bool = True         # auto-detect domain from task text

    # Graph memory
    enable_graph: bool = True
    max_edges_per_node: int = 20
    temporal_decay_rate: float = 0.01         # per-day staleness decay

    # Retrieval
    top_k: int = 4                            # cases returned per query
    min_similarity: float = 0.1              # discard below this score

    # Dedup: skip storing a case whose normalized task+answer fingerprint is
    # already in memory — repeated identical experiences add retrieval noise,
    # not knowledge (Phase 4.1).
    dedup_on_store: bool = True

    # Persistence. A path ending in ".db" selects the SQLite backend
    # (incremental O(1) writes); any other path uses atomic full-JSON.
    persist_path: str | None = None           # None = in-memory only


class RetrievalConfig(BaseModel):
    """Configuration for the ensemble retrieval system."""

    # Embedding backend: "sentence-transformers" | "openai" | "anthropic"
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Ensemble weights (must sum to ~1.0)
    weight_semantic: float = 0.5
    weight_bm25: float = 0.3
    weight_graph: float = 0.1
    weight_temporal: float = 0.1

    # Retrieval settings
    top_k: int = 4
    min_similarity: float = 0.1
    # Injection gates (small models are hurt by weak/verbose memory):
    # cases below this ensemble score are retrieved for bookkeeping but NOT
    # injected into prompts; injected blocks share a hard token budget.
    min_injection_score: float = 0.35
    injection_token_budget: int = 1500

    # Plan reuse (Phase 3.1): on a strong retrieval hit with a successful
    # stored plan, skip free-form planning and adapt the proven plan — saves
    # an LLM call and replaces the weakest small-model step with an artifact.
    enable_plan_reuse: bool = True
    plan_reuse_score: float = 0.75

    # GRPO policy
    enable_grpo: bool = True
    grpo_sample_size: int = 8                 # candidate sets to sample
    grpo_lr: float = 1e-3
    grpo_update_every: int = 10               # update policy every N cases stored


class LearningConfig(BaseModel):
    """Configuration for the learning system."""

    # Outcome verification (LLM-as-judge after each execution attempt).
    # Without it, any non-empty final answer counts as success.
    enable_verification: bool = True

    # Reflexion
    enable_reflexion: bool = True
    max_reflexion_retries: int = 2
    reflexion_failure_reward: float = -0.5

    # Trajectory filtering
    enable_quality_filter: bool = True
    min_trajectory_steps: int = 1
    max_trajectory_steps: int = 50
    variance_filter_threshold: float = 0.05  # discard if reward variance < this

    # Reward
    success_reward: float = 1.0
    partial_reward: float = 0.3
    failure_reward: float = -0.2
    # Step-efficiency shaping: each step subtracts this from a SUCCESS reward
    # (floored at half the base), so GRPO learns to prefer retrieving cases
    # that historically produced SHORT trajectories. 0.0 = off (default).
    step_penalty: float = 0.0


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    # LLM
    llm_provider: str = "anthropic"           # "anthropic" | "openai"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Execution
    max_steps: int = 20
    max_retries: int = 3
    timeout_seconds: float = 120.0
    # Stepwise exemplar guidance (Phase 3.2): each executor turn shows only
    # the NEXT action from the matched successful trajectory — a tiny
    # sequential scaffold against mid-trajectory drift in small models.
    enable_stepwise_guidance: bool = True

    # Sub-configs
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    # Logging
    verbose: bool = False
    log_trajectories: bool = True
