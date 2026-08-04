"""Self-Healing AI Security Pipeline.

A modular proof-of-concept implementing a probe -> patch -> verify feedback
cycle for adversarial LLM vulnerability assessment. The pipeline uses
garak as the external probe layer, applies configurable prompt- and
output-level mitigations through a local OpenAI-compatible proxy, and
re-runs the probe to quantify mitigation effectiveness.
"""

__version__ = "0.1.0"
