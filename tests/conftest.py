"""Forces the scripted (no-network, no-credentials) LLM backend for the
whole test suite, regardless of what's in the environment/.env, tests
must be hermetic and must not silently start making real API calls just
because a developer happens to have OPENAI_API_KEY set locally.
"""

import os

os.environ["JANUS_LLM_BACKEND"] = "scripted"
