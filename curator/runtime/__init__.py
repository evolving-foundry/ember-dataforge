"""DataForge Python Runtime — ZMQ-wrapped execution kernel.

Long-lived Python runtime (doc §17) that wraps the existing
``curator`` kernel behind a ZMQ DEALER socket and the
versioned ``dataforge-runtime`` protocol. The kernel itself is NOT modified
(doc §41.2 Phase 1: zero intrusion, new files only).
"""

__version__ = "0.1.0"

PROTOCOL = "dataforge-runtime"
PROTOCOL_VERSION = "1"
