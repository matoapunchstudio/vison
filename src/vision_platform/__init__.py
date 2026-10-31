"""Machine-side composable computer-vision core.

Layers (see docs/2026-11-01-cv-platform-and-face-encoding-research.md):
frame plane (sources, detection, tracking, face) emits TrackObservations;
event plane (zones, activities, conditions, debounce) emits MachineEvents;
emitters persist payloads to an outbox and deliver them downstream.
"""

__version__ = "0.1.0"
