"""
Converts pixel measurements to real-world units (cm). We assume the trough is
100cm wide (TROUGH_WIDTH_PX pixels) and use that as the reference for the
conversion.

When measuring pumpkins, we assume they're on the trough for this conversion
to make sense.
"""

TROUGH_WIDTH_CM = 100.0
TROUGH_WIDTH_PX = 165
SCALE_CM_PER_PX = TROUGH_WIDTH_CM / TROUGH_WIDTH_PX
