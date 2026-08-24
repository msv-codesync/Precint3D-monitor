# Tool whitelist (Store A)

The language model does not send G-code. Machine writes are only:

- none
- pause
- set_parameter for hotend 180–260 °C, bed 40–90 °C, feedrate 50–120 %

Out of range is CLIP (refused), not a write.
IMU, microphone, and a spare thermocouple are not wired. Do not invent those readings.
A write still needs verify_proposal ALLOW after the same finding held across frames.
