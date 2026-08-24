# Pause policy (Store A)

Pause is reserved for held critical classes: spaghetti, first layer failed, layer shift, nozzle clog.
Warnings are logged. They may get a bounded parameter nudge if the operator enabled verified corrections.
One frame is never enough. Sticky hold (default 3 frames) is required.
Camera-only watches never write.
Resume is not issued from this copilot.
