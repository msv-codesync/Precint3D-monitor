# FDM defect taxonomy (Store A)

Use only as context. Do not invent a different class than the detector.

## spaghetti_failure / spaghetti
Filament printed in air, not attached to the part. Critical. Pause. Do not raise temperature.

## first_layer_failure
Bead did not stick to the bed. Critical. Pause. Bed/nozzle changes will not recover a missing first layer.

## layer_shift
Layers offset in X/Y. Mechanical skip. Critical. Pause. Do not treat as under-extrusion.

## nozzle_clog
Flow stopped or severely reduced with a blocked nozzle. Critical. Pause and clear. Do not keep heating blindly.

## warping
Corner or edge lifted. Warning. Slow feedrate toward 70%. Raise bed target by about 5 °C inside 40–90 °C. Not a pause class.

## under_extrusion
Thin walls, gaps, pale extrusion. Warning. Slow toward 80%. Raise hotend target by about 5 °C inside 180–260 °C.

## stringing
Hairs between features. Warning. Drop hotend about 3 °C inside bounds. Retraction is not a live write.

## blob_or_zit / zits_blobs
Nubs on the wall. Warning. Slow toward 85%. Watch the nozzle.

## good_print
Looks fine. No write.
