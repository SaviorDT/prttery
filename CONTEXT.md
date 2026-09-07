# Background-Removal Segmentation

This context defines the terms used to train, evaluate, and export foreground
segmentation models for the project's video frames.

## Language

**Run Parameters**:
A typed, reusable definition of one complete execution scenario, including its
data, model, training, and output settings.
_Avoid_: CLI arguments, flags

**Run Parameter Path**:
A filesystem location used by a run. Every relative value is interpreted from
the execution's current working directory.
_Avoid_: data-relative path, loader-relative path

**Dataset Directory Pattern**:
A Run Parameter Path selecting one or more dataset directories. Pattern syntax
is expanded before the run starts, and a pattern that matches nothing is invalid.
_Avoid_: directory-name string, CLI glob

**Mask Format**:
The representation of a ground-truth mask and the semantic meaning of a
model's output: either binary foreground probability or CVAT six-class indices.
_Avoid_: output type, label encoding


**Annotation Pattern**:
A Run Parameter Path selecting one or more CVAT annotation files. Pattern syntax
is expanded before the run starts, and a pattern that matches nothing is invalid.
_Avoid_: annotation file path

**External Dataset Directory**:
A source-frame directory outside the repository's ``./data`` root, resolved
as a normalized filesystem path before frame matching.
_Avoid_: data-relative directory

**Binary Loss**:
A loss valid only for a single-channel foreground-probability model: BCE or
the equal-weight BCE-Dice combination.
_Avoid_: multi-class loss

**Monitoring Criterion**:
The validation value that consistently controls checkpoint selection, learning
rate decay, encoder unfreezing, and early stopping.
_Avoid_: early-stop check, best-loss check

**Boundary F1**:
The foreground-contour F1 score computed at native image resolution, with a
prediction boundary considered correct when it lies within the configured pixel
tolerance of the reference boundary.
_Avoid_: IoU, Dice
