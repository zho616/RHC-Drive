# RHC-Drive documentation

Start with the [project overview](../README.md). For the current matched roundabout experiments, use the first guide below.

## Reproduction and deployment

- [Matched 500K roundabout training, Ours, five ablations, and simulator runtime](cloud_roundabout_training.md)
- [Independent server/GPU training for T-intersection, roundabout, and nuPlan](parallel_server_training.md)
- [Server setup and earlier T-intersection reproduction](server_training.md)
- [MetaDrive environment basics](metadrive.md)
- [Policy visualization and rendering](metadrive_render.md)
- [Troubleshooting](troubleshooting.md)

## Implementation background

- [History context and world-model integration](context_dreamer.md)
- [Chassis randomization](chassis_randomization.md)
- [Tensor shapes](tensor_shapes.md)

## Historical experiments

The other `context_*` and dated queue documents describe exploratory configurations, diagnostics, and earlier experiment schedules. They provide provenance and may differ from the current matched recipe. The [experiment archive](../experiments/README.md) preserves earlier records; it is not the final paper results table.

The inherited [Docker guide](docker.md) and [IsaacLab guide](isaaclab_env.md) describe optional backbone infrastructure rather than the matched driving reproduction environment.
