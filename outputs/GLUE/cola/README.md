# GLUE Cola

- Epochs: 10
- Learning Rate: 0.001
- Scheduler: StepLR
    - step_size: 3
    - gamma: 0.1
    - Update lr with new_lr if lr >= 1e-5.
- Run Time
    - Device: NVIDIA GeForce RTX 3090
    - AdaGrad: 2,324 seconds
    - Adam: 1,861 seconds
    - DeltaAdaGrad: 1,868 seconds
