# GLUE MRPC

- Epochs: 20
- Learning Rate: 0.001
- Scheduler: StepLR
    - step_size: 6
    - gamma: 0.1
    - Update lr with new_lr if lr > 1e-5.
- Run Time
    - Device: NVIDIA GeForce RTX 3090
    - AdaGrad: 1,620 seconds
    - Adam: 1,900 seconds
    - DeltaAdaGrad: 1,449 seconds
