# Media experiments

- [01 — Conway's decoder of Life](01-conways-decoder-of-life/README.md): a generated
  H.264 bitstream makes software and hardware video decoders compute Conway's
  Game of Life using motion compensation, weighted prediction, and clipping.
- [02 — Maze flood-fill decoder solver](02-maze-flood-fill-decoder-solver/README.md):
  the decoder expands reachability through a maze; decoded arrival times provide
  shortest distances and paths while walls block propagation.
- [03 — Denoising decoder](03-denoising-decoder/README.md): a PyTorch-trained binary
  CNN compiles into H.264 prediction operations, including signed 3×3 filters,
  binary feature maps, and a streaming 1×1 output layer.
- [04 — Binary MNIST classification decoder](04-binary-mnist-classification-decoder/README.md):
  a trained binary network classifies handwritten 3s versus 8s using decoder
  pooling, signed arithmetic, and thresholds; 94.96% accuracy on the official
  two-class test subset.

The experiments use the shared [H.264 writer and decoder tools](decoder_core).
