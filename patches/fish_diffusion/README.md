# Fish base compatibility patch

Applies only to upstream commit `8e8f8cd849717f2db561e6c194df1a320c910e62`. `files.json` records before/after SHA-256; setup refuses unknown source changes.

The supported route imports only ContentVec, Parselmouth, WaveNet and NSF-HiFiGAN registries. Optional separation/visualization dependencies and MMEngine optimizer imports are delayed. The patch explicitly loads the local NSF checkpoint and fixes single-nonzero-F0 indexing. It does not replace the diffusion loss or denoiser. Training orchestration is in the project adapter, not the upstream CLI.

Upstream code license remains MIT. Consult the original asset licenses for pretrained weights, including the NSF release license delivered with the assets.
