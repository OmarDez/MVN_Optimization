# Benchmark results

_162 records; hosts: INTEL(R) XEON(R) PLATINUM 8573C, Neoverse-N2_

## Crossover: neon vs ONNX against layer size

| arch | model | layer MACs | total MACs | b | batch | input | onnx ms | neon ms | neon/onnx |
|---|---|---|---|---|---|---|---|---|---|
| aarch64 | head_512x10 | 0.005M | 0.000G | 3 | 64 | complex | 0.19 | 1.00 | **0.193** |
| aarch64 | head_512x10 | 0.005M | 0.000G | 4 | 64 | complex | 0.19 | 0.98 | **0.195** |
| aarch64 | head_512x10 | 0.005M | 0.003G | 3 | 512 | complex | 1.21 | 7.39 | **0.164** |
| aarch64 | head_512x10 | 0.005M | 0.003G | 4 | 512 | complex | 1.20 | 7.39 | **0.163** |
| aarch64 | head_512x10 | 0.005M | 0.011G | 3 | 2048 | complex | 5.16 | 29.29 | **0.176** |
| aarch64 | head_512x10 | 0.005M | 0.011G | 4 | 2048 | complex | 4.75 | 29.21 | **0.163** |
| aarch64 | layer_2048x64 | 0.131M | 0.008G | 3 | 64 | complex | 1.65 | 4.60 | **0.358** |
| aarch64 | layer_2048x64 | 0.131M | 0.008G | 4 | 64 | complex | 1.65 | 4.57 | **0.360** |
| aarch64 | layer_2048x64 | 0.131M | 0.067G | 3 | 512 | complex | 12.98 | 36.83 | **0.352** |
| aarch64 | layer_2048x64 | 0.131M | 0.067G | 4 | 512 | complex | 12.90 | 36.34 | **0.355** |
| aarch64 | layer_2048x64 | 0.131M | 0.269G | 3 | 2048 | complex | 55.11 | 155.64 | **0.354** |
| aarch64 | layer_2048x64 | 0.131M | 0.269G | 4 | 2048 | complex | 61.99 | 156.01 | **0.397** |
| aarch64 | layer_4096x256 | 1.049M | 0.067G | 3 | 64 | complex | 12.05 | 16.14 | **0.746** |
| aarch64 | layer_4096x256 | 1.049M | 0.067G | 4 | 64 | complex | 11.71 | 16.09 | **0.728** |
| aarch64 | layer_4096x256 | 1.049M | 0.537G | 3 | 512 | complex | 90.58 | 127.99 | **0.708** |
| aarch64 | layer_4096x256 | 1.049M | 0.537G | 4 | 512 | complex | 90.46 | 131.29 | **0.689** |
| aarch64 | layer_4096x256 | 1.049M | 2.148G | 3 | 2048 | complex | 374.81 | 527.45 | **0.711** |
| aarch64 | layer_4096x256 | 1.049M | 2.148G | 4 | 2048 | complex | 373.08 | 524.11 | **0.712** |

## Latency

| arch | model | b | batch | angular_tiled | complex128 | complex64 | neon | onnx | best angular vs onnx |
|---|---|---|---|---|---|---|---|---|---|
| aarch64 | head_512x10 | 3 | 64 | 2.55 | 0.08 | 0.07 | 1.00 | 0.19 | 0.193x |
| aarch64 | head_512x10 | 3 | 512 | 20.01 | 0.56 | 0.45 | 7.39 | 1.21 | 0.164x |
| aarch64 | head_512x10 | 3 | 2048 | 79.60 | 2.17 | 1.91 | 29.29 | 5.16 | 0.176x |
| aarch64 | head_512x10 | 4 | 64 | 2.52 | 0.08 | 0.07 | 0.98 | 0.19 | 0.195x |
| aarch64 | head_512x10 | 4 | 512 | 19.86 | 0.56 | 0.45 | 7.39 | 1.20 | 0.163x |
| aarch64 | head_512x10 | 4 | 2048 | 78.74 | 2.16 | 1.79 | 29.21 | 4.75 | 0.163x |
| aarch64 | layer_2048x64 | 3 | 64 | 44.23 | 1.25 | 0.72 | 4.60 | 1.65 | 0.358x |
| aarch64 | layer_2048x64 | 3 | 512 | 353.26 | 8.62 | 5.01 | 36.83 | 12.98 | 0.352x |
| aarch64 | layer_2048x64 | 3 | 2048 | 1420.08 | 36.35 | 21.45 | 155.64 | 55.11 | 0.354x |
| aarch64 | layer_2048x64 | 4 | 64 | 43.94 | 1.16 | 0.70 | 4.57 | 1.65 | 0.360x |
| aarch64 | layer_2048x64 | 4 | 512 | 351.22 | 8.61 | 4.95 | 36.34 | 12.90 | 0.355x |
| aarch64 | layer_2048x64 | 4 | 2048 | 1415.94 | 37.27 | 24.02 | 156.01 | 61.99 | 0.397x |
| aarch64 | layer_4096x256 | 3 | 64 | 333.81 | 8.84 | 5.00 | 16.14 | 12.05 | 0.746x |
| aarch64 | layer_4096x256 | 3 | 512 | 2649.78 | 59.62 | 32.07 | 127.99 | 90.58 | 0.708x |
| aarch64 | layer_4096x256 | 3 | 2048 | 10921.00 | 237.56 | 127.88 | 527.45 | 374.81 | 0.711x |
| aarch64 | layer_4096x256 | 4 | 64 | 340.83 | 8.65 | 4.80 | 16.09 | 11.71 | 0.728x |
| aarch64 | layer_4096x256 | 4 | 512 | 2679.54 | 59.74 | 31.90 | 131.29 | 90.46 | 0.689x |
| aarch64 | layer_4096x256 | 4 | 2048 | 10750.18 | 237.18 | 128.52 | 524.11 | 373.08 | 0.712x |
| x86_64 | head_512x10 | 3 | 64 | 2.13 | 0.06 | 0.09 | - | 0.09 | 0.040x |
| x86_64 | head_512x10 | 3 | 512 | 16.98 | 0.52 | 0.61 | - | 0.93 | 0.055x |
| x86_64 | head_512x10 | 3 | 2048 | 69.23 | 2.08 | 2.00 | - | 3.64 | 0.053x |
| x86_64 | head_512x10 | 4 | 64 | 2.35 | 0.07 | 0.07 | - | 0.11 | 0.045x |
| x86_64 | head_512x10 | 4 | 512 | 17.60 | 0.57 | 0.60 | - | 0.95 | 0.054x |
| x86_64 | head_512x10 | 4 | 2048 | 69.48 | 2.07 | 2.00 | - | 3.69 | 0.053x |
| x86_64 | layer_2048x64 | 3 | 64 | 43.97 | 0.85 | 0.61 | - | 0.85 | 0.019x |
| x86_64 | layer_2048x64 | 3 | 512 | 358.55 | 4.23 | 3.46 | - | 5.25 | 0.015x |
| x86_64 | layer_2048x64 | 3 | 2048 | 1438.23 | 20.06 | 17.06 | - | 28.35 | 0.020x |
| x86_64 | layer_2048x64 | 4 | 64 | 43.09 | 0.86 | 0.58 | - | 0.81 | 0.019x |
| x86_64 | layer_2048x64 | 4 | 512 | 350.13 | 4.26 | 3.32 | - | 5.17 | 0.015x |
| x86_64 | layer_2048x64 | 4 | 2048 | 1441.00 | 20.62 | 18.03 | - | 26.69 | 0.019x |
| x86_64 | layer_4096x256 | 3 | 64 | 343.00 | 4.58 | 2.77 | - | 3.33 | 0.010x |
| x86_64 | layer_4096x256 | 3 | 512 | 2699.91 | 27.01 | 14.73 | - | 28.62 | 0.011x |
| x86_64 | layer_4096x256 | 3 | 2048 | 10867.90 | 117.59 | 73.74 | - | 146.90 | 0.014x |
| x86_64 | layer_4096x256 | 4 | 64 | 346.19 | 4.59 | 2.77 | - | 3.27 | 0.009x |
| x86_64 | layer_4096x256 | 4 | 512 | 2709.24 | 25.70 | 14.79 | - | 26.23 | 0.010x |
| x86_64 | layer_4096x256 | 4 | 2048 | 10894.39 | 123.27 | 70.85 | - | 148.52 | 0.014x |

## Weight memory

| model | weights | complex128 | fp32 re/im | uint8 idx | packed b-bit | vs fp32 |
|---|---|---|---|---|---|---|
| head_512x10 (b=3) | 5,130 | 80 KiB | 40 KiB | 5 KiB | 2 KiB | **21x** |
| head_512x10 (b=4) | 5,130 | 80 KiB | 40 KiB | 5 KiB | 3 KiB | **16x** |
| layer_2048x64 (b=3) | 131,136 | 2049 KiB | 1024 KiB | 128 KiB | 48 KiB | **21x** |
| layer_2048x64 (b=4) | 131,136 | 2049 KiB | 1024 KiB | 128 KiB | 64 KiB | **16x** |
| layer_4096x256 (b=3) | 1,048,832 | 16388 KiB | 8194 KiB | 1024 KiB | 384 KiB | **21x** |
| layer_4096x256 (b=4) | 1,048,832 | 16388 KiB | 8194 KiB | 1024 KiB | 512 KiB | **16x** |

## Accuracy

| model | b | backend | agreement vs fp | multiplier-free |
|---|---|---|---|---|
| head_512x10 | 3 | angular_tiled | 0.7285 | True |
| head_512x10 | 3 | angular_tiled | 0.7285 | True |
| head_512x10 | 3 | complex128 | 0.8203 | False |
| head_512x10 | 3 | complex128 | 0.8203 | False |
| head_512x10 | 3 | complex64 | 0.8203 | False |
| head_512x10 | 3 | complex64 | 0.8203 | False |
| head_512x10 | 3 | neon | 0.7285 | True |
| head_512x10 | 3 | onnx | 0.8203 | False |
| head_512x10 | 3 | onnx | 0.8203 | False |
| head_512x10 | 4 | angular_tiled | 0.8730 | True |
| head_512x10 | 4 | angular_tiled | 0.8730 | True |
| head_512x10 | 4 | complex128 | 0.9023 | False |
| head_512x10 | 4 | complex128 | 0.9023 | False |
| head_512x10 | 4 | complex64 | 0.9023 | False |
| head_512x10 | 4 | complex64 | 0.9023 | False |
| head_512x10 | 4 | neon | 0.8730 | True |
| head_512x10 | 4 | onnx | 0.9023 | False |
| head_512x10 | 4 | onnx | 0.9023 | False |
| layer_2048x64 | 3 | angular_tiled | 0.6343 | True |
| layer_2048x64 | 3 | angular_tiled | 0.6343 | True |
| layer_2048x64 | 3 | complex128 | 0.7261 | False |
| layer_2048x64 | 3 | complex128 | 0.7261 | False |
| layer_2048x64 | 3 | complex64 | 0.7261 | False |
| layer_2048x64 | 3 | complex64 | 0.7261 | False |
| layer_2048x64 | 3 | neon | 0.6343 | True |
| layer_2048x64 | 3 | onnx | 0.7261 | False |
| layer_2048x64 | 3 | onnx | 0.7261 | False |
| layer_2048x64 | 4 | angular_tiled | 0.7959 | True |
| layer_2048x64 | 4 | angular_tiled | 0.7959 | True |
| layer_2048x64 | 4 | complex128 | 0.8525 | False |
| layer_2048x64 | 4 | complex128 | 0.8525 | False |
| layer_2048x64 | 4 | complex64 | 0.8525 | False |
| layer_2048x64 | 4 | complex64 | 0.8525 | False |
| layer_2048x64 | 4 | neon | 0.7959 | True |
| layer_2048x64 | 4 | onnx | 0.8525 | False |
| layer_2048x64 | 4 | onnx | 0.8525 | False |
| layer_4096x256 | 3 | angular_tiled | 0.5737 | True |
| layer_4096x256 | 3 | angular_tiled | 0.5737 | True |
| layer_4096x256 | 3 | complex128 | 0.6890 | False |
| layer_4096x256 | 3 | complex128 | 0.6890 | False |
| layer_4096x256 | 3 | complex64 | 0.6890 | False |
| layer_4096x256 | 3 | complex64 | 0.6890 | False |
| layer_4096x256 | 3 | neon | 0.5737 | True |
| layer_4096x256 | 3 | onnx | 0.6890 | False |
| layer_4096x256 | 3 | onnx | 0.6890 | False |
| layer_4096x256 | 4 | angular_tiled | 0.7651 | True |
| layer_4096x256 | 4 | angular_tiled | 0.7651 | True |
| layer_4096x256 | 4 | complex128 | 0.8291 | False |
| layer_4096x256 | 4 | complex128 | 0.8291 | False |
| layer_4096x256 | 4 | complex64 | 0.8291 | False |
| layer_4096x256 | 4 | complex64 | 0.8291 | False |
| layer_4096x256 | 4 | neon | 0.7651 | True |
| layer_4096x256 | 4 | onnx | 0.8291 | False |
| layer_4096x256 | 4 | onnx | 0.8291 | False |
