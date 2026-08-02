# Benchmark results

_182 records; hosts: AMD EPYC 7763 64-Core Processor, Neoverse-N2_

## Crossover: neon vs ONNX against layer size

| arch | model | layer MACs | total MACs | b | batch | input | onnx ms | neon ms | neon/onnx |
|---|---|---|---|---|---|---|---|---|---|
| aarch64 | head_512x10 | 0.005M | 0.000G | 3 | 64 | complex | 0.20 | 0.99 | **0.205** |
| aarch64 | head_512x10 | 0.005M | 0.000G | 4 | 64 | complex | 0.21 | 0.98 | **0.208** |
| aarch64 | head_512x10 | 0.005M | 0.003G | 3 | 512 | complex | 1.21 | 7.16 | **0.169** |
| aarch64 | head_512x10 | 0.005M | 0.003G | 4 | 512 | complex | 1.22 | 7.14 | **0.170** |
| aarch64 | head_512x10 | 0.005M | 0.011G | 3 | 2048 | complex | 4.53 | 28.97 | **0.156** |
| aarch64 | head_512x10 | 0.005M | 0.011G | 4 | 2048 | complex | 4.97 | 29.26 | **0.170** |
| aarch64 | layer_2048x64 | 0.131M | 0.008G | 3 | 64 | complex | 1.65 | 4.62 | **0.358** |
| aarch64 | layer_2048x64 | 0.131M | 0.008G | 4 | 64 | complex | 1.67 | 4.57 | **0.365** |
| aarch64 | layer_2048x64 | 0.131M | 0.067G | 3 | 512 | complex | 12.57 | 36.56 | **0.344** |
| aarch64 | layer_2048x64 | 0.131M | 0.067G | 4 | 512 | complex | 12.61 | 36.73 | **0.343** |
| aarch64 | layer_2048x64 | 0.131M | 0.269G | 3 | 2048 | complex | 56.10 | 154.66 | **0.363** |
| aarch64 | layer_2048x64 | 0.131M | 0.269G | 4 | 2048 | complex | 58.48 | 155.14 | **0.377** |
| aarch64 | layer_4096x256 | 1.049M | 0.067G | 3 | 64 | complex | 11.11 | 15.85 | **0.701** |
| aarch64 | layer_4096x256 | 1.049M | 0.067G | 4 | 64 | complex | 11.09 | 15.88 | **0.698** |
| aarch64 | layer_4096x256 | 1.049M | 0.537G | 3 | 512 | complex | 89.37 | 129.84 | **0.688** |
| aarch64 | layer_4096x256 | 1.049M | 0.537G | 4 | 512 | complex | 88.50 | 127.06 | **0.697** |
| aarch64 | layer_4096x256 | 1.049M | 2.148G | 3 | 2048 | complex | 369.63 | 522.44 | **0.708** |
| aarch64 | layer_4096x256 | 1.049M | 2.148G | 4 | 2048 | complex | 375.20 | 527.07 | **0.712** |
| aarch64 | layer_6144x384 | 2.360M | 4.833G | 4 | 2048 | complex | 803.53 | 1000.15 | **0.803** |
| aarch64 | layer_8192x512 | 4.195M | 8.591G | 4 | 2048 | complex | 1443.36 | 1620.71 | **0.891** |
| x86_64 | head_512x10 | 0.005M | 0.011G | 4 | 2048 | complex | 3.58 | 55.33 | **0.065** |
| x86_64 | layer_2048x64 | 0.131M | 0.269G | 4 | 2048 | complex | 41.19 | 566.05 | **0.073** |
| x86_64 | layer_4096x256 | 1.049M | 2.148G | 4 | 2048 | complex | 222.06 | 3471.62 | **0.064** |
| x86_64 | layer_6144x384 | 2.360M | 4.833G | 4 | 2048 | complex | 460.95 | 7554.73 | **0.061** |
| x86_64 | layer_8192x512 | 4.195M | 8.591G | 4 | 2048 | complex | 810.01 | 13172.56 | **0.061** |

## Latency

| arch | model | b | batch | angular_tiled | complex128 | complex64 | neon | onnx | best angular vs onnx |
|---|---|---|---|---|---|---|---|---|---|
| aarch64 | head_512x10 | 3 | 64 | 2.54 | 0.09 | 0.07 | 0.99 | 0.20 | 0.205x |
| aarch64 | head_512x10 | 3 | 512 | 19.82 | 0.56 | 0.46 | 7.16 | 1.21 | 0.169x |
| aarch64 | head_512x10 | 3 | 2048 | 78.80 | 2.16 | 1.78 | 28.97 | 4.53 | 0.156x |
| aarch64 | head_512x10 | 4 | 64 | 2.53 | 0.09 | 0.07 | 0.98 | 0.21 | 0.208x |
| aarch64 | head_512x10 | 4 | 512 | 19.76 | 0.56 | 0.46 | 7.14 | 1.22 | 0.170x |
| aarch64 | head_512x10 | 4 | 2048 | 78.96 | 2.15 | 1.71 | 29.26 | 4.97 | 0.170x |
| aarch64 | layer_2048x64 | 3 | 64 | 44.01 | 1.18 | 0.71 | 4.62 | 1.65 | 0.358x |
| aarch64 | layer_2048x64 | 3 | 512 | 351.81 | 8.22 | 4.78 | 36.56 | 12.57 | 0.344x |
| aarch64 | layer_2048x64 | 3 | 2048 | 1416.46 | 36.65 | 22.75 | 154.66 | 56.10 | 0.363x |
| aarch64 | layer_2048x64 | 4 | 64 | 44.14 | 1.20 | 0.70 | 4.57 | 1.67 | 0.365x |
| aarch64 | layer_2048x64 | 4 | 512 | 352.30 | 8.31 | 4.76 | 36.73 | 12.61 | 0.343x |
| aarch64 | layer_2048x64 | 4 | 2048 | 1415.52 | 36.53 | 22.69 | 155.14 | 58.48 | 0.377x |
| aarch64 | layer_4096x256 | 3 | 64 | 329.33 | 8.46 | 4.48 | 15.85 | 11.11 | 0.701x |
| aarch64 | layer_4096x256 | 3 | 512 | 2626.88 | 59.61 | 31.61 | 129.84 | 89.37 | 0.688x |
| aarch64 | layer_4096x256 | 3 | 2048 | 10582.43 | 236.44 | 125.83 | 522.44 | 369.63 | 0.708x |
| aarch64 | layer_4096x256 | 4 | 64 | 330.62 | 8.47 | 4.51 | 15.88 | 11.09 | 0.698x |
| aarch64 | layer_4096x256 | 4 | 512 | 2638.24 | 59.35 | 31.42 | 127.06 | 88.50 | 0.697x |
| aarch64 | layer_4096x256 | 4 | 2048 | 10813.39 | 236.29 | 125.65 | 527.07 | 375.20 | 0.712x |
| aarch64 | layer_6144x384 | 4 | 2048 | - | - | - | 1000.15 | 803.53 | 0.803x |
| aarch64 | layer_8192x512 | 4 | 2048 | - | - | - | 1620.71 | 1443.36 | 0.891x |
| x86_64 | head_512x10 | 3 | 64 | 2.90 | 0.09 | 0.09 | - | 0.13 | 0.046x |
| x86_64 | head_512x10 | 3 | 512 | 22.76 | 0.54 | 0.45 | - | 1.01 | 0.044x |
| x86_64 | head_512x10 | 3 | 2048 | 91.63 | 2.19 | 2.13 | - | 4.05 | 0.044x |
| x86_64 | head_512x10 | 4 | 64 | 3.62 | 0.09 | 0.09 | - | 0.17 | 0.047x |
| x86_64 | head_512x10 | 4 | 512 | 22.73 | 0.55 | 0.45 | - | 1.01 | 0.044x |
| x86_64 | head_512x10 | 4 | 2048 | 91.69 | 2.25 | 2.19 | 55.33 | 3.58 | 0.065x |
| x86_64 | layer_2048x64 | 3 | 64 | 45.46 | 1.04 | 0.65 | - | 0.97 | 0.021x |
| x86_64 | layer_2048x64 | 3 | 512 | 363.85 | 6.99 | 4.56 | - | 7.46 | 0.021x |
| x86_64 | layer_2048x64 | 3 | 2048 | 1473.88 | 33.89 | 18.75 | - | 40.43 | 0.027x |
| x86_64 | layer_2048x64 | 4 | 64 | 45.50 | 1.03 | 0.63 | - | 0.95 | 0.021x |
| x86_64 | layer_2048x64 | 4 | 512 | 364.48 | 7.03 | 4.42 | - | 7.45 | 0.020x |
| x86_64 | layer_2048x64 | 4 | 2048 | 1475.23 | 33.50 | 21.42 | 566.05 | 41.19 | 0.073x |
| x86_64 | layer_4096x256 | 3 | 64 | 330.41 | 7.02 | 4.06 | - | 6.07 | 0.018x |
| x86_64 | layer_4096x256 | 3 | 512 | 2666.37 | 49.56 | 27.31 | - | 51.27 | 0.019x |
| x86_64 | layer_4096x256 | 3 | 2048 | 10697.46 | 203.13 | 110.69 | - | 217.09 | 0.020x |
| x86_64 | layer_4096x256 | 4 | 64 | 329.58 | 7.00 | 4.06 | - | 6.07 | 0.018x |
| x86_64 | layer_4096x256 | 4 | 512 | 2662.54 | 49.86 | 27.35 | - | 54.94 | 0.021x |
| x86_64 | layer_4096x256 | 4 | 2048 | 10740.23 | 202.72 | 115.62 | 3471.62 | 222.06 | 0.064x |
| x86_64 | layer_6144x384 | 4 | 2048 | - | - | - | 7554.73 | 460.95 | 0.061x |
| x86_64 | layer_8192x512 | 4 | 2048 | - | - | - | 13172.56 | 810.01 | 0.061x |

## Weight memory

| model | weights | complex128 | fp32 re/im | uint8 idx | packed b-bit | vs fp32 |
|---|---|---|---|---|---|---|
| head_512x10 (b=3) | 5,130 | 80 KiB | 40 KiB | 5 KiB | 2 KiB | **21x** |
| head_512x10 (b=4) | 5,130 | 80 KiB | 40 KiB | 5 KiB | 3 KiB | **16x** |
| layer_2048x64 (b=3) | 131,136 | 2049 KiB | 1024 KiB | 128 KiB | 48 KiB | **21x** |
| layer_2048x64 (b=4) | 131,136 | 2049 KiB | 1024 KiB | 128 KiB | 64 KiB | **16x** |
| layer_4096x256 (b=3) | 1,048,832 | 16388 KiB | 8194 KiB | 1024 KiB | 384 KiB | **21x** |
| layer_4096x256 (b=4) | 1,048,832 | 16388 KiB | 8194 KiB | 1024 KiB | 512 KiB | **16x** |
| layer_6144x384 (b=4) | 2,359,680 | 36870 KiB | 18435 KiB | 2304 KiB | 1152 KiB | **16x** |
| layer_8192x512 (b=4) | 4,194,816 | 65544 KiB | 32772 KiB | 4096 KiB | 2048 KiB | **16x** |

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
| head_512x10 | 4 | neon | 0.8687 | True |
| head_512x10 | 4 | neon | 0.8687 | True |
| head_512x10 | 4 | onnx | 0.9023 | False |
| head_512x10 | 4 | onnx | 0.9023 | False |
| head_512x10 | 4 | onnx | 0.9019 | False |
| head_512x10 | 4 | onnx | 0.9019 | False |
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
| layer_2048x64 | 4 | neon | 0.7949 | True |
| layer_2048x64 | 4 | neon | 0.7949 | True |
| layer_2048x64 | 4 | onnx | 0.8525 | False |
| layer_2048x64 | 4 | onnx | 0.8525 | False |
| layer_2048x64 | 4 | onnx | 0.8477 | False |
| layer_2048x64 | 4 | onnx | 0.8477 | False |
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
| layer_4096x256 | 4 | neon | 0.7563 | True |
| layer_4096x256 | 4 | neon | 0.7563 | True |
| layer_4096x256 | 4 | onnx | 0.8291 | False |
| layer_4096x256 | 4 | onnx | 0.8291 | False |
| layer_4096x256 | 4 | onnx | 0.8213 | False |
| layer_4096x256 | 4 | onnx | 0.8213 | False |
| layer_6144x384 | 4 | neon | 0.7373 | True |
| layer_6144x384 | 4 | neon | 0.7373 | True |
| layer_6144x384 | 4 | onnx | 0.8154 | False |
| layer_6144x384 | 4 | onnx | 0.8154 | False |
| layer_8192x512 | 4 | neon | 0.7510 | True |
| layer_8192x512 | 4 | neon | 0.7510 | True |
| layer_8192x512 | 4 | onnx | 0.8081 | False |
| layer_8192x512 | 4 | onnx | 0.8081 | False |
