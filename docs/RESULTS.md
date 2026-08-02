# Benchmark results

_190 records; hosts: AMD EPYC 9V74 80-Core Processor, Neoverse-N2_

## Crossover: neon vs ONNX against layer size

| arch | model | layer MACs | b | batch | onnx ms | neon ms | neon/onnx | fp32 re/im | uint8 idx | packed b-bit | smaller by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aarch64 | head_512x10 | 0.005M | 3 | 64 | 0.19 | 0.97 | **0.192** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **21x** |
| aarch64 | head_512x10 | 0.005M | 4 | 64 | 0.18 | 0.98 | **0.188** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **16x** |
| aarch64 | head_512x10 | 0.005M | 3 | 512 | 1.17 | 7.05 | **0.166** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **21x** |
| aarch64 | head_512x10 | 0.005M | 4 | 512 | 1.17 | 7.05 | **0.166** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **16x** |
| aarch64 | head_512x10 | 0.005M | 3 | 2048 | 4.40 | 28.10 | **0.157** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **21x** |
| aarch64 | head_512x10 | 0.005M | 4 | 2048 | 4.54 | 28.75 | **0.158** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **16x** |
| aarch64 | layer_2048x64 | 0.131M | 3 | 64 | 1.63 | 4.53 | **0.360** | 1.0 MiB | 0.1 MiB | 0.0 MiB | **21x** |
| aarch64 | layer_2048x64 | 0.131M | 4 | 64 | 1.62 | 4.53 | **0.358** | 1.0 MiB | 0.1 MiB | 0.1 MiB | **16x** |
| aarch64 | layer_2048x64 | 0.131M | 3 | 512 | 12.13 | 36.52 | **0.332** | 1.0 MiB | 0.1 MiB | 0.0 MiB | **21x** |
| aarch64 | layer_2048x64 | 0.131M | 4 | 512 | 12.78 | 36.10 | **0.354** | 1.0 MiB | 0.1 MiB | 0.1 MiB | **16x** |
| aarch64 | layer_2048x64 | 0.131M | 3 | 2048 | 57.02 | 153.41 | **0.372** | 1.0 MiB | 0.1 MiB | 0.0 MiB | **21x** |
| aarch64 | layer_2048x64 | 0.131M | 4 | 2048 | 55.28 | 152.53 | **0.362** | 1.0 MiB | 0.1 MiB | 0.1 MiB | **16x** |
| aarch64 | layer_4096x256 | 1.049M | 3 | 64 | 11.03 | 15.78 | **0.699** | 8.0 MiB | 1.0 MiB | 0.4 MiB | **21x** |
| aarch64 | layer_4096x256 | 1.049M | 4 | 64 | 10.98 | 15.79 | **0.696** | 8.0 MiB | 1.0 MiB | 0.5 MiB | **16x** |
| aarch64 | layer_4096x256 | 1.049M | 3 | 512 | 90.13 | 128.16 | **0.703** | 8.0 MiB | 1.0 MiB | 0.4 MiB | **21x** |
| aarch64 | layer_4096x256 | 1.049M | 4 | 512 | 88.24 | 128.56 | **0.686** | 8.0 MiB | 1.0 MiB | 0.5 MiB | **16x** |
| aarch64 | layer_4096x256 | 1.049M | 3 | 2048 | 366.99 | 522.39 | **0.703** | 8.0 MiB | 1.0 MiB | 0.4 MiB | **21x** |
| aarch64 | layer_4096x256 | 1.049M | 4 | 2048 | 368.69 | 521.78 | **0.707** | 8.0 MiB | 1.0 MiB | 0.5 MiB | **16x** |
| aarch64 | layer_6144x384 | 2.360M | 4 | 2048 | 799.74 | 993.41 | **0.805** | 18.0 MiB | 2.3 MiB | 1.1 MiB | **16x** |
| aarch64 | layer_8192x512 | 4.195M | 4 | 2048 | 1426.57 | 1609.12 | **0.887** | 32.0 MiB | 4.0 MiB | 2.0 MiB | **16x** |
| aarch64 | layer_12288x768 | 9.438M | 4 | 256 | 390.48 | 406.39 | **0.961** | 72.0 MiB | 9.0 MiB | 4.5 MiB | **16x** |
| aarch64 | layer_16384x1024 | 16.778M | 4 | 256 | 697.88 | 687.07 | **1.016** | 128.0 MiB | 16.0 MiB | 8.0 MiB | **16x** |
| x86_64 | head_512x10 | 0.005M | 4 | 2048 | 3.76 | 63.84 | **0.059** | 0.0 MiB | 0.0 MiB | 0.0 MiB | **16x** |
| x86_64 | layer_2048x64 | 0.131M | 4 | 2048 | 40.31 | 727.83 | **0.055** | 1.0 MiB | 0.1 MiB | 0.1 MiB | **16x** |
| x86_64 | layer_4096x256 | 1.049M | 4 | 2048 | 230.14 | 4738.86 | **0.049** | 8.0 MiB | 1.0 MiB | 0.5 MiB | **16x** |
| x86_64 | layer_6144x384 | 2.360M | 4 | 2048 | 490.70 | 10363.35 | **0.047** | 18.0 MiB | 2.3 MiB | 1.1 MiB | **16x** |
| x86_64 | layer_8192x512 | 4.195M | 4 | 2048 | 866.05 | 18214.34 | **0.048** | 32.0 MiB | 4.0 MiB | 2.0 MiB | **16x** |
| x86_64 | layer_12288x768 | 9.438M | 4 | 256 | 233.75 | 5023.17 | **0.047** | 72.0 MiB | 9.0 MiB | 4.5 MiB | **16x** |
| x86_64 | layer_16384x1024 | 16.778M | 4 | 256 | 413.09 | 8869.04 | **0.047** | 128.0 MiB | 16.0 MiB | 8.0 MiB | **16x** |

## Latency

| arch | model | b | batch | angular_tiled | complex128 | complex64 | neon | onnx | best angular vs onnx |
|---|---|---|---|---|---|---|---|---|---|
| aarch64 | head_512x10 | 3 | 64 | 2.51 | 0.08 | 0.07 | 0.97 | 0.19 | 0.192x |
| aarch64 | head_512x10 | 3 | 512 | 19.42 | 0.54 | 0.43 | 7.05 | 1.17 | 0.166x |
| aarch64 | head_512x10 | 3 | 2048 | 78.02 | 2.07 | 1.66 | 28.10 | 4.40 | 0.157x |
| aarch64 | head_512x10 | 4 | 64 | 2.50 | 0.08 | 0.07 | 0.98 | 0.18 | 0.188x |
| aarch64 | head_512x10 | 4 | 512 | 19.41 | 0.54 | 0.43 | 7.05 | 1.17 | 0.166x |
| aarch64 | head_512x10 | 4 | 2048 | 77.80 | 2.10 | 1.63 | 28.75 | 4.54 | 0.158x |
| aarch64 | layer_12288x768 | 4 | 256 | - | - | - | 406.39 | 390.48 | 0.961x |
| aarch64 | layer_16384x1024 | 4 | 256 | - | - | - | 687.07 | 697.88 | 1.016x |
| aarch64 | layer_2048x64 | 3 | 64 | 43.71 | 1.15 | 0.69 | 4.53 | 1.63 | 0.360x |
| aarch64 | layer_2048x64 | 3 | 512 | 352.18 | 8.10 | 4.56 | 36.52 | 12.13 | 0.332x |
| aarch64 | layer_2048x64 | 3 | 2048 | 1410.55 | 36.22 | 22.68 | 153.41 | 57.02 | 0.372x |
| aarch64 | layer_2048x64 | 4 | 64 | 43.63 | 1.15 | 0.69 | 4.53 | 1.62 | 0.358x |
| aarch64 | layer_2048x64 | 4 | 512 | 351.39 | 8.26 | 4.75 | 36.10 | 12.78 | 0.354x |
| aarch64 | layer_2048x64 | 4 | 2048 | 1414.56 | 35.76 | 22.16 | 152.53 | 55.28 | 0.362x |
| aarch64 | layer_4096x256 | 3 | 64 | 328.87 | 8.26 | 4.47 | 15.78 | 11.03 | 0.699x |
| aarch64 | layer_4096x256 | 3 | 512 | 2627.87 | 59.35 | 31.56 | 128.16 | 90.13 | 0.703x |
| aarch64 | layer_4096x256 | 3 | 2048 | 10500.46 | 236.30 | 126.98 | 522.39 | 366.99 | 0.703x |
| aarch64 | layer_4096x256 | 4 | 64 | 327.77 | 8.20 | 4.44 | 15.79 | 10.98 | 0.696x |
| aarch64 | layer_4096x256 | 4 | 512 | 2628.55 | 59.04 | 31.45 | 128.56 | 88.24 | 0.686x |
| aarch64 | layer_4096x256 | 4 | 2048 | 10507.87 | 235.98 | 126.10 | 521.78 | 368.69 | 0.707x |
| aarch64 | layer_6144x384 | 4 | 2048 | - | - | - | 993.41 | 799.74 | 0.805x |
| aarch64 | layer_8192x512 | 4 | 2048 | - | - | - | 1609.12 | 1426.57 | 0.887x |
| x86_64 | head_512x10 | 3 | 64 | 2.69 | 0.13 | 0.10 | - | 0.13 | 0.048x |
| x86_64 | head_512x10 | 3 | 512 | 21.74 | 0.55 | 0.52 | - | 1.00 | 0.046x |
| x86_64 | head_512x10 | 3 | 2048 | 86.00 | 2.12 | 2.05 | - | 4.43 | 0.052x |
| x86_64 | head_512x10 | 4 | 64 | 3.35 | 0.13 | 0.10 | - | 0.16 | 0.049x |
| x86_64 | head_512x10 | 4 | 512 | 21.75 | 0.56 | 0.53 | - | 1.01 | 0.046x |
| x86_64 | head_512x10 | 4 | 2048 | 86.70 | 2.27 | 2.14 | 63.84 | 3.76 | 0.059x |
| x86_64 | layer_12288x768 | 4 | 256 | - | - | - | 5023.17 | 233.75 | 0.047x |
| x86_64 | layer_16384x1024 | 4 | 256 | - | - | - | 8869.04 | 413.09 | 0.047x |
| x86_64 | layer_2048x64 | 3 | 64 | 39.14 | 1.26 | 0.80 | - | 0.99 | 0.025x |
| x86_64 | layer_2048x64 | 3 | 512 | 314.20 | 7.59 | 4.73 | - | 9.09 | 0.029x |
| x86_64 | layer_2048x64 | 3 | 2048 | 1266.38 | 34.47 | 24.59 | - | 46.29 | 0.037x |
| x86_64 | layer_2048x64 | 4 | 64 | 39.00 | 1.27 | 0.80 | - | 0.99 | 0.025x |
| x86_64 | layer_2048x64 | 4 | 512 | 315.03 | 7.63 | 4.74 | - | 9.19 | 0.029x |
| x86_64 | layer_2048x64 | 4 | 2048 | 1277.28 | 34.37 | 24.42 | 727.83 | 40.31 | 0.055x |
| x86_64 | layer_4096x256 | 3 | 64 | 279.65 | 7.57 | 4.44 | - | 7.12 | 0.025x |
| x86_64 | layer_4096x256 | 3 | 512 | 2249.95 | 53.41 | 28.74 | - | 56.44 | 0.025x |
| x86_64 | layer_4096x256 | 3 | 2048 | 8968.80 | 214.73 | 115.85 | - | 232.27 | 0.026x |
| x86_64 | layer_4096x256 | 4 | 64 | 279.71 | 7.64 | 4.38 | - | 7.17 | 0.026x |
| x86_64 | layer_4096x256 | 4 | 512 | 2251.57 | 53.10 | 28.95 | - | 55.99 | 0.025x |
| x86_64 | layer_4096x256 | 4 | 2048 | 8931.65 | 213.48 | 115.65 | 4738.86 | 230.14 | 0.049x |
| x86_64 | layer_6144x384 | 4 | 2048 | - | - | - | 10363.35 | 490.70 | 0.047x |
| x86_64 | layer_8192x512 | 4 | 2048 | - | - | - | 18214.34 | 866.05 | 0.048x |

## Weight memory

| model | weights | complex128 | fp32 re/im | uint8 idx | packed b-bit | vs fp32 |
|---|---|---|---|---|---|---|
| head_512x10 (b=3) | 5,130 | 80 KiB | 40 KiB | 5 KiB | 2 KiB | **21x** |
| head_512x10 (b=4) | 5,130 | 80 KiB | 40 KiB | 5 KiB | 3 KiB | **16x** |
| layer_12288x768 (b=4) | 9,437,952 | 147468 KiB | 73734 KiB | 9217 KiB | 4608 KiB | **16x** |
| layer_16384x1024 (b=4) | 16,778,240 | 262160 KiB | 131080 KiB | 16385 KiB | 8192 KiB | **16x** |
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
| layer_2048x64 | 3 | angular_tiled | 0.6431 | True |
| layer_2048x64 | 3 | angular_tiled | 0.6431 | True |
| layer_2048x64 | 3 | complex128 | 0.7314 | False |
| layer_2048x64 | 3 | complex128 | 0.7314 | False |
| layer_2048x64 | 3 | complex64 | 0.7314 | False |
| layer_2048x64 | 3 | complex64 | 0.7314 | False |
| layer_2048x64 | 3 | neon | 0.6431 | True |
| layer_2048x64 | 3 | onnx | 0.7314 | False |
| layer_2048x64 | 3 | onnx | 0.7314 | False |
| layer_2048x64 | 4 | angular_tiled | 0.8032 | True |
| layer_2048x64 | 4 | angular_tiled | 0.8032 | True |
| layer_2048x64 | 4 | complex128 | 0.8574 | False |
| layer_2048x64 | 4 | complex128 | 0.8574 | False |
| layer_2048x64 | 4 | complex64 | 0.8574 | False |
| layer_2048x64 | 4 | complex64 | 0.8574 | False |
| layer_2048x64 | 4 | neon | 0.8032 | True |
| layer_2048x64 | 4 | neon | 0.7964 | True |
| layer_2048x64 | 4 | neon | 0.7964 | True |
| layer_2048x64 | 4 | onnx | 0.8574 | False |
| layer_2048x64 | 4 | onnx | 0.8574 | False |
| layer_2048x64 | 4 | onnx | 0.8540 | False |
| layer_2048x64 | 4 | onnx | 0.8540 | False |
| layer_4096x256 | 3 | angular_tiled | 0.5640 | True |
| layer_4096x256 | 3 | angular_tiled | 0.5640 | True |
| layer_4096x256 | 3 | complex128 | 0.6851 | False |
| layer_4096x256 | 3 | complex128 | 0.6851 | False |
| layer_4096x256 | 3 | complex64 | 0.6851 | False |
| layer_4096x256 | 3 | complex64 | 0.6851 | False |
| layer_4096x256 | 3 | neon | 0.5640 | True |
| layer_4096x256 | 3 | onnx | 0.6851 | False |
| layer_4096x256 | 3 | onnx | 0.6851 | False |
| layer_4096x256 | 4 | angular_tiled | 0.7617 | True |
| layer_4096x256 | 4 | angular_tiled | 0.7617 | True |
| layer_4096x256 | 4 | complex128 | 0.8218 | False |
| layer_4096x256 | 4 | complex128 | 0.8218 | False |
| layer_4096x256 | 4 | complex64 | 0.8218 | False |
| layer_4096x256 | 4 | complex64 | 0.8218 | False |
| layer_4096x256 | 4 | neon | 0.7617 | True |
| layer_4096x256 | 4 | neon | 0.7622 | True |
| layer_4096x256 | 4 | neon | 0.7622 | True |
| layer_4096x256 | 4 | onnx | 0.8218 | False |
| layer_4096x256 | 4 | onnx | 0.8218 | False |
| layer_4096x256 | 4 | onnx | 0.8359 | False |
| layer_4096x256 | 4 | onnx | 0.8359 | False |
| layer_6144x384 | 4 | neon | 0.7593 | True |
| layer_6144x384 | 4 | neon | 0.7593 | True |
| layer_6144x384 | 4 | onnx | 0.8286 | False |
| layer_6144x384 | 4 | onnx | 0.8286 | False |
| layer_8192x512 | 4 | neon | 0.7671 | True |
| layer_8192x512 | 4 | neon | 0.7671 | True |
| layer_8192x512 | 4 | onnx | 0.8281 | False |
| layer_8192x512 | 4 | onnx | 0.8281 | False |
