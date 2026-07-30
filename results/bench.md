| encoder | runtime | precision | modality | batch | p50 ms | p95 ms | items/s | model MB | ΔRSS MB | parity cos |
|---|---|---|---|---|---|---|---|---|---|---|
| clip | pytorch | fp32 | image | 1 | 69.9 | 120.2 | 13.8 | - | 0 | 1.0000 |
| clip | pytorch | fp32 | image | 8 | 347.8 | 451.9 | 22.6 | - | 1 | 1.0000 |
| clip | pytorch | fp32 | text | 4 | 30.8 | 44.4 | 127.9 | - | 0 | 1.0000 |
| clip | onnxruntime | fp32 | image | 1 | 78.1 | 91.6 | 12.6 | 335 | 0 | 1.0000 |
| clip | onnxruntime | fp32 | image | 8 | 506.1 | 553.4 | 15.6 | 335 | 0 | 1.0000 |
| clip | onnxruntime | fp32 | text | 4 | 27.8 | 31.3 | 144.5 | 242 | 0 | 1.0000 |
| clip | onnxruntime | int8 | image | 1 | 55.4 | 68.5 | 16.8 | 84 | 0 | 0.9764 |
| clip | onnxruntime | int8 | image | 8 | 354.1 | 369.7 | 22.5 | 84 | 0 | 0.9764 |
| clip | onnxruntime | int8 | text | 4 | 14.3 | 16.4 | 282.3 | 61 | 0 | 0.8830 |
