
### T0 (HF bf16 n=500 do mesmo recorte: FIM 0.384, exec 0.564, campos inexist. 0.040, sintaxe 0.992; n=2000: {'fim': 0.403, 'exec': 0.582})

| variante | MB | tok.nativo=HF | greedy idêntico % | texto idêntico % | prefixo aceito % | FIM acc (Δpp; p) | exec-valid (Δpp; p) | campos inex. (Δpp) | sintaxe | TTFT méd/p95 ms | tok/s | RAM pico GB | VRAM MB | KapiLab TTFT méd/p95 | gates proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32-cpu | 1904 | 100.0% | 97.0 | 97.5 | 98.0 | 0.382 (-0.2; 1.0) | 0.564 (+0.0; 1.0) | 0.040 (+0.0) | 0.992 | 350/572 | 27.3 | 3.326 | - | - | FALHA: ttft_mean_ms,ttft_p95_ms,ram_peak_gb,disk_mb,greedy (cenário informativo) |
| int4 | 392 | 100.0% | 79.0 | 79.0 | 84.9 | 0.390 (+0.6; 0.5811) | 0.562 (-0.2; 1.0) | 0.036 (-0.4) | 0.986 | 285/480 | 113.4 | 1.533 | - | 297/502 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_peak_gb,disk_mb,delta_apt,exec_match |
| int4-rtn | 317 | 100.0% | 70.0 | 70.0 | 78.7 | 0.388 (+0.4; 0.8238) | 0.562 (-0.2; 1.0) | 0.048 (+0.8) | 0.994 | 297/492 | 120.1 | 1.394 | - | 302/487 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_peak_gb,disk_mb,delta_apt,exec_match |
| int4-kquantmixed | 423 | 100.0% | 81.5 | 81.5 | 86.7 | 0.392 (+0.8; 0.3438) | 0.568 (+0.4; 0.6875) | 0.032 (-0.8) | 0.990 | 287/475 | 101.6 | 1.563 | - | 281/476 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_peak_gb,disk_mb,delta_apt,exec_match |
| int8 | 1071 | 100.0% | 95.0 | 95.0 | 96.7 | 0.384 (+0.0; 1.0) | 0.568 (+0.4; 0.5) | 0.038 (-0.2) | 0.992 | 262/440 | 92.8 | 1.575 | - | 276/462 | FALHA: ttft_mean_ms,ttft_p95_ms,disk_mb | ok: ram_peak_gb,delta_apt |
| int8-emb4 | 633 | 100.0% | 93.0 | 93.5 | 95.4 | 0.386 (+0.2; 1.0) | 0.568 (+0.4; 0.5) | 0.036 (-0.4) | 0.996 | 263/442 | 92.1 | 1.575 | - | 269/442 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_peak_gb,disk_mb,delta_apt |
| dml-fp16 | 959 | 100.0% | 97.5 | 98.0 | 98.4 | 0.382 (-0.2; 1.0) | 0.564 (+0.0; 1.0) | 0.040 (+0.0) | 0.992 | 109/124 | 166.4 | 2.271 | 2933.1 | - | FALHA: ram_peak_gb,disk_mb | ok: ttft_mean_ms,ttft_p95_ms,greedy (cenário informativo) |
| dml-int4 | 544 | 100.0% | 76.5 | 76.5 | 85.1 | 0.382 (-0.2; 1.0) | 0.560 (-0.4; 0.7539) | 0.046 (+0.6) | 0.990 | 142/165 | 216.3 | 2.5 | 2530.3 | - | FALHA: ram_peak_gb | ok: ttft_mean_ms,ttft_p95_ms,disk_mb,delta_apt,exec_match (cenário informativo) |

### T1 (HF bf16 n=500 do mesmo recorte: FIM 0.412, exec 0.582, campos inexist. 0.026, sintaxe 0.994; n=2000: {'fim': 0.4105, 'exec': 0.595})

| variante | MB | tok.nativo=HF | greedy idêntico % | texto idêntico % | prefixo aceito % | FIM acc (Δpp; p) | exec-valid (Δpp; p) | campos inex. (Δpp) | sintaxe | TTFT méd/p95 ms | tok/s | RAM pico GB | VRAM MB | KapiLab TTFT méd/p95 | gates proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32-cpu | 5916 | 100.0% | 96.0 | 96.0 | 97.4 | 0.408 (-0.4; 0.5) | 0.580 (-0.2; 1.0) | 0.028 (+0.2) | 0.992 | - | - | - | - | - | - |
| int4 | 1090 | 100.0% | 78.5 | 78.5 | 84.3 | 0.398 (-1.4; 0.0391) | 0.580 (-0.2; 1.0) | 0.032 (+0.6) | 0.994 | 788/1317 | 47.0 | 2.631 | - | 787/1293 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_gb,delta_apt,exec_match (cenário informativo) |
| int4-rtn | 952 | 100.0% | 73.0 | 73.0 | 80.5 | 0.394 (-1.8; 0.0039) | 0.568 (-1.4; 0.0391) | 0.038 (+1.2) | 0.990 | 800/1372 | 49.3 | 2.386 | - | 790/1282 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_gb,delta_apt,exec_match (cenário informativo) |
| int8 | 2578 | 100.0% | 94.5 | 94.5 | 96.6 | 0.408 (-0.4; 0.5) | 0.582 (+0.0; 1.0) | 0.026 (+0.0) | 0.994 | 697/1192 | 33.1 | 3.012 | - | 698/1139 | FALHA: ttft_mean_ms,ttft_p95_ms | ok: ram_gb,delta_apt |
| dml-fp16 | 2964 | 100.0% | 96.5 | 96.5 | 97.7 | 0.408 (-0.4; 0.5) | 0.582 (+0.0; 1.0) | 0.026 (+0.0) | 0.992 | 148/170 | 103.8 | 2.261 | 5133.1 | - | FALHA: greedy | ok: ttft_mean_ms,ttft_p95_ms,decode_tok_s |
| dml-int4 | 1293 | 100.0% | 78.0 | 78.0 | 83.3 | 0.394 (-1.8; 0.0117) | 0.574 (-0.8; 0.2188) | 0.034 (+0.8) | 0.992 | 173/208 | 167.4 | 2.511 | 3062.4 | - | FALHA: ttft_mean_ms,ttft_p95_ms,delta_apt | ok: decode_tok_s,exec_match (cenário informativo) |
