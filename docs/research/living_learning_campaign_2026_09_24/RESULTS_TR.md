# Tur 23 — ilk dalganın eksiksiz sonuçları

48 koşullu iddia; 3456 eşleştirilmiş sonuç; 288 motor-yaşamı. 24 keşif, 48 doğrulama tohumu.

Karar doğrulama grubuna aittir. Pozitif Δ = kontrol − aday. Ölçütler aileler arasında aynı değildir.
Sanal hesap sayacı CPU süresi değildir. Destek yalnız belirtilen yapay dünya, ölçüt ve rakibe ilişkindir.

| Kimlik | Rejim | Aday / kontrol | Aday ölçütü | Kontrol ölçütü | Δ | + / = / − | Holm p (destek / ters) | Karar |
|---|---|---|---:|---:|---:|---|---|---|
| ENV-01 | stationary_gap | ucb / greedy | 0.00561523 | 0.0512695 | 0.0456543 | 4 / 0 / 44 | 1 / 4.54e-08 | Belirsiz |
| ENV-02 | stationary_gap | epsilon / greedy | 0.0333496 | 0.0512695 | 0.0179199 | 4 / 0 / 44 | 1 / 4.54e-08 | Belirsiz |
| ENV-03 | stationary_gap | window_ucb / ucb | 0.0270996 | 0.00561523 | -0.0214844 | 0 / 0 / 48 | 1 / 3.41e-13 | Ters sonuç |
| ENV-04 | stationary_small | ucb / greedy | 0.0184245 | 0.0419027 | 0.0234782 | 20 / 0 / 28 | 1 / 1 | Belirsiz |
| ENV-05 | stationary_small | epsilon / greedy | 0.0205119 | 0.0419027 | 0.0213908 | 20 / 0 / 28 | 1 / 1 | Belirsiz |
| ENV-06 | stationary_small | window_ucb / ucb | 0.031075 | 0.0184245 | -0.0126506 | 9 / 0 / 39 | 1 / 0.000426 | Ters sonuç |
| ENV-07 | single_reversal | ucb / greedy | 0.0727539 | 0.30127 | 0.228516 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| ENV-08 | single_reversal | epsilon / greedy | 0.28313 | 0.30127 | 0.0181396 | 34 / 3 / 11 | 0.0218 / 1 | Desteklendi |
| ENV-09 | single_reversal | window_ucb / ucb | 0.0365479 | 0.0727539 | 0.0362061 | 31 / 2 / 15 | 0.673 / 1 | Belirsiz |
| ENV-10 | recurring_reversal | ucb / greedy | 0.113525 | 0.301123 | 0.187598 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| ENV-11 | recurring_reversal | epsilon / greedy | 0.278564 | 0.301123 | 0.0225586 | 33 / 5 / 10 | 0.0164 / 1 | Desteklendi |
| ENV-12 | recurring_reversal | window_ucb / ucb | 0.0663818 | 0.113525 | 0.0471436 | 39 / 1 / 8 | 0.000158 / 1 | Desteklendi |
| MEM01 | clustered | reservoir / fifo | 0.123138 | 0.196757 | 0.0736186 | 45 / 0 / 3 | 4e-09 / 1 | Desteklendi |
| MEM02 | stable | count / reservoir | 0.0229899 | 0.0923869 | 0.069397 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM03 | stable | count / last | 0.0229899 | 0.248708 | 0.225718 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM04 | abrupt | window / count | 0.101507 | 0.231883 | 0.130376 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM05 | staggered | decay / count | 0.133718 | 0.387304 | 0.253586 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM06 | abrupt | window / reservoir | 0.101507 | 0.228861 | 0.127355 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM07 | stable | count / decay | 0.0229899 | 0.0511983 | 0.0282084 | 47 / 0 / 1 | 1.17e-11 / 1 | Desteklendi |
| MEM08 | rare | window / fifo | 0.083725 | 0.17454 | 0.0908152 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM09 | revisions | revision_delta / revision_ignore | 0.257996 | 0.36852 | 0.110524 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM10 | revisions | revision_delta / revision_append | 0.257996 | 0.306895 | 0.0488993 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM11 | revisions | revision_delta / revision_rebuild | 15 | 619.25 | 604.25 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| MEM12 | stable | count / window | 0.0229899 | 0.0785929 | 0.055603 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| REP01 | interaction | adaptive_decay1_replay / linear_decay1_replay | 0.228576 | 1.16095 | 0.932371 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| REP02 | interaction | adaptive_decay1_replay / full_decay1_replay | 0.228576 | 0.0933235 | -0.135253 | 0 / 0 / 48 | 1 / 3.41e-13 | Ters sonuç |
| REP03 | additive | adaptive_decay1_replay / full_decay1_replay | 0.0347116 | 0.0481675 | 0.0134559 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| REP04 | interaction | adaptive_decay1_replay / random_decay1_replay | 0.228576 | 0.762083 | 0.533506 | 44 / 4 / 0 | 3.87e-12 / 1 | Desteklendi |
| REP05 | switch | adaptive_decay0.97_replay / adaptive_decay1_replay | 0.468709 | 0.751435 | 0.282726 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| REP06 | switch | adaptive_decay0.97_replay / full_decay0.97_replay | 0.468709 | 0.300908 | -0.1678 | 0 / 0 / 48 | 1 / 3.41e-13 | Ters sonuç |
| REP07 | noisy | adaptive_decay1_replay / full_decay1_replay | 1.66571 | 1.78165 | 0.115935 | 47 / 0 / 1 | 1.17e-11 / 1 | Desteklendi |
| REP08 | rare | adaptive_decay1_replay / linear_decay1_replay | 0.137988 | 0.497115 | 0.359128 | 47 / 0 / 1 | 1.17e-11 / 1 | Desteklendi |
| REP09 | covariate | adaptive_decay1_replay / full_decay1_replay | 0.250027 | 0.0905464 | -0.15948 | 2 / 0 / 46 | 1 / 2.63e-10 | Ters sonuç |
| REP10 | onset | adaptive_decay0.97_replay / full_decay0.97_replay | 0.176514 | 0.155911 | -0.0206027 | 11 / 0 / 37 | 1 / 0.00611 | Ters sonuç |
| REP11 | interaction | adaptive_decay1_replay / full_decay1_replay | 72270.9 | 160662 | 88391.1 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| REP12 | interaction | adaptive_decay1_replay / adaptive_decay1_zero_history | 0.228576 | 0.265449 | 0.0368731 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| UPD01 | stationary_low | gain_meta / fixed_005 | 0.000178319 | 0.000788771 | 0.000610451 | 41 / 0 / 7 | 1.81e-05 / 1 | Desteklendi |
| UPD02 | stationary_noisy | gain_meta / fixed_005 | 0.00324448 | 0.00121361 | -0.00203087 | 0 / 0 / 48 | 1 / 3.41e-13 | Ters sonuç |
| UPD03 | drift | gain_meta / fixed_02 | 0.000525892 | 0.000742054 | 0.000216162 | 48 / 0 / 0 | 3.41e-13 / 1 | Belirsiz |
| UPD04 | jumps | gain_meta / fixed_02 | 0.00304254 | 0.00487513 | 0.00183259 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| UPD05 | reversals | gain_meta / fixed_02 | 0.00113605 | 0.00209785 | 0.000961808 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| UPD06 | mixed | gain_meta / fixed_02 | 0.00446481 | 0.00589336 | 0.00142855 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| UPD07 | stationary_noisy | gain_meta / expert_bank | 0.00324448 | 0.00205836 | -0.00118613 | 1 / 0 / 47 | 1 / 1.17e-11 | Ters sonuç |
| UPD08 | drift | gain_meta / expert_bank | 0.000525892 | 0.00127989 | 0.000754002 | 48 / 0 / 0 | 3.41e-13 / 1 | Desteklendi |
| UPD09 | jumps | gain_meta / expert_bank | 0.00304254 | 0.00337151 | 0.000328969 | 46 / 0 / 2 | 2.63e-10 / 1 | Belirsiz |
| UPD10 | mixed | gain_meta / expert_bank | 0.00446481 | 0.0045362 | 7.13929e-05 | 30 / 0 / 18 | 1 / 1 | Belirsiz |
| UPD11 | transfer_matched | frozen_gain_transfer / fixed_02_reset | 0.000540757 | 0.000762919 | 0.000222162 | 48 / 0 / 0 | 3.41e-13 / 1 | Belirsiz |
| UPD12 | transfer_mismatch | frozen_gain_transfer / fixed_02_reset | 0.00289013 | 0.00184652 | -0.00104362 | 0 / 0 / 48 | 1 / 3.41e-13 | Ters sonuç |

## Ölçüt ve kaynak dökümü

| Kimlik | Ölçüt | Pratik eşik | Aday sanal işlem | Kontrol sanal işlem | Keşif Δ |
|---|---|---:|---:|---:|---:|
| ENV-01 | mean_pseudo_regret_512 | 0.01 | 12257 | 7157 | 0.0966797 |
| ENV-02 | mean_pseudo_regret_512 | 0.01 | 8126.48 | 7157 | 0.0647949 |
| ENV-03 | mean_pseudo_regret_512 | 0.01 | 167264 | 12257 | -0.0224121 |
| ENV-04 | mean_pseudo_regret_512 | 0.01 | 12257 | 7157 | 0.0419352 |
| ENV-05 | mean_pseudo_regret_512 | 0.01 | 8126.04 | 7157 | 0.0395833 |
| ENV-06 | mean_pseudo_regret_512 | 0.01 | 167265 | 12257 | -0.0167887 |
| ENV-07 | mean_pseudo_regret_512 | 0.01 | 12257 | 7157 | 0.220068 |
| ENV-08 | mean_pseudo_regret_512 | 0.01 | 8127.25 | 7157 | 0.00698242 |
| ENV-09 | mean_pseudo_regret_512 | 0.01 | 167262 | 12257 | 0.0316406 |
| ENV-10 | mean_pseudo_regret_512 | 0.01 | 12257 | 7157 | 0.156348 |
| ENV-11 | mean_pseudo_regret_512 | 0.01 | 8125.33 | 7157 | -0.00151367 |
| ENV-12 | mean_pseudo_regret_512 | 0.01 | 167264 | 12257 | 0.0703613 |
| MEM01 | whole_life_squared_loss | 0.005 | 22523.3 | 105184 | 0.0528259 |
| MEM02 | whole_life_squared_loss | 0.005 | 10240 | 22523.3 | 0.0668742 |
| MEM03 | whole_life_squared_loss | 0.005 | 10240 | 9216 | 0.232483 |
| MEM04 | whole_life_squared_loss | 0.005 | 18688 | 10240 | 0.12322 |
| MEM05 | whole_life_squared_loss | 0.005 | 75776 | 10240 | 0.244639 |
| MEM06 | whole_life_squared_loss | 0.005 | 18688 | 22523.3 | 0.130981 |
| MEM07 | whole_life_squared_loss | 0.005 | 10240 | 75776 | 0.0316366 |
| MEM08 | whole_life_squared_loss | 0.005 | 18688 | 105184 | 0.0901794 |
| MEM09 | whole_life_squared_loss | 0.005 | 7680 | 4352 | 0.110087 |
| MEM10 | whole_life_squared_loss | 0.005 | 7680 | 4864 | 0.0517171 |
| MEM11 | virtual_operations_per_event | 5.0 | 7680 | 317056 | 604.25 |
| MEM12 | whole_life_squared_loss | 0.005 | 10240 | 18688 | 0.0619812 |
| REP01 | mse | 0.01 | 72270.9 | 40746 | 0.935558 |
| REP02 | mse | 0.01 | 72270.9 | 160662 | -0.123295 |
| REP03 | mse | 0.01 | 69492.3 | 160662 | 0.0141692 |
| REP04 | mse | 0.01 | 72270.9 | 127639 | 0.660749 |
| REP05 | mse | 0.01 | 187951 | 186956 | 0.294233 |
| REP06 | mse | 0.01 | 187951 | 160662 | -0.186456 |
| REP07 | mse | 0.01 | 147062 | 160662 | 0.116525 |
| REP08 | mse | 0.01 | 87045.3 | 40746 | 0.387398 |
| REP09 | mse | 0.01 | 87140 | 160662 | -0.103581 |
| REP10 | mse | 0.01 | 173866 | 160662 | -0.0339606 |
| REP11 | cost | 1000 | 72270.9 | 160662 | 87758.1 |
| REP12 | mse | 0.01 | 72270.9 | 80814.8 | 0.0424113 |
| UPD01 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 1922 | 0.000572248 |
| UPD02 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 1922 | -0.00190969 |
| UPD03 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 1922 | 0.000200794 |
| UPD04 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 1922 | 0.00162126 |
| UPD05 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 1922 | 0.000970483 |
| UPD06 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 1922 | 0.00136171 |
| UPD07 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 47251 | -0.00110838 |
| UPD08 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 47251 | 0.000734876 |
| UPD09 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 47251 | 0.000357002 |
| UPD10 | mean_normalized_latent_prequential_squared_error | 0.0005 | 42652 | 47251 | 4.61186e-05 |
| UPD11 | mean_normalized_latent_prequential_squared_error | 0.0005 | 29726 | 2564 | 0.000252391 |
| UPD12 | mean_normalized_latent_prequential_squared_error | 0.0005 | 29726 | 2564 | -0.000998619 |

Bütün yaşamların bellek/edinim/evre ayrıntıları araştırma klasöründeki discovery ve confirmation JSONL kayıtlarındadır.
Reddedilemeyen bir iddia doğru veya yanlış kabul edilmez; belirsizlik eşdeğerlik kanıtı değildir.
