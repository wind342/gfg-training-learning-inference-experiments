# Functional step-size smoke: actual adjudication

Exploratory historical-data reuse; not fresh prospective confirmation.

Population: 12 runs, 72 sections, 15264 targets.

Selected amplitudes (sections): `{'0.25': 36, '0.5': 10, '0.75': 9, '1.0': 17}`.

| Method | Correct / 15264 | Correct % | Mean margin | Median margin | Delta vs native (pp) |
|---|---:|---:|---:|---:|---:|
| fixed_0.25 | 11084 | 72.615304 | 4.36092452 | 4.55917549 | +12.427935 |
| fixed_0.5 | 10196 | 66.797694 | 3.16287127 | 2.67931628 | +6.610325 |
| fixed_0.75 | 9576 | 62.735849 | 2.42666317 | 1.89360237 | +2.548480 |
| fixed_1.0 | 9187 | 60.187369 | 1.90540126 | 1.47496605 | +0.000000 |
| controller | 11419 | 74.810010 | 4.46524139 | 4.60123205 | +14.622642 |
| oracle | 11458 | 75.065514 | 4.47461351 | 4.57103753 | +14.878145 |
| alpha0_reference | 11426 | 74.855870 | 5.08725153 | 7.00601530 | +14.668501 |

Controller vs native: prevented correct-to-wrong 2295; additional wrong-to-correct 162; newly caused correct-to-wrong 2; lost wrong-to-correct 223.

Best fixed candidate (hindsight): 0.25. Controller minus best fixed: +335 targets.
Section-level oracle minus controller: 39 targets (0.255503 pp).

## Run-level comparisons

| Run | Controller correct | Native correct | Delta | Delta pp |
|---|---:|---:|---:|---:|
| entry-481b86f81d58d496a687 | 991 | 733 | +258 | +20.283019 |
| entry-4ed462761347d6b87e61 | 978 | 782 | +196 | +15.408805 |
| entry-54ba9566f731754a0e3f | 942 | 797 | +145 | +11.399371 |
| entry-5bb1186bc27eb82111fb | 784 | 452 | +332 | +26.100629 |
| entry-786d0a3628f6f791399f | 955 | 808 | +147 | +11.556604 |
| entry-7a22f51938059541de98 | 1154 | 1014 | +140 | +11.006289 |
| entry-7c1a3094f8acf9cf4bb0 | 932 | 709 | +223 | +17.531447 |
| entry-8fa6576fc7128f93a228 | 950 | 823 | +127 | +9.984277 |
| entry-abc8f864a49ee0e056f4 | 1037 | 904 | +133 | +10.455975 |
| entry-d5b80ca9b9cd18fa343f | 815 | 569 | +246 | +19.339623 |
| entry-d8b1bf9cd00ddf314725 | 870 | 719 | +151 | +11.871069 |
| entry-f34e7e61444c90976b36 | 1011 | 877 | +134 | +10.534591 |

| Against fixed alpha | Run wins | Ties | Losses | Mean run delta pp |
|---|---:|---:|---:|---:|
| 0.25 | 12 | 0 | 0 | +2.194706 |
| 0.5 | 12 | 0 | 0 | +8.012317 |
| 0.75 | 12 | 0 | 0 | +12.074161 |
| 1.0 | 12 | 0 | 0 | +14.622642 |

## Boundaries

Alpha 0 is not a candidate. The oracle uses one shared amplitude per section, not one per target.
Targets are pooled descriptions; runs are the outer units. The original 8/4 split is descriptive only.
This tests one-step counterfactual outcomes along already formed updates, not an optimizer or a sequential controller.
No claim of superior long-term training, generalization, support organization, or other architectures is justified.
Full fixed-alpha, transition, split, source-hash and section results are retained in RESULTS.json and the ledgers.
No candidate or selector rule was changed after outcome access.
