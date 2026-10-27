# Data dictionary — `paddy_dataset.csv`

One row = one district × season × year. Built by `src/build_dataset.py`.

- **Rows:** 2,244 (27 districts/zones, Maha and Yala seasons, 1981–2025)
- **Target:** `yield_total` (kg/ha)
- **Year convention:** `year` is the harvest year. Maha 2021/22 → `year = 2022`.

## Columns

| Column | Unit | Source | Role |
|---|---|---|---|
| `district` | – | DCS | **Feature** (categorical) |
| `season` | Maha / Yala | DCS | **Feature** (categorical) |
| `year` | harvest year | DCS | Used for the train/test split; may be used as a trend feature |
| `season_label` | e.g. "2021/22 Maha" | derived | Display only |
| `yield_total` | kg/ha | DCS | **Target** |
| `yield_major`, `yield_minor`, `yield_rainfed` | kg/ha | DCS | ❌ Leakage — parts of the target |
| `sown_total` | ha | DCS | **Feature** — area planted |
| `sown_major`, `sown_minor`, `sown_rainfed` | ha | DCS | EDA (replaced by the shares below) |
| `sown_share_major`, `sown_share_minor`, `sown_share_rainfed` | 0–1 | derived | **Feature** — irrigation mix |
| `harvested_*`, `harvest_ratio` | ha, 0–1 | DCS | ❌ Leakage — only known after harvest (EDA only) |
| `rain_total_mm` | mm | NASA POWER | **Feature** — total season rainfall |
| `rain_min_month_mm`, `rain_max_month_mm` | mm | NASA POWER | **Feature** — driest and wettest month (drought / flood signal) |
| `rain_cv` | – | NASA POWER | **Feature** — how uneven rainfall was across months |
| `rain_pre_season_mm` | mm | NASA POWER | **Feature** — rainfall in the month before planting (Sep for Maha, Mar for Yala) |
| `temp_mean_c`, `temp_max_c`, `temp_min_c` | °C | NASA POWER | **Feature** |
| `humidity_pct` | % | NASA POWER | **Feature** |
| `solar_mj_m2_day` | MJ/m²/day | NASA POWER | **Feature** (missing before 1984) |
| `wind_m_s` | m/s | NASA POWER | **Feature** |
| `fertilizer_shock` | 0/1 | derived | **Feature** — 1 for the 2021/22 Maha and 2022 Yala seasons |

## Season windows used for the weather features
- **Maha** (harvest year Y): October (Y−1) to February (Y). Pre-season month: September (Y−1).
- **Yala** (year Y): April to August (Y). Pre-season month: March (Y).

## Known data issues (for the report)
1. **Coarse weather grid.** NASA POWER cells are about 0.5° × 0.625° (~55 km), so these districts share identical weather:
   - Colombo / Gampaha
   - Kandy / Matale / Kegalle
   - Ratnapura / Udawalawe
   - Badulla / Monaragala
   - Vavuniya / Mullaitivu

   Rainfall amounts are biased low in very wet areas, and Hambantota reads too wet.
2. **Northern districts have gaps** during the civil war years. Kilinochchi has data for only 35 seasons and Jaffna for 44.
3. **No irrigation split before 1983/84.** The `sown_share_*` columns are missing for those rows (6.3%).
4. **Solar radiation starts in 1984.** It is missing for 1981–83 (5.8%).
5. **2025/26 Maha is excluded**, because the season's weather was not yet complete.
6. In the DCS sheets, a `-` inside an irrigation split means "none of this type" and was set to 0. A completely empty split was kept as missing.
7. **Fertilizer shock is visible in the data.** Area-weighted national yield:
   - Maha fell from 4,505 kg/ha (2020/21) to 2,831 (2021/22).
   - Yala fell from 4,685 (2021) to 3,415 (2022).

## Sources
- Department of Census and Statistics, Sri Lanka — Paddy Statistics: https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics
- NASA POWER (Agroclimatology), monthly point API: https://power.larc.nasa.gov
