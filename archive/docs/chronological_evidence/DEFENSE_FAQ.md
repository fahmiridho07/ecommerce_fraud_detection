# FAQ Sidang — Tugas Akhir IEEE-CIS Fraud Detection

Status: archived chronological FAQ. Superseded by the 2026-06-17 stratified split reset. Do not use this file for defense rehearsal until it is rewritten around `docs/THESIS_SCOPE.md` and `docs/STRATIFIED_SPLIT_RESET.md`.

Active defense preparation must start from `docs/AI_AGENT_BRIEF.md`,
`docs/THESIS_SCOPE.md`, and `docs/BAB3_METHOD_ADJUSTMENT.md`. The AE-05 claims
below are historical chronological evidence, not active post-reset thesis
claims.

Dokumen ringkas pertanyaan–jawaban untuk ujian/sidang. Menggabungkan diskusi metodologi, evaluasi, literatur, dan scope penelitian.

**Hasil proposal (P01–P04):** P02 (tuned LightGBM) — Test PR-AUC **0.5049**; latent replacement (P03/P04) kalah.  
**Hasil extended (post-diagnostic):** **AE-05** (hybrid top-25 `V*` + LD32 latent + reconstruction error) — Test PR-AUC **0.5098** > P02; bootstrap Δ **+0.0049**, 95% CI **[+0.0007, +0.0093]**, p≈0.009.  
**Protokol:** split kronologis, PR-AUC utama, tanpa SMOTE pre-split, fitur asli IEEE-CIS.  
**Artefak:** `initial_proposal_comparison.csv` (P01–P04) · `extended_proposal_comparison.csv` (termasuk AE-05)

---

## A. Metrik & Evaluasi

### Q1. Kenapa pakai PR-AUC? Bukankah itu berarti “lebih akurat” memprediksi fraud?

**Tidak persis.** PR-AUC dipilih karena **lebih tepat menilai deteksi fraud** pada data sangat tidak seimbang (~3,5% fraud), bukan karena angkanya “lebih akurat” dalam arti persentase prediksi benar.

- **Accuracy** menyesatkan: model yang selalu prediksi “normal” bisa ~96,5% accuracy tapi **nol fraud tertangkap**.
- **ROC-AUC** bisa terlihat bagus karena banyak true negative (Saito & Rehmsmeier, 2015).
- **PR-AUC** fokus trade-off **precision vs recall** — relevan untuk early blocking vs false alarm (friction pelanggan vs fraud lolos → chargeback).

PR-AUC = ringkasan kurva di **banyak threshold**. Untuk keputusan operasional tunggal, tetap perlu precision/recall/F1 di threshold terpilih (sudah dilaporkan di registry).

---

### Q2. AP ~0,50 artinya model cuma “50% akurat”?

**Tidak.** Pada imbalance ekstrem, baca PR-AUC dengan **baseline no-skill**:

| Referensi | Test PR-AUC |
|-----------|-------------|
| No-skill (random, prevalence ~3,5%) | ~**0,035** |
| P01 baseline | 0,486 |
| P02 tuned (P01–P04 terbaik) | 0,505 |
| **AE-05 extended terbaik** | **0,510** |

AE-05 ~**14×** di atas tebakan acak. Angka 0,50 bukan accuracy klasifikasi; itu **Average Precision** pada kurva PR.

---

### Q3. Ujung-ujungnya kan model terbaik dilihat dari skor?

**Ya — dalam kontes evaluasi yang sama.** Ada dua blok yang harus dibedakan di sidang:

**Blok proposal (P01–P04, pertanyaan asli):**

| Model | Test PR-AUC | Ranking |
|-------|-------------|---------|
| **P02** | **0.5049** | **1** |
| P01 | 0.4858 | 2 |
| P04 | 0.4845 | 3 |
| P03 | 0.4802 | 4 |

→ Latent **replacement** kalah dari tuned baseline. Itu jawaban proposal.

**Blok extended (post-diagnostic, setelah ablasi):**

| Model | Test PR-AUC | vs P02 |
|-------|-------------|--------|
| **AE-05** | **0.5098** | **+0.0049** |
| P02 | 0.5049 | — |

AE-05 = hybrid (pertahankan top-25 `V*` supervised) + LD32 latent untuk `V*` sisanya + 2 fitur reconstruction error. Protokol split/metrik sama; paired bootstrap mendukung keunggulan vs P02.

**Catatan:** skor antar **paper berbeda** (Moradi 0,89 vs kamu 0,50) **bukan** kontes yang sama — tidak dipakai untuk memilih model terbaik lintas studi.

---

### Q4. Kenapa tidak pakai accuracy / ROC-AUC saja?

- Fraud **minoritas** → accuracy tidak informatif.
- ROC-AUC kurang sensitif terhadap performa kelas positif pada imbalance berat.
- PR-AUC + ROC-AUC/F1/MCC sebagai pelengkap = praktik yang disarankan literatur fraud (Saito 2015, Williams 2021, Dal Pozzolo 2018).

---

## B. Protokol Evaluasi

### Q5. Kenapa pakai protokol berbeda dari paper yang AP-nya tinggi?

**Bukan protokol acak** — disengaja **lebih realistis dan terkontrol** untuk pertanyaan proposal:

| Aspek | Banyak paper tinggi | Penelitian ini | Alasan |
|-------|---------------------|----------------|--------|
| Split | Random/stratified | **Kronologis** (`TransactionDT`) | Early detection & drift (Lucas 2019, Carcillo 2018) |
| Fitur | FE berat (+80 fitur) | **Asli IEEE-CIS** | Isolasi efek AE |
| Resampling | SMOTE/ADASYN | **Tidak pre-split** | Cegah leakage (Kabane 2024) |
| Model | Ensemble stacking | **LightGBM (+ AE)** | Sesuai proposal |
| Tujuan | Maksimalkan skor | **Bandingkan strategi AE vs LGBM** | Pertanyaan penelitian |

**Jawaban 30 detik:**  
> “Kami memakai split kronologis tanpa resampling pre-split dan fitur asli agar perbandingan AE–LightGBM vs LightGBM tidak tercampur FE atau leakage. Protokol ini mengikuti evaluasi fraud realistis (Dal Pozzolo, Lucas, Carcillo) dan peringatan Kabane soal sampling leakage.”

---

### Q6. Apakah sengaja pilih protokol “keras” supaya AP rendah?

**Tidak.** Chronological split biasanya **menurunkan** AP vs stratified — bukan jalan pintas.

- Eksplorasi arsip (FE) pernah AP lebih tinggi → **sengaja tidak jadi klaim utama** karena di luar scope perbandingan terkontrol.
- AP ~0,50 pada protokol ketat tetap jauh di atas no-skill ~0,035.

---

### Q7. Kenapa chronological, bukan stratified split?

**Stratified** (acak, proporsi fraud sama di train/test):

- Mencampur transaksi masa lalu & masa depan → test **terlalu mudah**.
- Menyembunyikan **concept drift** harian.

**Chronological** (train masa lalu → test masa depan):

- Meniru deployment fraud detection sungguhan.
- Estimasi performa **lebih jujur** untuk early detection.

Stratified **terlihat bagus di kertas** ≠ **bagus di production** setelah drift. Sering kebalikannya.

---

### Q8. Kalau stratified AP 0,89 dan kamu 0,50, model kamu kalah telak?

**Belum tentu.** Itu **bukan apple-to-apple**:

- Moradi (2025): AUC-PR 0,891 — FE berat, ensemble, split tidak time-aware → **related work, not comparable**.
- HTGNN: AP ~0,64 — graph temporal, paradigma berbeda.
- Zheng/SilIF unsupervised: AP ~0,09–0,14 — framing berbeda.

Kamu **tidak** klaim juara leaderboard; kamu klaim: **pada protokol ini, latent replacement (P03/P04) < P02; integrasi hybrid AE-05 > P02** setelah ablasi post-proposal.

---

## C. Konteks Bisnis (CNP, Chargeback, Early Detection)

### Q9. Ini kan card-not-present & chargeback — kalau evaluasi “kurang bagus”, manfaatnya apa?

ML dipakai untuk **scoring risiko real-time** saat transaksi, bukan prediksi chargeback 1:1 (label chargeback terlambat — Carcillo 2018).

**Manfaat penelitian:**

1. **Keputusan desain:** latent replacement AE–LightGBM kalah dari tuned baseline; integrasi hybrid + reconstruction error (AE-05) memberi uplift terukur vs P02.
2. **Estimasi deployment jujur:** AP chronological lebih dekat performa masa depan daripada AP stratified+FE.
3. **Kontribusi metodologis:** pipeline missingness-preserving (`v_missing_*`, masked AE loss) terdokumentasi & reproducible.
4. **Literatur:** posisi jelas vs hybrid AE kompleks (Prabha, Alharbi) dan tabular FE (Moradi).

Evaluasi “ketat” + skor moderat > evaluasi optimistis + skor tinggi yang **drop** di production.

---

### Q10. PR-AUC sudah cukup untuk early detection?

PR-AUC = **langkah 1:** pilih model terbaik antar kandidat (→ AE-05 extended; P02 untuk blok proposal P01–P04).

**Langkah 2 (operasional):** precision/recall/F1 di threshold bisnis (“minimal recall X%, precision tidak di bawah Y”).

Skripsi fokus langkah 1; diskusi bisa menyebut langkah 2 sebagai implementasi lanjutan.

---

## D. Desain Penelitian & Scope

### Q11. Kenapa tidak feature engineering di semua eksperimen?

FE universal mengubah pertanyaan penelitian:

- **Proposal:** AE–LightGBM vs LightGBM pada fitur asli.
- **Dengan FE tetap:** “dengan preprocessing kaya, apakah AE membantu?” — pertanyaan lain, butuh rerun P05–P08 & revisi proposal.

Arsip internal: FE tuned AP ~0,51–0,53 > P02 → baseline kaya fitur memang kuat; AE sulit menang. Itu **mendukung** narasi “AE replacement kehilangan sinyal `V*` yang LGBM eksploitasi natively”.

**Untuk sidang:** FE disebut di diskusi/limitasi + related work (Moradi), bukan klaim utama.

---

### Q12. Kenapa AE–LightGBM kalah dari baseline? Apakah penelitian gagal?

**Tidak gagal** — hasilnya berlapis dan defensible:

**Proposal (P01–P04):** hipotesis latent **replacement** ditolak — P03/P04 < P02.

- Replacement menghilangkan nilai `V*` asli yang informatif untuk LGBM.
- LD128 rekonstruksi lebih baik, tapi P04 tetap < P02.

**Post-diagnostic (AE-03 → AE-05):** setelah mempertahankan top-25 `V*` + menambah reconstruction error, **AE-05 mengalahkan P02** (test AP 0,5098, bootstrap CI positif). Kesimpulan: AE membantu bila **augmentasi/hybrid**, bukan replacement penuh.

**Jawaban 30 detik:**  
> “Pada desain proposal, replacement kalah. Setelah ablasi, hybrid + anomaly score AE menang — kami dokumentasikan kedua blok secara eksplisit.”

---

### Q13. Apakah penelitian ini outdated vs riset 2025–2026?

**Tidak outdated** pada aspek kritis:

| Aspek | Status |
|-------|--------|
| PR-AUC utama | Selaras tren (masih jarang dilaporkan eksplisit) |
| Chronological split | **Lebih ketat** dari banyak paper AE-IEEE-CIS (stratified) |
| LightGBM + Optuna | Masih mainstream tabular |
| AE hybrid kompleks (GAN, graph) | Di luar scope — sengaja |

Kamu di niche: **evaluasi realistis + perbandingan terkontrol AE vs LGBM**, bukan chase leaderboard Moradi.

---

### Q14. Kenapa tidak bandingkan langsung dengan Moradi / Prabha / HTGNN?

Karena **protokol tidak comparable** (split, FE, resampling, model, subset data). Mereka masuk **Bab 2 related work** + tabel comparability di **Bab 5**, bukan pembuktian “model kami lebih baik”.

---

## E. Organisasi Proyek (dari audit workspace)

### Q15. Di mana file penting?

| Kebutuhan | Lokasi |
|-----------|--------|
| Scope & hasil | `docs/THESIS_SCOPE.md`, `docs/EXPERIMENT_REGISTRY.md`, `archive/docs/chronological_evidence/DEFENSE_FAQ.md` |
| Extended comparison | `outputs/initial_proposal/final_comparison/extended_proposal_comparison.csv` |
| Bootstrap AE-05 vs P02 | `outputs/initial_proposal/representation_ablation/bootstrap_ae05_vs_p02/` |
| Rerun P01–P04 | `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md` |
| Artefak canonical | `outputs/initial_proposal/` |
| Proposal/skripsi | `0. Skripsi/proposal/`, `0. Skripsi/skripsi/` |
| PDF referensi | `2. Reference/` |
| FAQ ini | `archive/docs/chronological_evidence/DEFENSE_FAQ.md` |

---

## F. Cheat Sheet — Satu Kalimat per Topik

| Topik | Jawaban satu kalimat |
|-------|----------------------|
| PR-AUC | Metrik utama karena informatif untuk imbalance fraud, bukan karena “accuracy 50%”. |
| Model terbaik | AE-05 extended (0,510); P02 terbaik blok proposal P01–P04 (0,505). |
| AP vs paper lain | Beda protokol → tidak comparable; Moradi = konteks, bukan leaderboard. |
| Chronological split | Simulasi latih masa lalu / uji masa depan — realistis untuk drift. |
| Stratified tinggi | Sering optimistis; production bisa jauh di bawah angka kertas. |
| Protokol berbeda | Sesuai pertanyaan proposal + literatur evaluasi realistis, bukan hindari perbandingan. |
| FE tidak dipakai | Agar isolasi efek AE; FE = diskusi/future work. |
| AE replacement | Temuan valid: kalah dari P02 pada P01–P04. |
| AE hybrid (AE-05) | Mengalahkan P02; reconstruction error = fitur gain #1–2. |
| Manfaat | Jawaban hipotesis + kandidat deploy (AE-05) + pipeline reproducible. |

---

## G. Flowchart Mental Sidang

```
Penguji: "AP rendah / protokol beda / kenapa tidak FE?"
    │
    ├─► Bukan accuracy 50% → banding no-skill ~0.035
    │
    ├─► Dua blok: P02 menang proposal; AE-05 menang extended (kontes sama)
    │
    ├─► Paper 0.89 = protokol lain → related work only
    │
    ├─► Chronological = lebih keras & realistis (bukan malas)
    │
    └─► Manfaat = jawaban hipotesis AE vs LGBM dengan validitas tinggi
```

---

## H. AE-05 — Pertanyaan Sidang Khusus

### Q16. Apa itu AE-05 dan kenapa tidak mengganti P01–P04?

AE-05 adalah kandidat **post-diagnostic**, bukan rerun proposal asli:

| Komponen | Isi |
|----------|-----|
| Supervised `V*` | Top-25 gain (dari P01 importance) dipertahankan mentah |
| AE latent | LD32 untuk `V*` sisanya (bukan replacement penuh) |
| Anomaly score | `v_ae_reconstruction_mse`, `v_ae_reconstruction_log1p_mse` |
| Classifier | LightGBM (params dari AE-04 hybrid tuned) |

P01–P04 tetap sebagai **blok historis** yang menjawab pertanyaan proposal (“replacement vs baseline”). AE-05 menjawab pertanyaan lanjutan setelah diagnostic menunjukkan 33% gain dari `V*` asli.

### Q17. Apakah peningkatan +0,005 signifikan?

Paired bootstrap 1000× pada baris test yang sama:

| Metrik | Nilai |
|--------|-------|
| Δ test PR-AUC | +0.004921 |
| 95% CI | [+0.000650, +0.009316] |
| One-sided p(Δ≤0) | ≈ 0.009 |

Interval tidak mencakup nol → uplift kecil tapi **konsisten** pada split kronologis yang sama. Frame sebagai peningkatan incremental, bukan lompatan dramatis.

### Q18. Kenapa ROC-AUC AE-05 sedikit lebih rendah?

AE-05 test ROC-AUC 0,882 vs P02 0,883 (−0,0014). PR-AUC dan F1/MCC naik. Untuk fraud imbalance, **PR-AUC lebih relevan** untuk early blocking; ROC-AUC dipakai sebagai pelengkap, bukan pemilih utama.

### Q19. Apakah ini “curang” karena pakai params AE-04?

AE-05 run dengan `--n_trials 0` memakai hyperparameter AE-04 hybrid tuned, lalu menambah 2 fitur recon. Itu **fair comparison** karena baseline P02 juga sudah di-Optuna. Run Optuna independen AE-05 (15 trials) tersedia di `outputs/initial_proposal/optuna/ae_lgbm_ld32_top25v_recon_tuned/` untuk verifikasi tambahan.

---

*Terakhir disusun: 2026-06-16. Sesuaikan angka jika rerun registry berubah.*
