# Bildiri Deney Planı — 3D Beyin-MRI SimSiam SSL

**Headline katkı (KİLİTLİ 2026-07-23).** *Yöntem birincil* → **çöküş-farkında, yakınsama-kapılı,
"kararlı-nokta" (final-plato) okuyan ETİKETSİZ model-seçim algoritması** (`MODEL_SELECTION.md`);
naif seçimin (yakınsama-piki RankMe) kararsızlığı (seed-CV ≤%24) ana bulgu/ablasyon. Backbone
karşılaştırması (densenet121>r18>r34) yöntemin *demonstrasyonu*.
**Kapsam.** Saf SSL pretraining; IDH sınıflandırma sonraki makale (etiket yok → D3 gelecek-iş).
**Format.** Tam konferans makalesi (8+ sayfa). **Bütçe.** Geniş (7×H100), tam ablasyon.
**Değerlendirme.** Yalnızca etiketsiz batarya (RankMe birincil + LiDAR doğrulayıcı /PR/uniformity/
alignment/z_std + spektrum), çok-seed + **jackknife** GA (bootstrap DEĞİL, L9).

## 🔒 KAPSAM KİLİTLENDİ (2026-07-23) — kritik yol
Kararlar: **manşet=yöntem**, **format=tam makale**, opsiyonellerin **hepsi dahil** (LiDAR + instabilite
ablasyonu Fig 3 + site-level jackknife).
- **Kritik yol (bel kemiği):** A) 9 unified koşu bitsin (r34 s43/s44 dahil) → B) GPU COMPARE→final-plato
  leaderboard+seed-CI → C) Fig1 yörünge + D) Fig2 backbone bar (İng/600dpi).
- **Zorunlu (seçildi):** E) Fig3 instabilite ablasyonu (R1 pik CI-örtüşür ↔ final CI-ayrışır);
  LiDAR final-checkpoint doğrulayıcı; site-level (delete-one-cohort) block jackknife robustness.
- **Prose senkron:** PAPER_* taslakları eski D1 çerçevesinden final-plato/R1-ablasyon çerçevesine.
- **GPU beklerken yapılabilir (unblocked):** LiDAR kodu, site-jackknife kodu, Related Work + Method
  yazımı, figür şablonları (İng/600dpi), dataset bölümü.
- **İkincil (Aşama C/D):** mutlak-kalite (VICReg/W-MSE/bighead/longaug) ve recipe/crop ablasyonları —
  tam makalede yer varsa; kritik yolu bloklamaz.

> **Bilimsel gereklilik.** Etiket olmadığı için AUC ile doğrulayamıyoruz; dolayısıyla bildirinin
> gücü tamamen **(a) seçim metodolojisinin savunulabilirliğine** ve **(b) metriklerin istatistiksel
> sağlamlığına** dayanır. Her iddia çok-seed + GA ile desteklenmeli.

---

## Aşama 0 — Altyapı ✅ TAMAMLANDI (tüm A0.x + smoke exit 0)

- [x] **A0.1** VICReg-tarzı VC-regülarizatör: `models.vc_regularizer` (var+cov, ayrı döner);
      `train_simsiam` opsiyonel `+w_var·var +w_cov·cov` (config/CLI `vicreg_var/cov/gamma`; kapalıyken
      DAVRANIŞ DEĞİŞMEZ, var/cov loglanır). Config: `configs/simsiam_densenet121_vicreg.yaml` (noucsf).
      Doğrulandı: birim test (çökmüş→var yüksek, sağlıklı→var 0), smoke train çalışıyor. **Uyarı (L10):**
      ham ağırlıklar kalibre değil (cov invaryansı boğabiliyor) → Stage C.2 λ-sweep zorunlu.
- [x] **A0.2** W-MSE whitening kaybı: `models.whitening_mse_loss` (Cholesky ZCA, batch→~birim kovaryans,
      anti-collapse yapısal); `train_simsiam` opsiyonel `+w_wmse·wmse` (config/CLI `wmse_weight/eps`;
      kapalıyken davranış değişmez, loglanır). Config: `configs/simsiam_densenet121_wmse.yaml` (proj_dim=8).
      **Kısıt (L11):** whitening için batch>proj_dim gerekir → batch 16'da küçük projektör zorunlu.
      Doğrulandı: whitened cov~I, özdeş view→wmse=-1, smoke train çalışıyor.
- [x] **A0.3** Büyük projector/predictor: `configs/simsiam_densenet121_bighead.yaml` (proj 512/2048/512,
      feature_dim=256 KİLİTLİ). Saf config. Doğrulandı: model kuruluyor, feature_dim=256, 17.9M param.
- [x] **A0.4** Uzun-eğitim + güçlü-aug: `configs/simsiam_densenet121_longaug.yaml` (400 ep, aug_strength 2.0).
      Küçük kod kancası gerekti (config-driven olması için): `data.ClinicalTransform(strength=)` +
      make_views'a `aug_strength` (varsayılan 1.0 → davranış değişmez) + `--aug_strength` CLI. Doğrulandı:
      strength 1.0 aynı / 2.0 magnitüdler iki katı; clinical veri yolu gerçek denekte çalışıyor, maske korunuyor.
- [x] **NOT çözüldü:** noucsf baseline config'leri oluşturuldu (yukarıda A.1 configs); eski `_pretrain`
      config'leri (ikisi de hariç split) artık kullanılmıyor — paper `_noucsf` config'lerini kullanıyor.
- [x] **A0.5** Geniş etiketsiz eval kohortu: paper split `splits/splits_pretrain_noucsf.json`
      (train=889, val=99). UPENN dahil, UCSF hariç (bkz. lessons L8). val=99 → paper-grade RankMe CI.
- [x] **A0.6** `evaluate.py`'ye **jackknife %95 GA** (RankMe/PR/uniformity); leaderboard'a `*_ci_lo/hi`
      sütunları. `utils.jackknife_ci`. **Bootstrap DEĞİL** — naive bootstrap tekrarlı satırlarla rank'ı
      aşağı saptırır (L9); jackknife leave-one-out + deterministik. Selector CI-örtüşme kuralına
      otomatik geçiyor. Doğrulandı: nokta-tahmin içeriliyor, uçtan uca sütunlar yazılıyor, tie rule çalışıyor.
- [x] **A0.7** **D2 run-arası seçici CLI** (`select_model.py`): leaderboard'u okur, MODEL_SELECTION
      §3 D2'yi deterministik uygular (uygunluk→RankMe→δ-berabere→PR/U→doğrulayıcı), kazananı raporlar.
      Gerçek leaderboard'da doğrulandı → densenet e19 net kazanan (margin +2.79); eğitilmemiş e4 ve
      çökmüş run doğru reddedildi. CI-tabanlı δ, A0.6 gelince otomatik devreye girer (rankme_ci_lo/hi
      sütunları). **Bulgu:** eval-protokol alignment eşiği ayrı olmalı (L7).
- [x] **A0.8** `scripts/smoke_test.sh` genişletildi: 2-3. adım defaults (reg KAPALI, davranış değişmemiş),
      4. adım reg-add-ons AÇIK (VICReg+W-MSE+clinical aug_strength). Tam koşu **exit 0** — regresyon yok.
      Yan-düzeltme: `--aug_preset` CLI'sine "clinical" eklendi (YAML'da vardı, CLI'da eksikti — tutarlılık).

## ⭐ PLAN REVİZE (2026-07-23, L14) — TEK KOHORT, TÜM VERİ, YENİDEN KOŞU
Veri seti değişti: harici UCSF (501) + UPENN (610) + BraTS-diğerleri (585) = **1696**. noucsf/withucsf ayrımı
KALKTI. Önceki 18 koşu (noucsf/withucsf) artık geçersiz (yeni veriyle değil).
- [x] **Stage 0.yeni:** `scripts/build_unified_dataset.py` → `data_unified/` (1696 symlink) ✓;
      tek split `splits/splits_pretrain_all.json` (train=1526/val=170) ✓; `configs/simsiam_{r18,r34,densenet121}_unified.yaml` ✓;
      launcher VARIANT=unified default ✓; harici-veri smoke (tümör-crop+auto-seg+8-bit) hatasız ✓.
- [ ] **Stage A.yeni RUN (GPU):** `NODE=ai02 bash scripts/pretrain_stageA.sh` + `NODE=ai01 bash ...`
      (VARIANT=unified default) → `COMPARE=1 bash ...`. 3 backbone × ≥3 seed = 9 koşu. Karşılaştırma = final plato (L13).
- [x] **LiDAR + α-ReQ implement edildi** (`utils.lidar`, `utils.alpha_req`; `evaluate.py` leaderboard'a
      `lidar`/`alpha_req`(+lidar CI) sütunları; `select_model._NUM` genişletildi). Convergent-validity için.
      Sentetik doğrulama: LiDAR spread↔nuisance-heavy'i ayırıyor (47↔35), RankMe ayıramıyor (59↔59) → knee
      argümanının kanıtı. Kalan: 18 koşuda (ve unified'da) leaderboard'ı üretip üç metriğin uzlaşmasını raporla.
- [ ] **Figürler İngilizce + 600 dpi:** trajektori, backbone bar (RankMe+LiDAR±CI), spektrum, t-SNE (batch-effect + 8/16-bit).

## Aşama A — (ESKİ, tek-seed + eski veri; referans olarak duruyor)

- [x] resnet18 / resnet34 / densenet121, adaptive-clinical, 1 seed → densenet121 kazandı.
- [x] best.pth convergence-gate düzeltmesi (epoch 19) + tazelenmiş spektrum/t-SNE.
- [x] **A.1 configs** hazır: `configs/simsiam_{r18,r34,densenet121}_noucsf.yaml` (noucsf, adaptive-clinical,
      saf SimSiam = Stage-C matched baseline) + `scripts/pretrain_stageA.sh` (3 backbone × 3 seed = 9 iş,
      7 GPU'ya sıralı-kuyruk dağıtımı). Doğrulandı: config'ler yükleniyor, dağıtım doğru, --device çözülüyor.
- [x] **A.1 runs BİTTİ** (noucsf ×9 + withucsf ×9 = 18 koşu, hepsi yakınsadı, collapse yok).
- [x] **A.2 analiz BİTTİ** (metrics.csv'den, GPU'suz): **densenet121 > r18 > r34** (final epoch, CI-anlamlı,
      her iki kohortta). r34 EN KÖTÜ, r18 sağlam ikinci (tek-seed sonucunu düzeltti). +UCSF null (bkz. L13).
      **KARAR:** karşılaştırma = final/plato checkpoint; R1 (converged-maxRankMe) oynak → ablasyon. Kod/doc
      güncellendi (select_model --select latest, MODEL_SELECTION §D1/D2, launcher COMPARE→last.pth).
      Figür: `figures/rankme_trajectory.png`. (Resmi leaderboard/GA için GPU'da `COMPARE=1 ... last.pth`.)
- [x] **A.2 otomasyon** hazır: launcher run bitince otomatik evaluate (GA'lı leaderboard + t-SNE + spektrum)
      → `select_model --group_by backbone --mlflow` → **RankMe ± seed-arası %95 CI bar grafiği** + D2 kazanan,
      hepsi DagsHub'a. select_model artık seed'leri backbone bazında birleştiriyor (mean ± across-seed CI;
      tek seed'de jackknife CI'ya düşer), karşılaştırma figürü üretiyor, MLflow'a basıyor. Doğrulandı (figür).
- [ ] **A.2 runs**: gerçek 9 koşu bitince tablo/figür otomatik üretilecek (launcher içinde).

## Aşama B — Seçim algoritması validasyonu (METODOLOJİ KATKISI, figür)

- [ ] **B.1** Ablasyon: convergence-gate **açık vs kapalı** → gate kapalıyken best.pth epoch 4'e
      (eğitilmemiş) düşer; açıkken epoch 19. Bu, algoritmanın gerekliliğini gösteren ana figür.
- [ ] **B.2** Kararlılık: seçilen epoch + backbone sıralaması seed'ler arası tutarlı mı? (GA ile)
- [ ] **B.3** RankMe yörüngesi figürü: dimensional collapse'ın erken oluşup stabilize olması,
      epoch-19 "diz" seçimi. [Jing2022]

## Aşama C — Mutlak kalite / collapse-azaltma ablasyonu (densenet121)

En iyi backbone üzerinde, hepsi çok-seed + GA. Ölçüt: **RankMe yörüngesi çökmüyor** + uniformity.

- [ ] **C.1** SimSiam baseline (referans).
- [ ] **C.2** + VICReg VC-reg (λ taraması: küçük grid). [Bardes2022]
- [ ] **C.3** + Whitening / W-MSE. [Ermolov2021]
- [ ] **C.4** Büyük projector/predictor.
- [ ] **C.5** Uzun eğitim + güçlü aug.
- [ ] **C.6** Karşılaştırma tablosu + "hangi müdahale effective rank'ı en çok korudu" figürü.

## Aşama D — Recipe & veri-kohort ablasyonu

- [ ] **D.1** densenet121: `crop_mode=tumor` (adaptive-clinical) **vs** `crop_mode=brain`. Recipe'yi
      gerekçelendirir. (CLAUDE.md ablasyon merdiveni ile uyumlu.)
- [ ] **D.2** (ops.) ROI 128³ vs 96³/112³ — hesap/kalite ödünleşmesi (adaptive_crop kapsama tablosu).
- [x] **D.3 veri-kohort HAZIR:** split `splits_pretrain_withucsf.json` (train=1152 = noucsf 889 + UCSF 263,
      val=99 noucsf ile BİREBİR aynı). **3 backbone** config'i: `simsiam_{r18,r34,densenet121}_withucsf.yaml`.
      Launcher `VARIANT` ile genelleştirildi (noucsf/withucsf aynı iki-node mantığı). Doğrulandı.
- [ ] **D.3 runs:** `NODE=ai02 VARIANT=withucsf bash ...` + `NODE=ai01 VARIANT=withucsf bash ...` →
      `COMPARE=1 VARIANT=withucsf bash ...`. Sonra noucsf vs withucsf'i karşılaştır (aynı val=99 → "UCSF
      eklemek RankMe'yi artırıyor mu?"). Çapraz-makale: withucsf encoder harici-test için kullanılamaz (L12).

## Aşama E — Yazım / gerekçelendirme

- [ ] **E.1** `DESIGN_JUSTIFICATION.md`'yi bu ablasyonlarla güncelle (her eksen: Karar→Gerekçe→Alt→Kaynak).
- [ ] **E.2** Metrik açıklama bölümü (RankMe/alignment/uniformity/PR/spektrum/t-SNE — daha önce hazırlandı).
- [ ] **E.3** Literatür karşılaştırması: RankMe [Garrido2023], collapse [Jing2022], VICReg [Bardes2022],
      alignment/uniformity [WangIsola2020], eleştiri [Otero2024], LiDAR [Thilak2024].
- [ ] **E.4** Bütçe muhasebesi: `scripts/time_epochs.sh` ile epoch süresi → toplam GPU-saat tahmini.

---

## Açık kararlar / notlar
- Etiket yok → proxy'yi OUR-data'da doğrulamak için opsiyonel **surrogate serbest-etiket probe**
  (ör. glioma grade / site-kontrollü) düşünülebilir; batch-effect riski nedeniyle dikkatli. (Tartışılacak.)
- Tam Kartezyen grid (3×5×2×≥3) çok pahalı → **referans-etrafı OFAT + seçili etkileşim** (yukarıdaki
  aşamalar bunu uyguluyor), tam grid değil.

## Review

### 2026-07-26 — UCSF çıkarımı + çok-metrikli checkpoint seçimi (kullanıcı isteği)
- **UCSF pretraining'den TAMAMEN çıkarıldı** (encoder eğitimi/validation/checkpoint seçimi/hiperparametre/
  erken durdurma). UCSF = harici, tek-seferlik downstream doğrulama. Aktif split
  `splits/splits_pretrain_noucsf_all.json` (data_unified − UCSF; 1135 train / 60 val, UPENN içeride).
- **Validation her epoch** (`val_every: 1`); her epoch **RankMe + LiDAR + α-ReQ** + loss/LR/erken-durdurma
  `metrics.csv`'e loglanıyor.
- **Üç best checkpoint**: `bestRankMe.pth` (max RankMe), `bestLiDAR.pth` (max LiDAR), `bestA-ReQ.pth`
  (min|α−1|); ortak yakınsama+çöküş kapısı; her checkpoint'te `selection` provenans; `best.pth`=alias.
- **Erken durdurma = üçü de plato** (union, metrik-başına sayaç).
- Yeni: `braintumor_ssl/report_run.py` (run-başına 6-maddelik rapor + `results/checkpoint_selection_summary.csv`),
  `braintumor_ssl/downstream_ablation.py` (3-checkpoint transfer iskelesi; UCSF yalnız `--external` final),
  `braintumor_ssl/CHECKPOINT_SELECTION_METHODOLOGY.md`, `scripts/pretrain_unified_noucsf.sh` (backbone×seed).
- Genellenebilir: DenseNet121/seed42 yalnız örnek; cfg checkpoint'ten okunur. Bkz. lessons **L17**.
- _Sonraki aşamalar tamamlandıkça sonuç özeti + çıkarımlar buraya; lessons.md'ye de yansıt._
