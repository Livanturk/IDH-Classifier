# Bildiri Deney Planı — 3D Beyin-MRI SimSiam SSL

**Headline katkı.** 3D beyin-MRI üzerinde SimSiam SSL için backbone karşılaştırması + **literatüre
dayalı, çökme-farkında, ETİKETSİZ model-seçim algoritması** (`MODEL_SELECTION.md`).
**Kapsam.** Saf SSL pretraining; IDH sınıflandırma sonraki makale (etiket yok → D3 gelecek-iş).
**Bütçe.** Geniş (7×H100), tam ablasyon. **Değerlendirme.** Yalnızca etiketsiz batarya
(RankMe/PR/uniformity/alignment/z_std + spektrum), çok-seed + bootstrap GA.

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

## Aşama A — Baseline backbone karşılaştırması ✅ (kısmen bitti)

- [x] resnet18 / resnet34 / densenet121, adaptive-clinical, 1 seed → densenet121 kazandı.
- [x] best.pth convergence-gate düzeltmesi (epoch 19) + tazelenmiş spektrum/t-SNE.
- [x] **A.1 configs** hazır: `configs/simsiam_{r18,r34,densenet121}_noucsf.yaml` (noucsf, adaptive-clinical,
      saf SimSiam = Stage-C matched baseline) + `scripts/pretrain_stageA.sh` (3 backbone × 3 seed = 9 iş,
      7 GPU'ya sıralı-kuyruk dağıtımı). Doğrulandı: config'ler yükleniyor, dağıtım doğru, --device çözülüyor.
- [ ] **A.1 runs** (GPU, İKİ NODE): `NODE=ai02 bash scripts/pretrain_stageA.sh` (4 GPU: densenet+r34) ve
      `NODE=ai01 bash scripts/pretrain_stageA.sh` (3 GPU: r18); ikisi bitince `COMPARE=1 bash ...`.
      Node-aware launcher doğrulandı (9 iş, çift atama yok, makespan-optimal 2 dalga).
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

## Review (doldurulacak)
- _Aşamalar tamamlandıkça sonuç özeti + çıkarımlar buraya; lessons.md'ye de yansıt._
