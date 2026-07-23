# Model İzleme, Durdurma ve Seçim Algoritması

Bu belge, SSL pretraining'de **hangi loss/schedule ile eğitildiğini**, **eğitimin ne zaman
durdurulduğunu** ve **etiketsiz metriklerden (RankMe, alignment, uniformity, participation
ratio, z_std) en iyi modelin nasıl deterministik biçimde seçildiğini** tanımlar. Amaç: seçimin
"gözle bakıp karar verdik" değil, **her adımı literatüre dayalı, tekrarlanabilir bir algoritma**
olarak bildiride savunulabilmesi.

Tasarım ilkesi: metrikleri **ağırlıklı toplamla** birleştirmiyoruz (ağırlıklar keyfi olur ve
etiketsiz ortamda savunulamaz). Bunun yerine **katmanlı/leksikografik (gated) bir karar
prosedürü** kullanıyoruz: bazı kriterler *zorunlu kapı*dır (çökme yok, yakınsama var) — yüksek
rank ile "geri satın alınamaz"; ancak tüm kapılardan geçen adaylar arasında, en iyi
gerekçelendirilmiş tek proxy (RankMe) ile sıralama yapılır. Bu, RankMe'nin dengesiz görevlerde
tek-başına-vekil olarak başarısız olabileceği eleştirisini [Otero2024] doğrudan karşılar:
RankMe yalnızca *geçerli bir rejim içinde* (çökmemiş + yakınsamış) ve *üçgenlenmiş* olarak
kullanılır, koşulsuz değil.

---

## 0. Metrik bataryası (tanımlar)

Tümü, sabit bir **etiketsiz** izleme kohortu üzerinde, precise-BN uyarlaması sonrası hesaplanır
(SimSiam'ın L2-normalize kaybı BN istatistiklerini serbest bıraktığından; bkz. `utils.recompute_bn_stats`).

| Sembol | Metrik | Ölçtüğü | İyi yön | Kaynak |
|---|---|---|---|---|
| `z_std` | representation std | çökme (tam) | ≈ 1/√d | [ChenHe2021] |
| `L_val` | SimSiam neg-kosinüs val loss | invaryans / yakınsama | → −1 | [ChenHe2021] |
| `A` | alignment | pozitif çift yakınlığı | düşük | [WangIsola2020] |
| `RankMe` | spektral effective rank | efektif boyut (temsil zenginliği) | yüksek | [Garrido2023] |
| `PR` | participation ratio | efektif boyut (ikinci ölçü) | yüksek | — |
| `U` | uniformity | küre üzerinde bilgi korunumu | düşük/negatif | [WangIsola2020] |

Eşikler (config anahtarları): `z_std_min=0.01`, `best_val_loss_max=−0.8`, `best_align_max=0.15`.
Gerekçeleri Bölüm 4'te.

---

## 1. Eğitim algoritması (loss + schedule)

**Karar.** Kayıp = simetrik negatif-kosinüs benzerliği, stop-gradient + predictor asimetrisi ile
[ChenHe2021]. Optimizasyon: SGD, **lineer LR ölçekleme** (`lr = base_lr × batch/256`) [Goyal2017],
**cosine decay + warmup** [Loshchilov2017], predictor LR sabit (`fix_pred_lr`) [ChenHe2021], AMP.

```
L(view1, view2) = ½ · D(p1, stopgrad(z2)) + ½ · D(p2, stopgrad(z1))
   D(p, z) = − (p/‖p‖) · (z/‖z‖)            # negatif kosinüs, aralık [−1, 0]
```

**Gerekçe.** Bölüm 1/DESIGN_JUSTIFICATION'da: negatif çift/momentum encoder gerektirmez, küçük
batch'te (3D bellek kısıtı) stabildir. Loss'un kendisi seçim metriği DEĞİLDİR — configler arası
karşılaştırılamaz; yalnızca **yakınsama sinyali** olarak kullanılır (bkz. Bölüm 2-3).

---

## 2. Eğitim-içi izleme ve durdurma algoritması

Her `val_every` epoch'ta izleme kohortu üzerinde tüm metrikler hesaplanır ve `metrics.csv`'ye
yazılır. İki opsiyonel durdurma kuralı — **ikisi de loss yörüngesini ve `last.pth`'yi asla
değiştirmez** (yalnızca eğitimi erken bitirir):

```
her değerlendirme epoch'u e için:
  m ← validate(model, izleme_kohortu)                    # precise-BN sonrası
  collapsed ← ¬finite(m) ∨ (m.z_std < z_std_min)         # [ChenHe2021, Jing2022]
  converged ← (m.L_val ≤ best_val_loss_max) ∧ (m.A ≤ best_align_max)   # [WangIsola2020]

  # (a) COLLAPSE-STOP: art arda çökme → run'ı iptal et
  collapse_run ← collapse_run+1 if collapsed else 0
  if collapse_stop ∧ collapse_run ≥ collapse_patience: STOP("collapsed")

  # (b) best.pth güncelle (Bölüm 3, D1)
  eligible ← converged ∧ ¬collapsed ∧ finite(m.RankMe)
  if eligible ∧ (m.RankMe > best_RankMe + min_delta_rankme):
      best_RankMe ← m.RankMe ; best_epoch ← e ; no_improve ← 0 ; save best.pth
  elif best_epoch ≥ 0:  no_improve ← no_improve + 1        # plato sayacı YALNIZCA yakınsama sonrası

  # (c) EARLY-STOP: yakınsamış bir taban varken RankMe platoya girdi
  if early_stop ∧ (e+1 ≥ min_epochs) ∧ (no_improve ≥ patience): STOP("rankme plateau")
```

**Kritik tasarım kararı (profesör sorusunun cevabı).** Plato sayacı (`no_improve`) **yalnızca ilk
yakınsamış best bulunduktan sonra** artar. Böylece "henüz öğrenmedi" durumu (erken epoch'lar)
"gelişme durdu" ile karıştırılmaz — erken-durdurma, *yakınsama sonrası RankMe platosu* demektir,
"eğitim başlamadı" değil.

---

## 3. Seçim algoritması (üç kademe)

### D1 — Run-içi checkpoint seçimi (hangi checkpoint karşılaştırmaya girer)

**KARŞILAŞTIRMA TEMELİ = final (plato) checkpoint (last.pth).** Çok-seed analizi (18 koşu, L13)
şunu gösterdi: "yakınsamışlar arasında max-RankMe" kuralı (aşağıdaki *R1*) her zaman **ilk-yakınsayan
epoch**'u seçer, çünkü RankMe yakınsamada zirve yapıp düşer. O zirve dik bir geçiştir; her-5-epoch
ızgarasıyla, seed'e göre kayan bir yakınsama anında örneklendiğinden değeri **oynaktır** (CV ~%15–24)
ve backbone'ları ayıramaz. Final plato checkpoint'i ise **kararlı** (CV ~%8), ızgaradan ve
seed-zamanlamasından bağımsız, tekrarlanabilir → adil karşılaştırmanın temeli budur.

```
# KARŞILAŞTIRMA (önerilen): final/plato checkpoint
compare_ckpt = last.pth   (RankMe-platosu erken-durdurmasının bittiği, tam yakınsamış nokta)

# R1 (ABLASYON — kullanma; kararsızlığını GÖSTERMEK için):
eligible(e) ⟺ finite(m_e) ∧ z_std_e ≥ z_std_min ∧ L_val_e ≤ best_val_loss_max ∧ A_e ≤ best_align_max
best.pth_R1 = argmax_{e : eligible(e)} RankMe_e   # ≡ ilk-yakınsayan epoch → oynak (L13)
```

**best.pth'nin rolü.** best.pth (converged max-RankMe) hâlâ **"max korunmuş rank" checkpoint'i**
olarak yazılıyor (downstream'de daha yüksek-rank isteyen için bir seçenek), ama backbone
KARŞILAŞTIRMASININ temeli DEĞİL. Yakınsama kapısı yine de random-init zirvesini (epoch 4) eler —
o kısım doğrudur; sorun yalnızca *converged epoch'lar arasında* pikin oynaklığıdır. Downstream için
final (kararlı, en collapse) vs diz (daha yüksek rank) tercihi **etiketli D3**'e bırakılır.
*(Uygulanmış: `train_simsiam.py::is_converged` + seçim bloğu; `select_model.py --select latest`.)*

### D2 — Run-arası (backbone) seçimi

Her run'ın **final (plato) checkpoint'i** aynı sabit kohortta değerlendirilir (leaderboard;
`select_model.py --select latest`). *(R1/best-converged yalnızca ablasyon olarak.)*

```
1. Uygunluk: converged(run) ∧ ¬collapsed(run)              # yakınsamayan run diskalifiye
2. Kararlılık: final plato checkpoint (oynak yakınsama piki DEĞİL)  # L13; dimensional-collapse [Jing2022]
3. Birincil anahtar: RankMe ↓ sırala                        # [Garrido2023] en güçlü etiketsiz proxy
4. Belirsizlik kuralı: iki run'ın RankMe %95 GA'ları örtüşüyorsa BERABERE   # jackknife GA (bootstrap DEĞİL, L9)
      (GA sütunları yoksa |ΔRankMe| < δ mutlak marjına düşer)
      - net kazanan (Δ > δ)  → seç
      - berabere            → PR, sonra U ile boz; ikisi AYNI yönü göstermeli
                              (çelişirlerse → "etiketsiz ayrım yok" → D3'e bırak)
5. Doğrulayıcı (çelişmemeli): singular-value spektrum sırası + t-SNE batch-effect kontrolü
6. Kazananı MARJI ve destekleyici/çelişen ikincil metriklerle raporla
```

**Neden marj / jackknife.** RankMe küçük kohortta gürültülüdür; ondalık farkları "kazanan" ilan
etmek savunulamaz. Rank-tabanlı istatistikte tekrarlı-satır bootstrap efektif rankı aşağı yanlı
iter; bu nedenle kohort-belirsizliği jackknife %95 GA ile verilir. Tohum-belirsizliği ise ayrı
olarak tohumlar-arası %95 GA ile raporlanır. Ancak *ayrık/anlamlı* marjda kazanan sayılır.

### D3 — Nihai (denetimli, etiket geldiğinde)

```
256-d feature üzerinde katmanlı-CV linear-probe / kNN → ROC-AUC (+ %95 GA)
Bu, etiketsiz proxy'yi EZER (nihai karar). İkisi uyuşursa proxy bu görev için doğrulanmış olur;
uyuşmazsa fark raporlanır (dengesiz görevlerde proxy başarısız olabilir [Otero2024]).
```

Garrido'nun kendi önerisi de budur: RankMe etiketsiz *ön eleme/seçim* içindir; nihai söz
etiketli probe'dadır [Garrido2023].

---

## 4. Eşiklerin gerekçesi

- **`z_std_min = 0.01`** — sağlıklı z_std ≈ 1/√256 ≈ 0.0625; 0.01 bunun ~%16'sı, net çökme sınırı
  [ChenHe2021]. Küçük (smoke) kohortta otomatik gevşetilir.
- **`best_val_loss_max = −0.8`** — neg-kosinüs tabanı −1; −0.8, tabanın %80'i = güçlü invaryans.
  Ampirik: epoch 4 (L_val≈−0.1, eğitilmemiş) elenir, epoch ≥14 (L_val≈−0.9) geçer.
- **`best_align_max = 0.15` (eğitim protokolü)** — random-init alignment ≈ 0.5; yakınsamış ≈ 0.02–0.15.
  0.15, "iki görünüm gerçekten birlikte haritalanıyor" eşiği [WangIsola2020].
- **`align_max = 0.30` (eval protokolü, D2/`select_model.py`)** — `evaluate.py` alignment'ı precise-BN
  uyarlanmış feature'larda, taze örneklenmiş augment view'larla ölçer; aynı checkpoint eğitim-zamanına
  göre sistematik olarak daha yüksek okunur (ör. densenet e19: 0.107 eğitim vs 0.151 eval). Eşik,
  yakınsamış (≈0.05–0.15) ile random-init'i (≈0.5) ayırmalı; **boşluğun ortasına (0.30)** konur ki
  gerçekten yakınsamış model sınır gürültüsüyle elenmesin, eğitilmemiş checkpoint yine de reddedilsin.
  Eşiklerin protokole-özgü olması gerektiği, D2'yi gerçek leaderboard'da test ederken ortaya çıktı
  (bkz. tasks/lessons.md L7).

Bu eşikler config'ten ayarlanabilir; değerler bu veri kümesinin yörüngesinden kalibre edildi ve
`metrics.csv`'den yeniden üretilebilir.

---

## Kaynaklar

- **[ChenHe2021]** Chen & He, *Exploring Simple Siamese Representation Learning*, CVPR 2021.
- **[WangIsola2020]** Wang & Isola, *Understanding Contrastive Representation Learning through
  Alignment and Uniformity on the Hypersphere*, ICML 2020.
- **[Garrido2023]** Garrido, Balestriero, Najman, LeCun, *RankMe: Assessing the Downstream
  Performance of Pretrained Self-Supervised Representations by Their Rank*, ICML 2023 (arXiv:2210.02885).
- **[Jing2022]** Jing, Vincent, LeCun, Tian, *Understanding Dimensional Collapse in Contrastive
  Self-Supervised Learning*, ICLR 2022.
- **[Otero2024]** Otero et al., *Self-Supervised Anomaly Detection in the Wild* — RankMe'nin
  dengesiz görevlerde tek-başına-vekil olarak başarısızlığı.
- **[Bardes2022]** Bardes, Ponce, LeCun, *VICReg: Variance-Invariance-Covariance Regularization
  for Self-Supervised Learning*, ICLR 2022.
- **[Thilak2024]** Thilak et al., *LiDAR: Sensing Linear Probing Performance in Joint Embedding
  SSL Architectures* — RankMe'yi rafine eden takip metriği.
- **[Goyal2017]** Goyal et al., *Accurate, Large Minibatch SGD*, 2017 (lineer LR ölçekleme).
- **[Loshchilov2017]** Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017.
- **[Efron&Tibshirani1993]** Efron & Tibshirani, *An Introduction to the Bootstrap* — jackknife GA;
  rank/spektral istatistiklerde tekrarlı-satır bootstrap yanlılığından kaçınmak için.
