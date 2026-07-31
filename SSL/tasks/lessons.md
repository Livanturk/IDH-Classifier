# Lessons — kararlar, düzeltmeler ve çıkarımlar

Her giriş: **Bağlam → Ne oldu → Çıkarım/Kural → Kaynak**. Amaç, aynı hatayı tekrarlamamak ve
bildiride her kararı gerekçeyle savunabilmek. Yeni bir düzeltme/karar geldikçe buraya eklenir.

---

## L28 — Aynı encoder checkpoint'i, ayrı precise-BN geçişleriyle ayrı policy sonucu gibi raporlama

**Bağlam.** RankMe, LiDAR ve AlphaReQ beş canonical seed'in her birinde aynı pretraining epoch'unu
ve birebir aynı `model` state'ini seçti.
**Ne oldu.** Downstream evaluator, her checkpoint dosyası için precise-BN'i `shuffle=True` ve
`drop_last=True` ile yeniden çalıştırıyordu. Sonuç olarak aynı encoder, farklı batch sırası ve farklı
atılan son batch üzerinden BN statistics alıyor; çok küçük feature/AUC farkları yapay olarak oluşuyordu.
**Çıkarım/Kural.** Precise-BN, policy-specific stochastic işlem değil encoder evaluation'ın parçasıdır:
sabit subject sırası, `drop_last=False` ve aynı `(run, pretraining epoch)` için ortak canonical feature
cache zorunludur. Aynı weights'ten gelen policy'ler aynı downstream prediction'ı vermelidir.
**Uygulandı.** `idh_probe.py`: `detbn-v2` cache, deterministic precise-BN ve epoch-canonical cache key;
`scripts/final_paired_selection_analysis.py`: yalnız bu yeni cache üzerinden final inference.

---

## L1 — RankMe random-init'te tavan yapar; "best" çökme değil, YAKINSAMA kapılanmalı

**Bağlam.** best.pth "en yüksek RankMe" ile kaydediliyordu.
**Ne oldu.** Üç run da best.pth olarak **epoch 4'ü** (val_loss≈−0.1, alignment≈0.5 = eğitilmemiş)
seçti; çünkü RankMe random-init dağınıklığı yüzünden en yüksek başlangıçta. Yani "best" = neredeyse
rastgele ağırlıklar.
**Çıkarım/Kural.** RankMe'yi checkpoint seçiminde **tek başına ve koşulsuz kullanma.** Önce
yakınsama kapısı: `val_loss ≤ −0.8 ∧ alignment ≤ 0.15`. Sadece bu kapıdan geçen epoch'lar arasında
argmax RankMe. Plato sayacı da yalnızca ilk yakınsamış best'ten sonra artmalı.
**Uygulandı.** `train_simsiam.py::is_converged` + gate'li seçim; `MODEL_SELECTION.md` D1.
**Kaynak.** [Garrido2023] (RankMe proxy), gözlem (RankMe↓ eğitim boyunca) = [Jing2022].

## L2 — SimSiam'da dimensional collapse gerçek; "diz" epoch'unu seç

**Bağlam.** RankMe eğitim boyunca 16–20'den 4–7'ye düşüyor (aynı 29 hasta → düşüş gerçek).
**Çıkarım/Kural.** Bu tam collapse değil (z_std≈0.044 sağlıklı), **boyutsal collapse** — SimSiam'ın
rank-koruyucu terimi olmadığı için beklenen [Jing2022]. Düşüş erken (epoch ~4→14) olup sonra
stabilize oluyor. **best.pth = yakınsamanın DİZİ (epoch ~19)**: invaryansı öğrenmiş ama rank henüz
çökmemiş nokta. last.pth (epoch 79) daha çok collapse olmuş → downstream için epoch 19 tercih.
**Sonraki iş.** Mutlak rank'ı artırmak için VICReg-tarzı variance/covariance regülarizasyonu
(SimSiam'a ek terim, yöntemi bozmadan) → bkz. tasks/todo.md. **Kaynak.** [Jing2022], [Bardes2022].

## L3 — RankMe bir PROXY'dir, kesin gerçek değil → üçgenle + AUC'ye bırak

**Bağlam.** Ana metrik neden RankMe, doğru mu?
**Çıkarım/Kural.** RankMe etiketsiz JE-SSL için literatürde yerleşik, downstream linear-probe ile
korelasyonlu [Garrido2023]. AMA dengesiz/karmaşık görevlerde tek-başına başarısız olabilir
[Otero2024]. Bu yüzden **tek metriğe güvenme**: PR + uniformity + spektrum ile üçgenle, nihai sözü
IDH linear-probe AUC'ye bırak (D3). Bildiride bu çerçeveleme zorunlu.
**Kaynak.** [Garrido2023], [Otero2024].

## L4 — n=29 val, mutlak RankMe için küçük; bildiride daha geniş etiketsiz kohort kullan

**Bağlam.** İzleme val'ı 29 hasta (%5). RankMe SVD'si min(29,256)=29 örnek üzerinde → gürültülü,
tavan ~28.
**Çıkarım/Kural.** Backbone *seçimi* için 29 yeterliydi (göreli sıralama tutarlı ve büyük marjlı).
Ama **bildiriye konacak mutlak sayılar için** metrik etiketsiz olduğundan train'i feda etmeden daha
geniş bir kohortta (≥100 denek) ayrı bir değerlendirme geçişiyle hesaplanmalı; RankMe farklarına
**bootstrap %95 GA** eklenmeli (D2, δ kuralı). **Kaynak.** [Garrido2023] (büyük batch kullanır).

## L5 — Tümör-merkezli recipe seg'e bağımlıdır (train + val + inference)

**Bağlam.** "Validation için seg gerekli miydi?"
**Çıkarım/Kural.** `crop_mode=tumor` seçildiği an seg, ROI'yi *konumlandırmak/maskelemek* için
okunur (ağ girdisi DEĞİL — kural korunur). Bu train, val ve downstream feature çıkarımının **hepsini**
seg'e bağımlı kılar → IDH kohortunda da tümör maskesi (GT ya da oto-seg) gerekir. `crop_mode=brain`
seçilseydi seg hiç okunmazdı. Bu bir tasarım-sonucu; bildiride bağımlılık açıkça belirtilmeli.

## L7 — Yakınsama eşiği PROTOKOLE özgüdür; eval-alignment eğitim-alignment'tan yüksek

**Bağlam.** D2 seçici (`select_model.py`) mevcut leaderboard'da test edilirken **densenet (gerçek
kazanan) diskalifiye oldu** ve yanlışlıkla r18 kazandı.
**Ne oldu.** densenet e19 eval-alignment'ı 0.1507, gate eşiği 0.15'in 0.0007 üstünde. Kök neden:
`evaluate.py` alignment'ı precise-BN + taze augment view'larla ölçüyor → aynı checkpoint eğitimde
0.107, eval'de 0.151 okunuyor. İkisi de doğru; ama 0.15 eşiği yakınsamış-sınırına oturduğu için kırılgan.
**Çıkarım/Kural.** Yakınsama eşiği **ölçüm protokolüne göre** ayarlanmalı. Eşiği metriğin *sınırına*
değil, yakınsamış (≈0.1) ile random-init (≈0.5) arasındaki **boşluğun ortasına** (eval için 0.30) koy —
gürültüye dayanıklı olsun. Bir bildiride "en iyi modeli 0.0007 yüzünden eledik" savunulamaz.
**Uygulandı.** `select_model.py` ALIGN_MAX=0.30 (gerekçeli); MODEL_SELECTION.md §4. "Verify before
done" bunu yakaladı — tool'u gerçek veride koşturmasaydık gözden kaçardı. **Kaynak.** [WangIsola2020].

## L8 — Kohort: UPENN dahil, UCSF hariç (çapraz-makale leakage)

**Bağlam.** Bildiri saf SSL olduğu için "UCSF ve UPENN de dahil edilsin mi?"
**Çıkarım/Kural.** SSL etiketsiz → label leakage yok, veri eklemek iyi. AMA **UCSF gelecekteki IDH
makalesinin harici test seti**; şimdi üzerinde pretrain edersen image-level leakage olur, harici
doğrulama iddiası yanar. Bu yüzden: **UPENN DAHİL** (downstream train, sorun yok), **UCSF HARİÇ**
(temiz kalsın). Karar SSL'e değil, gelecek makalenin protokolüne bağlıdır. Split:
`splits_pretrain_noucsf.json` (train=889, val=99). **Kaynak.** CLAUDE.md data-leakage notu.

## L9 — RankMe için GA'da bootstrap DEĞİL jackknife kullan

**Bağlam.** A0.6'da RankMe/PR/uniformity için %95 GA eklerken naive bootstrap yazıldı.
**Ne oldu.** Birim-test: nokta-tahmin RankMe=93.5 ama bootstrap CI=[53,64] — nokta-tahmini
**içermiyor.** Kök neden: bootstrap örnekleri tekrarlı satır içerir (~%63 benzersiz); tekrarlı
satırlar lineer bağımlı → effective rank'ı düşürür → rank-tabanlı istatistik aşağı saptırılır.
**Çıkarım/Kural.** Rank-tabanlı (spektral) istatistiklerin GA'sında **naive bootstrap kullanma**;
**jackknife (leave-one-out)** kullan — alt-kümeler benzersiz satır içerir, rank çökmez. RankMe/PR/
uniformity/alignment *düzgün* fonksiyoneller olduğu için jackknife geçerli (medyan gibi düzgün-olmayanların
aksine), üstelik deterministik (tekrarlanabilirlik). **Uygulandı.** `utils.jackknife_ci` (bootstrap_ci
değil); MODEL_SELECTION.md D2 δ = CI-örtüşme. **Kaynak.** [Efron&Tibshirani1993].

## L10 — VICReg VC-reg ağırlıkları SimSiam'a doğrudan transfer OLMAZ; λ-sweep zorunlu

**Bağlam.** A0.1'de VC-reg eklenip smoke ile test edildi (var=1.0, cov=0.04).
**Ne oldu.** Toplam loss +39.7 çıktı — cov terimi (≈992) × 0.04 ≈ 40, SimSiam invaryansını (~1) boğdu.
**Çıkarım/Kural.** VICReg'in orijinal ağırlıkları (var:inv:cov = 25:25:1) **MSE-ölçekli** invaryans içindir;
bizim invaryansımız **kosinüs-ölçekli** (|L|~1). Bu yüzden ağırlıklar yeniden kalibre edilmeli.
Sweep'te ilk kontrol: `w_var·var` ve `w_cov·cov` terimleri, `|simsiam_loss|~1` ile **aynı büyüklük
mertebesinde** olmalı — biri diğerini boğmamalı. Config'teki değerler sweep MERKEZİ, kullanıma-hazır
değil. **Uygulandı.** models.vc_regularizer var/cov'u AYRI döndürüyor (loglanıp dengelenebilsin);
Stage C.2 sweep. **Kaynak.** [Bardes2022].

## L11 — W-MSE 3D küçük-batch rejiminde küçük projektör gerektirir

**Bağlam.** A0.2'de W-MSE whitening eklendi.
**Çıkarım/Kural.** Tam-rank batch whitening için **batch_size > proj_dim** şart (kovaryans tekil
olmasın). 3D bellek bütçesi batch'i 16'da tutuyor → whitening ancak **küçük projektörle** (ör.
proj_dim=8) anlamlı. Bu W-MSE'nin doğasında var (küçük projeksiyon kullanır) ama bizim batch tavanımız
onu alışılmadık derecede daraltıyor → W-MSE'nin, VICReg'in yumuşak covariance cezasına göre burada daha
**zayıf** bir rank kolu olması beklenir. Bu kısıt bildiride 3D küçük-batch rejimi için bir bulgu olarak
raporlanmalı. **Uygulandı.** models.whitening_mse_loss + config uyarısı (proj_dim>=batch ise warn).
**Kaynak.** [Ermolov2021].

## L12 — UCSF'li pretrain: veri-kohort ablasyonu (temiz kurulum = UCSF sadece train'e)

**Bağlam.** L8'de UCSF hariç tutulmuştu (harici test); şimdi "UCSF ekleyerek de bir pretrain yapalım".
**Çıkarım/Kural.** Bu L8'i çürütmez — ek bir **ablasyon ekseni**: "UCSF eklemek (daha çok veri)
RankMe/PR/uniformity'yi artırıyor mu?". Temiz olması için: UCSF **YALNIZCA train'e** eklenir, **val
noucsf'in 99'uyla aynı tutulur** → iki encoder birebir aynı benchmark'ta yargılanır, tek değişken
UCSF'in eğitimde olması. Split: `scripts/make_withucsf_split.py` → `splits_pretrain_withucsf.json`
(train=noucsf+UCSF, val=aynı). Ablasyon kazanan backbone'da (densenet121) yeterli. **ÇAPRAZ-MAKALE:**
bu encoder IDH makalesinde UCSF-harici-test için KULLANILAMAZ (UCSF görülmüş olur) — o rol noucsf'te kalır.
Headline hâlâ noucsf; bu yalnızca "daha çok veri" etkisini gösteren etiketsiz bir ablasyon. **Kaynak.** [[paper-scope]], L8.

## L13 — Çok-seed analizi: best.pth karşılaştırma temeli OLAMAZ; karşılaştırma = final plato

**Bağlam.** Stage-A (noucsf ×9) + D.3 (withucsf ×9) = 18 koşu bitti (hepsi yakınsadı, hiç collapse yok).
metrics.csv'lerden (GPU'suz) best.pth kuralını ve backbone sıralamasını analiz ettik.

### Bulgu 1 — "converged max-RankMe" (mevcut best.pth kuralı, "R1") OYNAK
- **R1 mekanik olarak her zaman ilk-yakınsayan epoch'u seçer**, çünkü RankMe yakınsamada zirve yapıp
  monoton düşer (best→last %20–57 çöküş). Yani "yakınsamışlar arasında max" = "converged bölgenin en
  solu" = ilk-yakınsayan.
- **O zirvenin DEĞERİ oynaktır** — üç sebep: (a) dik bir geçiş (transient) üzerinde; (b) her-5-epoch
  ızgarasıyla örneklendiğinden ızgara-bağımlı (her-1-epoch olsa farklı çıkardı); (c) yakınsama epoch'u
  **seed'e göre kayıyor** (9 vs 14). Kanıt — r18 noucsf: s42/s43 epoch 9'da yakınsadı → RankMe 12–13;
  s44 epoch 14'te → RankMe 8. Aynı backbone, **CV %24**.
- Sonuç: R1'de **densenet ile r18 istatistiksel olarak AYRILAMIYOR** (CI örtüşüyor) → karşılaştırma bozuluyor.
- **Kavramsal ek:** ilk-yakınsayan = yakınsamışların EN AZ yakınsamışı (alignment ~0.11, en yüksek);
  oradaki yüksek rank'ın bir kısmı henüz collapse edilmemiş **augmentasyon-nuisance varyansı** → "en çok
  bilgi" değil, kısmen "temizlenmemiş çöp".

### Bulgu 2 — Dört kuralın kıyası (seed-CV, sıralama anlamlılığı)
| Kural | epoch | ort.CV | sıralama |
|---|---|---|---|
| R1 first-conv maxRankMe (mevcut) | 9–14 (seed'e göre) | %15–17 | densenet**≈**r18 (muğlak) |
| R2 last epoch | 84–89 | %8–10 | densenet>r18>r34 ✓ |
| **R3 fixed epoch=79** | 79 (hep aynı) | **%7–8** | densenet>r18>r34 ✓ |
| R4 stability-window (+20ep≈34) | ~34 | %9–11 | densenet>r18>r34 ✓ |

### KARAR (bildiri için)
- **Karşılaştırma temeli = final/plato checkpoint (last.pth).** Basit, ızgara/seed-zamanlaması bağımsız,
  tekrarlanabilir, savunulabilir. Erken-durdurma zaten **RankMe-platosu**yla tetiklendiğinden "final" keyfi
  değil, prensipli bir durak.
- **R1'i seçim temeli olmaktan çıkar → "naif seçim kararsızdır" ABLASYONU olarak göster.** Bu, kapılı +
  kararlı-nokta metodolojimizin *neden gerekli* olduğunun kanıtı = zayıflık değil, katkıyı güçlendiren bulgu.
- **Ana figür:** RankMe-vs-epoch yörüngesi (`figures/rankme_trajectory.png`) — random-init yüksek → yakınsama
  piki → collapse; densenet tüm yörüngede üstte. Neden-zirve-değil sorusunu görselle önceden yanıtlar.
- **best.pth'nin rolü:** "max korunmuş rank" checkpoint'i olarak kalır (downstream için bir seçenek);
  final (kararlı, en collapse) vs diz (daha yüksek rank) tercihi **etiketli D3**'e bırakılır.

### Bulgu 3 — Backbone sıralaması (çok-seed, DÜZELTİLMİŞ) + kohort etkisi
- **densenet121 > r18 > r34**, her iki kohortta final epoch'ta CI-anlamlı. **r34 EN KÖTÜ** (tek-seed'de
  ortadaydı; çok-seed düzeltti — 63M parametresi işe yaramıyor). **r18 sağlam İKİNCİ** (tek-seed'de en
  kötü sanılıyordu, yanlışmış). Tek-seed sonucunun neden güvenilmez olduğunun kanıtı.
- **+UCSF etkisiz (null):** densenet 7.37→8.77 vb. ama tüm CI'lar örtüşüyor → daha çok veri etiketsiz
  temsil kalitesini anlamlı artırmadı (889 denek zaten yeterli / collapse yönteme bağlı). Raporlanabilir.

**Uygulandı.** `select_model.py --select latest` (default; best-converged=R1 ablasyon), MODEL_SELECTION.md
§D1/D2, `train_simsiam.py` best.pth yorumu, `pretrain_stageA.sh` COMPARE→last.pth, yörünge figürü.
**Kaynak.** [Garrido2023], [Jing2022], [[paper-scope]].

## L14 — Kohort kararı GÜNCELLENDİ (2026-07-23): tek kohort, TÜM veri (L8/L12'yi ezer)

**Bağlam.** cancerimagingarchive'den harici UCSF (501) + UPENN (610) indirildi (BraTS-içi 263/403'ten
çok); BraTS-içi olanlar bunlarla değiştiriliyor. Kullanıcı: "noucsf vs withucsf kalksın, hepsi olacak."
**Karar.** TEK kohort = BraTS-diğerleri (585) + harici UCSF (501) + harici UPENN (610) ≈ **1696**.
noucsf/withucsf ayrımı, +UCSF ablasyonu (eski D.3), ve `*_noucsf/withucsf` split/config'leri **iptal**.
**Çapraz-makale (L8/L12'yi bilinçli tersine çevirir):** UCSF artık pretraining'de → IDH makalesinde temiz
harici-test OLAMAZ; kullanıcı bunu bilerek onayladı. **Veri harmonizasyonu:** harici setler BraTS uzayında
(240×240×155, 1mm, skull-stripped, atlas) → geometri OK; **8-bit (harici) vs 16-bit (BraTS)** farkı z-score
+ aug ile örtülür (8-bit standarttır); UPENN harici seg=otomatik. Heterojenliği t-SNE batch-effect ile
doğrula + bildiride belgele. **Uygulama:** birleştirilmiş symlink dizini (kanonik `<subj>_t1/..._seg`) →
scan_subjects değişmeden çalışır. **Kaynak.** [[paper-scope]].

## L15 — Shared-storage `.nii.gz` okuması transient hata verebilir; dosya bozulmasını önce ayır

**Bağlam.** Unified Stage-A'da ai02 üzerindeki dört eşzamanlı run, epoch 0'da farklı DataLoader
worker'larında `zlib.error: invalid block type` ile durdu. Hata model/loss'ta değil,
`nibabel -> gzip` NIfTI okumasındaydı.
**Kontrol.** `data_unified` altındaki 8,480 symlink-hedefli `.nii.gz` için `gzip -t` taraması
başarılı oldu; kalıcı olarak bozuk bir dosya saptanmadı. Bu nedenle ilk açıklama, 4 run × 12 worker'ın
shared storage üzerinde oluşturduğu eşzamanlı gzip/I/O yükünde transient read failure'dır; persistent
bir dosya hatası olsaydı tarama veya retry'nin son hatası exact path'i vermelidir.
**Çıkarım/Kural.** Gzipped NIfTI bir shared filesystem'den okunuyorsa (i) tüm cohort'u `gzip -t` ile
önce doğrula, (ii) loader'da fresh `nib.load(..., mmap=False, keep_file_open=False)` ile sınırlı retry
kullan, (iii) persistent hata mesajına dosya yolunu yaz, (iv) eşzamanlı worker sayısını I/O bütçesine
göre indir. Retry, bozuk dosyayı gizlemek için değil transient hata ile kalıcı hatayı ayırmak içindir.
**Uygulandı.** `data._read_nifti`: 3 deneme + yol içeren nihai hata; unified config'lerde
`num_workers: 12 → 4` (4 GPU job'unda toplam 48 → 16 worker). Rerun epoch 0'da çöktüğü için checkpoint
ve deney sonucu kaybı yoktur.

## L6 — Repo yerleşimi

Kod `/datasets/mri_datasets/SSL`'de; git `/home/ibm`'de (onun `SSL/`'i silinmiş). `BraTS2021`
sembolik linki repo kökünde olmalı ki split'lerin data-root-göreli yolları çözülsün. Ağır veri
(`BraTS2021/`, `UCSF-PGDM/`) ve `.env` asla commit edilmez.

---

## L15 — best.pth = KNEE; etiketsiz kanıt = güvenilirlik + convergent validity (etiketli AUC YOK)

**Bağlam.** best.pth "converged içinde max RankMe" (=ilk-converged) seçiyordu; kullanıcı bunu ve
etiketli-AUC yokluğunu sorguladı.
**Ne oldu (18-koşu analizi, `scratchpad/knee_analysis.py`).** (a) Global RankMe peak HER koşuda epoch 4
(random-init), converged değil → gate temizliyor. (b) "converged max" = ilk-converged = **dik geçiş
üstünde**, seed-CV %13.2 (en oynak); önünde ort. **3.15 RankMe** daha düşecek. (c) Alignment-rank
DECOUPLING: rank kaybının **%81'i knee ÖNCESİ** ve alignment iyileşmesiyle eşleşiyor (nuisance removal =
istenen); knee sonrası alignment ~düz ama rank hâlâ azalıyor (SimSiam'ın decorrelation'sız hafif collapse
eğilimi). (d) last.pth'te bile z-std 0.040–0.044, PR>1 → gate tutuyor, kuyruk patolojik değil, hafif.
**Çıkarım/Kural.** **best.pth = KNEE (plateau onset)** — dik geçiş bitmiş, stable rank'ın en yükseği.
**D2 karşılaştırma istatistiği = plateau-mean** (knee→son; seed-CV %7, en reproducible). İkisi aynı plateau
bölgesinde → tutarlı. R1 (peak) ve last = ablasyon.
**SERT KISIT.** Etiketli AUC YOK → downstream-optimallik kanıtlanamaz; onun yerine (1) reliability
(seed-CV), (2) convergent validity (bağımsız proxy uzlaşması), (3) construct validity (teori), (4) ablasyon
(naif peak kararsız), (5) confound (r34 en çok param, en kötü). Rank→performans bağı Garrido'dan MİRAS.
**Uygulandı.** `utils.lidar` (nuisance-whitened LDA effective rank) + `utils.alpha_req` (spektrum power-law
eğimi) eklendi; `evaluate.py` leaderboard + lidar CI; sentetik test: LiDAR spread↔nuisance'ı ayırıyor
(47↔35), RankMe ayıramıyor (59↔59) → LiDAR'ın RankMe'ye kattığı ek ayrım kanıtı. **Kalan:** knee-tabanlı
best.pth kuralını `train_simsiam.py`/`select_model.py`'de kodla; `MODEL_SELECTION.md`'yi güncelle.
**Kaynak.** [Garrido2023], [Jing2022], [Hua2021], [Thilak2024], [Agrawal2022], [Satopää2011], [[paper-scope]].

## L16 — Convergent-validity GERÇEK sonucu: "densenet kazanır" robust, tam 3'lü sıra DEĞİL

**Bağlam.** LiDAR+α-ReQ'i 18 last.pth'te (val n=99) hesapladık (embedding tabanlı), backbone-sırası uzlaşmasını
test ettik. Doğru yönler: rankme↑, lidar↑, **alpha_req↓** (α≈1 ideal; rejimimizde α>>1 → düşük iyi), uniformity↓, PR↑.
**Ne oldu (per-backbone, 6'şar koşu).** rankme: D 9.22 > R18 6.08 > R34 4.95 (temiz). alpha_req: R18 2.62 ≈ D 2.74 <
R34 3.01 (R34 net en kötü, D≈R18 berabere). lidar: D 3.06 / R18 2.69 / R34 2.90 — **aşırı örtüşen, gürültülü,
sonuç çıkmıyor** (n=99 ≪ d=256 → Σ_b rank-eksik). uniformity/PR plateau'da dejenere.
**Çıkarım/Kural.** **ROBUST iddia = "DenseNet-121 kazanan"** (her metrikte en iyi ya da berabere; RankMe'de net) +
**"R34 üst-lig değil"** (RankMe & α-ReQ ikisi de R34'ü sona koyuyor → anti-'bigger is better'). **Tam 3'lü sıra
(D>R18>R34) SADECE RankMe'ye dayanır** — α-ReQ D≈R18 der. Paper'da "tüm metrikler tam sırayı verdi" DEME; "densenet
kazanır (çok-metrik) + tam sıra RankMe-birincil" de. **LiDAR'ı limitasyon** olarak raporla veya n'i büyüt (etiketsiz →
train denekleri de eklenebilir, leakage yok; n~500+ hedefle). Not: eski-18 verisi; unified'da tazelenecek.
**Uygulandı.** `convergence_analysis.py` alpha_req yönü -1'e düzeltildi. **Kalan:** docx §6.1'i bu dürüst sonuçla
güncelle; LiDAR'ı büyük-n ile yeniden hesapla; workflow/figürlerdeki "densenet>r18>r34"ü "densenet kazanır"a yumuşat.
**Kaynak.** [Garrido2023], [Thilak2024], [Agrawal2022], [[paper-scope]].

## L17 — UCSF pretraining'den ÇIKARILDI + çok-metrikli checkpoint seçimi (2026-07-26, kullanıcı; L14'ü kısmen ezer)
**Bağlam.** Kullanıcı kararı: UCSF-PDGM encoder-tarafı HİÇBİR adımda kullanılmaz (eğitim / validation /
checkpoint seçimi / hiperparametre / erken durdurma). UCSF = harici, tercihen tek-seferlik downstream
doğrulama seti. L14'teki "tek kohort, TÜM veri (UCSF dahil)" kararı bu yönüyle GEÇERSİZ.
**Uygulandı.**
- Yeni split `splits/splits_pretrain_noucsf_all.json` = data_unified − UCSF (1135 train / 60 val).
  `make_splits` artık exclude'u `--limit`'ten ÖNCE uygular; smoke da UCSF'siz; `smoke_unified.json` silindi.
- 3 `*_unified.yaml` → yeni split + `val_every: 1`; `*_withucsf.yaml` DEPRECATED banner'lı.
- `train_simsiam`: `validate()` artık her epoch RankMe + **LiDAR** + **α-ReQ** hesaplar; **üç ayrı best**
  (`bestRankMe/bestLiDAR/bestA-ReQ.pth`) ortak yakınsama+çöküş kapısıyla; her checkpoint'te `selection`
  provenans bloğu; `best.pth` = bestRankMe alias. **α-ReQ best = min |α−1|** (1'e en yakın, Agrawal2022),
  L16'daki "convergence_analysis'te düşük-α=iyi" yönüyle KARIŞTIRMA (o yön yüksek-α rejimi için sıralama içindi).
- **Erken durdurma = ÜÇÜ DE plato** (union; her metriğin ayrı `no_improve` sayacı) → hiçbir best checkpoint
  hâlâ tırmanırken kesilmez. metrics.csv'e lidar/alpha_req/lr/no_improve_*/stopped/stop_reason eklendi.
- Yeni: `report_run.py` (run-başına 6-maddelik karşılaştırma + filtrelenebilir `checkpoint_selection_summary.csv`),
  `downstream_ablation.py` (3-checkpoint transfer iskelesi; label stub; UCSF yalnız `--external` final),
  `CHECKPOINT_SELECTION_METHODOLOGY.md`, `scripts/pretrain_unified_noucsf.sh` (backbone × seed).
**Kural.** DenseNet121/seed42 yalnız ÖRNEK — tüm mimari/config/seed'lerde genellenebilir uygula (cfg
checkpoint'ten okunur). RankMe birincil KALIR; LiDAR bağımsız kontrol, α-ReQ spektrum-şekli. Selection-bias'ı
önceden-tanımlı kural + kapı + plato + kararı etikete/harici-UCSF'e erteleme ile sınırla.
**Kaynak.** [Garrido2023], [Thilak2024], [Agrawal2022], [[paper-scope]], `CHECKPOINT_SELECTION_METHODOLOGY.md`.

## L18 — Metrikler ölçülebilir olmadan seçilemez: val n, α fit penceresi, LiDAR view sayısı (2026-07-27)

**Bağlam.** L17 üç metrik için üç checkpoint tanımladı ama üçünün de **ölçüm tabanı** sorgulanmamıştı:
val kohortu 60 denek, feature boyutu 256, LiDAR denek başına 2 view, α tüm spektruma fit ediliyordu.

**Ders.** Sentetik (gerçek spektrumu bilinen) kalibrasyonla ölçüldü:
1. **RankMe aşağı yanlı, n ile düzelir.** Gerçek 23.9 → n=60'ta 13.2, n=200'de 19.8. Tavan (min(N,d))
   hiç bağlayıcı değildi; sorun tavan değil **yanlılık**. Val 60 → 200 (995/200, collection-stratifiye).
2. **α'nın yanlılığı n ile DÜZELMEZ, pencereyle düzelir.** Tüm spektrumu fit etmek n=200'de n=60'tan
   daha kötü (n<d rejiminde kuyruk aşağı yanlı → eğim dikleşir). `k_min=1, k_max_frac=0.30` en kötü
   |bias|'ı 0.21'den 0.055'e indiriyor. **Baştan atmak (k_min>1) yanlılığı artırır** — baş en iyi
   tahmin edilen kısımdır, sezginin tersi. n=200'ün α'ya katkısı varyanstır (SD 0.09 → 0.01).
3. **α asla R²'siz raporlanmaz.** OLS eğimi her spektrum için tanımlıdır; power-law tutmuyorsa α
   anlamsızdır. `areq_r2_min: 0.95` (gerçek power-law ≥0.996, power-law olmayan kontrol 0.76).
4. **LiDAR A=2 boyutun altında.** Within-subject scatter dof = n(A−1) = 200 < d=256. A=8 → 1400 = 5.5·d.
   Ek view'lar tek hacim okumasından türer (`BraTSViews.n_views`), maliyet ~%20–30, 4× değil.
5. **Her epoch val feature dump'ı** (`val_features/ep####.npz`) — yukarıdaki kararların hepsi post-hoc
   revize edilebilir hâle gelir. Bunu yapmadan her metodolojik seçimi run başlamadan doğru vermek
   zorundasın; yaparak hiçbirini vermek zorunda değilsin. **Yeni ölçüm eklerken önce dump'ı ekle.**

**Sonuç.** Eski (n=60/99) run'ların RankMe/α değerleri yenilerle karşılaştırılamaz — hepsi yeniden koşulacak.
**Kaynak.** `CHECKPOINT_SELECTION_METHODOLOGY.md` §6, [[paper-scope]], L17.

## L19 — Seçim kuralı: eşik veriden gelmeli, ve raporlanacak şey seçimin kararlılığı (2026-07-27)

**Bağlam.** L18 metrikleri ölçülebilir yaptı; geriye o ölçümlerden karar üretme kuralı kaldı.
Eski kural: ham per-epoch değerin argmax'ı + elle yazılmış `min_delta_*`.

**Ders.**
1. **Sabit eşik gürültünün onda biriydi.** n=200'de RankMe'nin denekler üzerindeki örnekleme SE'si
   ≈0.5; `min_delta_rankme` 0.05. Yani "iyileşme eşiği" diye konan sayı fiilen eşiksiz argmax
   demekti. Eşik **veriden** gelmeli: delete-d jackknife SE + 1-SE kuralı (`select_se_mult`).
   Sabitleri elle kalibre etme — kalibre edilebilir bir istatistik varsa onu kullan.
2. **Yumuşatma nedensel olabilir.** Trailing-mean online checkpoint kaydıyla uyumlu; "post-hoc
   seçim gerekir" diye düşünmeye gerek yok. Kaydedilen ağırlıklar pencerenin sonunda = plato içinde.
3. **Eşiği sıkılaştırınca `patience`'ı da büyüt.** Yavaş ama gerçek bir yükseliş artık eşiği geç
   geçer; patience sabit kalırsa plato sanılıp erken durdurulur (15 → 25).
4. **Raporlanacak büyüklük metrik değil SEÇİM.** İddia bir seçim hakkındaysa kararlı olması gereken
   şey seçimdir. `selection_stability.py`: alt-örneklemlerde P(aynı epoch) + kayma dağılımı.
   Düşük P + küçük kayma = düz plato (argmax değil platoyu raporla); düşük P + büyük kayma = seçim
   tanımlı değil, o metrik o run için kural olarak sunulamaz.
5. **Projeksiyon kontrolü PCA ile yapılmaz.** Bir epoch'ta fit edilen PCA o epoch'un bazını diğer
   tüm epoch'ların skoruna gömer. Sabit tohumlu rastgele ortogonal projeksiyon kullan (128 / 64).
6. **Tek run bir iddianın birimi değil.** `aggregate_runs.py` seed'ler üzerinden ortalama ± SD verir
   ve politikalar arası fark seed gürültüsünün içindeyse sıralama yapmayı reddeder.
7. **Terminoloji:** "best checkpoint" değil, "RankMe-selected / LiDAR-selected / α-ReQ-selected".

**Kaynak.** `CHECKPOINT_SELECTION_METHODOLOGY.md` §7, [Efron&Tibshirani1993], L18, L17.

## L20 — Bir kohortu ilk kez okuduğun yerde bütünlük kontrolü yap (2026-07-28)

**Bağlam.** Dokuz encoder pretraining'i sorunsuz koştu, ama ilk downstream işi UCSF'in ilk
deneklerinde `zlib.error: invalid block type` ile düştü ve **dokuz downstream işini birden**
bloke etti.

**Ders.**
1. **Geçici I/O ile gerçek bozulmayı ayır.** `data.py` paylaşımlı depoda transient okuma hatalarına
   karşı 3 kez deniyor (L15). Bu dosya üç denemede de aynı hatayı verdi → gerçek bozulma.
   `gzip -t` ile doğrula; deterministik hata = bozuk dosya, retry ile çözülmez.
2. **Bozulma sadece o kohortu ilk kullandığında ortaya çıkar.** UCSF pretraining'den dışlandığı için
   2505 dosyası aylarca hiç açılmamıştı. Yeni bir kohortu pipeline'a sokarken **önce** tara:
   `find <dir> -name '*.nii.gz' -print0 | xargs -0 -P12 -I{} sh -c 'gzip -t "{}" || echo "{}"'`
   (2505 dosya ~30 sn). 501 denekten 1'i bozuktu.
3. **Hatayı yutma, listele.** `try/except` ile sessizce atlamak yerine
   `splits/ucsf_excluded_subjects.txt`e gerekçesiyle yaz ve `idh_probe` oradan filtrelesin.
   Bildiride kohort **n=500** olarak raporlanmak zorunda; sessiz atlama bu sayıyı denetlenemez yapar.
4. **Tek bir bozuk dosya tüm downstream ızgarasını durdurur.** Otomasyon zincirinde tek noktadan
   çöken bir adım varsa, o adımın girdisini zincir kurulmadan önce doğrula.

**Kaynak.** L15 (transient okuma), `braintumor_ssl/idh_probe.py:load_exclusions`.

## L21 — Üç metrik AYNI checkpoint'i seçti: aday havuzu platoyu içermek zorunda (2026-07-28)

**Bulgu (ölçüldü, tahmin değil).** Dokuz unified run'ın hepsinde `bestRankMe.pth`, `bestLiDAR.pth`
ve `bestA-ReQ.pth` **byte-identical** (md5 aynı; r18 ep9, r34/densenet ep14). Dolayısıyla
"hangi seçim kuralı downstream'de kazanır?" sorusu ölçülemedi — UCSF probe'unda üç satır arasındaki
fark ≤0.003 AUC iken tek hücrenin CV std'si ±0.006–0.011.

**Kök neden.** `eligible = converged and (not collapsed)` — havuzda **plato koşulu yok**. RankMe ve
−|α−1| yakınsama sonrası monoton azaldığı için "uygun epoch'lar arasında argmax" her zaman **en erken
uygun epoch**a çözünüyor; üçü de aynı eval'de ateşliyor. Yan etki: `selection.rule.window` her zaman 1,
yani `select_smooth_window` (trailing mean) ve `select_se_mult` (jackknife SE eşiği) — L19'un iki ana
korumasi — **hiç devreye girmedi**; önceki best `-inf` olduğu için eşik trivially aşıldı.

**İroni.** `select_model.py` aynı hatayı bir üst katmanda (run'lar arası) zaten belgelemişti:
max-RankMe-among-converged "steep transient at a seed-dependent convergence epoch", CV ~%24, ve
oradaki çözüm plato checkpoint'ine geçmekti. Bu ders **epoch seçimi katmanına hiç uygulanmamıştı**.

**Metrikler bağımsız kanıt değil.** Uygun epoch'lar üzerinde Spearman:
ρ(RankMe, α-ReQ) = **+0.930 / +0.934 / +0.920** (r18 / r34 / densenet). α-ReQ bu rejimde RankMe'nin
neredeyse monoton bir dönüşümü. Kısmen bağımsız tek metrik LiDAR (ρ = +0.29…+0.89, backbone'a göre
oynak). Bildiride üç satır gösterilecekse bu korelasyon **birlikte** raporlanmalı.

**Karar.** Plato kapısı + SE-duyarlı Pareto, `braintumor_ssl/select_epoch.py` içinde **post-hoc**
uygulanır (online değil: eğitim sırasında running-min val_loss o anki değere eşit olduğundan ilk uygun
epoch her "platoya yakın" testini trivially geçer; plato ancak tam eğri varken tanımlı). Online
`best*.pth` olduğu gibi kalır ve bildiride **R1 baseline** rolünü üstlenir.
Kural: havuz = converged ∧ ¬collapsed ∧ areq_r2 ≥ 0.80 ∧ `val_loss ≤ min(val_loss) + plateau_tol`;
front = (RankMe, LiDAR, −|α−1|) üzerinde domination'ın `pareto_eps` × jackknife SE'yi aşmasını
gerektiren Pareto; |front| = 1 → seç, |front| > 1 → **"etiketsiz ayrım yok"** deyip `last.pth`'e düş.
Seçim yapmayı reddedebilmek kuralın özelliği, kusuru değil.

**Kaynak.** `select_epoch.py`, `select_model.py:8-14`, `CHECKPOINT_SELECTION_METHODOLOGY.md` §5.2, L13, L19.

## L22 — Validation ızgarası bir ölçüm parametresi değil, bir SEÇİM parametresidir (2026-07-28)

**Bulgu.** `val_every: 5` masum bir örnekleme sıklığı gibi görünüyordu. Seçim "kapıyı geçen ilk epoch"a
indirgendiği için (L21) ve kapı geçişi val_loss'un en hızlı değiştiği yere denk geldiği için, ızgara
**cevabı belirledi**:
- `r34_s43`: train loss e10=−0.719 → e11=−0.855, yani kapı ~e11'de geçildi; ızgara 14'e baktı.
  RankMe e9=18.47, e14=10.11 → seçilen checkpoint ~%40 farklı bir noktada.
- `densenet_s42`: e9'da val_loss=−0.8836 kapıyı geçmiş ama alignment 0.1632 vs eşik 0.15 — **kıl payı**
  eledi, 5 epoch ve ~5 RankMe puanı kayıp.
- Sonuç: r18 e9'da, r34/densenet e14'te seçildi. **Backbone karşılaştırması farklı eğitim miktarlarını
  kıyaslıyor**, ve bu fark bir backbone özelliği değil ızgara artefaktı.

**Ders.** Bir eşik-geçişi kuralı kullanıyorsan, örnekleme çözünürlüğünü eşiğin geçildiği yerde
belirle. Düzgün ızgara maliyeti yanlış yere harcar.

**Uygulama.** `val_schedule: {0: 5, 5: 1, 20: 2, 55: 5}` (42 eval, eskisi 20): geçiş penceresinde
(e5–19) her epoch, Pareto front'unun yaşadığı yerde (e20–54) her 2, düz platoda her 5. Eval başına
maliyet ~400 s ≈ 2.2 training epoch'u (r18: toplam 7s16dk, saf eğitim 5s02dk), yani **+2.4 h/run** —
her epoch validate etmenin +8.9 h'ına karşı. Her **validate edilen** epoch'a checkpoint yazılır, aksi
halde post-hoc seçicinin işaret edeceği ağırlık olmaz.

**Yan karar — early stopping KAPATILDI.** `patience` eval cinsinden sayılıyor: eski ızgarada
25 eval = 125 epoch > 100 bütçesi, yani `early_stop: true` **ölü koddu**. Yeni şemada ulaşılabilir
hale gelip run'ı ~e64'te kesecekti — Pareto havuzunun beslendiği plato örneklerini tam olarak yok
ederek. Plato burada inceleme nesnesi olduğu için run'lar sonuna kadar koşturulur.

**Kaynak.** `train_simsiam.parse_val_schedule`, `logs/pretrain_10058_*.out`, L21.

## L23 — LIMITATION: Pareto kuralı kendi hiperparametrelerine kararsız (2026-07-28)

**Bulgu.** `select_epoch.py --sweep`, 9 run × (`plateau_tol` ∈ {0.005, 0.01, 0.02}) ×
(`pareto_eps` ∈ {0, 0.5, 1, 2}): kesin seçim yapılan run sayısı **0/9 ile 7/9 arasında** oynuyor ve
monoton bir örüntü yok. Aynı run için seçilen epoch hücreden hücreye kayıyor (`r18_s42`: e29 / e39 /
deferred). eps=2 SE'de her yerde front şişip 0/9'a düşüyor; eps=0.5, tol=0.01'de 7/9.

**Neden.** Plato havuzu 5 epoch aralıklı yalnızca 13–16 nokta, ve platoda metrik farkları çoğunlukla
kendi SE'lerinin altında. SE bantlarıyla "A, B'yi domine eder mi" sorusu yazı-tura oluyor.

**Ders.** Bu tam olarak `CHECKPOINT_SELECTION_METHODOLOGY.md` §4'ün uyardığı *researcher degrees of
freedom* problemi. Seçim oranını maksimize eden (eps, tol) çiftini seçmek post-hoc tuning olur ve
bildiriyi savunulamaz kılar.

**Karar.**
1. (`plateau_tol` = 0.01, `pareto_eps` = 1.0) **ön-kayıtlıdır** ve gerekçesi a priori: 1 SE, L19'un
   `select_se_mult` ile zaten kullandığı "fark kendi belirsizliğini aşmalı" konvansiyonu; 0.01,
   val_loss'un [−1, 0] sınırlı ölçeğinin %1'i. Seçim oranına bakarak **seçilmedi** (o değerlerde
   yalnızca 3/9 kesin seçim çıkıyor — düşük, ama dürüst).
2. Duyarlılık ızgarası bildiride **limitation olarak raporlanır**, gizlenmez.
3. Yoğun validation (L22) plato çözünürlüğünü ~2 katına çıkarır; kararlılık **yeniden koşumdan sonra**
   tekrar ölçülmeli. Şu anki 3/9 rakamı 42-eval'lik run'lar için geçerli değildir.
4. Parametreler tek kaynakta sabit: `report_run.PLATEAU_TOL` / `PARETO_EPS`. Training configlerine
   konmadı — eğitim onları okumuyor, okunmayan config anahtarı tuzaktır.

**Kaynak.** `select_epoch.sweep`, `results/pareto_sensitivity.csv`, `CHECKPOINT_SELECTION_METHODOLOGY.md` §4.

## L24 — Seyreklik bir SONUÇ mu yoksa optimizer hatası mı? (2026-07-28)

**Bağlam.** Defterdeki downstream planı 258 sütunluk tasarımı (256 encoder + age + sex) ElasticNet
ile ~30 özelliğe indirmeyi öngörüyordu. İlk uygulamada `LogisticRegressionCV(solver="saga",
max_iter=5000)` gerçekten seyreklik üretti — 98/258 — ama `ConvergenceWarning` ile birlikte.

**Bulgu (UCSF, n=500, 103 pozitif).**

| konfigürasyon | AUC | tutulan özellik |
|---|---|---|
| L2 logistic (referans) | 0.901 | 258/258 |
| enet, max_iter=5000 (**yakınsamadı**) | 0.885 | **98**/258 |
| enet, max_iter=20000, tol=1e-3 (yakınsadı) | 0.904 | **231**/258 |
| enet + top-k=30 zorlaması | 0.900 (image+age+sex) / **0.678** (image) | 30 |

**Ders.** Yakınsamamış bir L1/elastic-net çözümü **yapay olarak seyrektir**: koordinatlar henüz
sıfırdan uzaklaşmamıştır. O 98 sayısı bir özellik seçimi değil, bir optimizer arızasıdır — ve
daha DÜŞÜK AUC ile gelir, yani "seyreklik ücretsiz" yanılsaması bile vermez. Herhangi bir seyreklik
rakamı raporlanmadan önce yakınsama doğrulanmalı (`ConvergenceWarning` = rakam geçersiz).

**Sonuç.** Düzgün yakınsayınca iç CV seyrekliği neredeyse hiç seçmiyor (231/258): bu kohortta
encoder boyutlarını atmak AUC'ye mal oluyor. Yani **"258→30" veriden gelen bir bulgu değil, analistin
koyduğu bir kısıt.** İkisi de destekleniyor: varsayılan `--head elasticnet` seyrekliği iç CV'ye
bırakıp fiilî sayıyı raporlar; `--enet_top_k 30` defterdeki varyantı üretir ama bedeli görünür —
saf `image` kolunda AUC 0.729 → 0.678.

**Ayrıca (bu tabloda asıl okunması gereken).** `image+age+sex` AUC'si 0.90 civarı, ama **age tek
başına ~0.90**. Bu kolda görüntünün katkısı ölçülmüyor. Seçim kurallarını ayırt edebilecek tek kol
saf `image` (~0.68–0.76), ve `paper_auc_table.py` varsayılanı bu yüzden `--arm image`.

**Kaynak.** `idh_probe._enet` / `TopKElasticNet`, L21.

## L25 — "Seçmeyi reddetmek" bir kola dönüşürse deneyi bozar (2026-07-28)

**Bağlam.** Kapsam `Proposed Experimental Framework` belgesiyle daraltıldı: tek backbone (ResNet18),
tek soru — önerdiğimiz seçim kuralı mevcut kurallardan daha iyi bir checkpoint mi seçiyor. Kollar:
random-init · naive argmax ×3 · gated argmax ×3 · **Proposed** · last epoch.

**Ne oldu.** L21'de tasarlanan "front > 1 → etiketsiz ayrım yok → `last.pth`'e düş" davranışı, tek
başına doğru bir refleksti (üç neredeyse-eşdoğrusal metrikle zorlamalı tie-break sahte kazanan
üretir). Ama bu davranış **`last.pth`'in ayrı bir kol olduğu** bir tabloya girince kendi kendini
sabote ediyor: Proposed bazı seed'lerde Last-epoch kolunun kopyası oluyor (r18: s42 deferred, s43
e24, s44 e34) ve iki kolun kıyası tanımsızlaşıyor. Ölçüm aracı, ölçtüğü şeyin içine karışıyor.

**Ders.**
1. **Bir kuralın "karar veremiyorum" çıkışı, karşılaştırmadaki başka bir kolun kendisi olamaz.**
   Belirsizliği bildirmek ile artefakt üretmek ayrı iki iştir; kural ikisini de yapmalı.
2. Çözüm: `deferred` **bayrağı korunur ve raporlanır**, ama checkpoint front'un **medyan epoch'u**
   olur. Front üyelerinin hepsi zaten converged ∧ plato ∧ non-dominated — aralarında etiketsiz
   tercih yok, ama hepsi geçerli aday. Medyan deterministik ve `pareto_eps`'e front'un tam
   büyüklüğünden çok daha az duyarlı (L23'ün duyarsızlık problemini kısmen hafifletir).
3. **Boş havuz ≠ geniş front.** Boş havuzda seçilecek aday yoktur (bir tie-break değil, bir yokluk),
   `last.pth` orada meşru. Üç değerli karar alanı: `selected` / `deferred` / `fallback`.
4. `median_low` kullan, ortalama değil: seçilen epoch'un diskte bir checkpoint'i olmak zorunda.

**Yan bulgu (aynı turda ölçüldü).** Kapısız argmax ("literatüre sadık" naive kol) üç r18 seed'inde
de, üç metrikte de **epoch 4**'ü seçiyor: val_loss ≈ −0.13…−0.18, `converged=False`, yani neredeyse
eğitilmemiş ağırlıklar. L1'de teorik olarak öngörülen başarısızlık modu artık ölçülmüş bir baseline.

**Uygulandı.** `select_epoch.front_representative` / `select` / `copy_epoch` / `--baselines`;
`CHECKPOINT_SELECTION_METHODOLOGY.md` §8 (ön-kayıt). **Kaynak.** L21, L23, [[paper-scope]].

## L26 — Determinizm bir optimizasyon lisansıdır: eval view'ı iki kez okuma (2026-07-28)

**Bağlam.** Validation maliyeti bütçeyi belirliyor: ölçüldü, **bir eval ≈ 2.1 training epoch**
(367 s vs 173 s). Bu yüzden `val_schedule` bir "ölçüm sıklığı" değil, plato çözünürlüğünü duvar
saatiyle takas eden bir karar (L22). Soru "her epoch validate edebilir miyiz" olunca maliyetin
nereden geldiğine bakıldı.

**Bulgu.** `validate()` `eval_ld`'yi **iki kez** dolaşıyordu — önce `recompute_bn_stats` (precise-BN),
sonra feature çıkarımı. Ama `mode="eval"` **deterministik**: tek merkezlenmiş crop, augmentasyon yok,
`_view(..., training=False)` RNG'ye hiç dokunmuyor (tümör crop'u yalnız `training=True` iken jitter
uyguluyor). Yani 200 deneğin 4 modalitesi **birebir aynı tensörler için** iki kez gzip'ten açılıyordu.
Validation'ın hacim okumasının **üçte biri** buydu (600 → 400 okuma).

**Ders.**
1. **Determinizmi önce kanıtla, sonra kullan.** İki bağımsız geçişin `torch.equal` ile aynı çıktığı
   gerçek deneklerde doğrulandı; ancak ondan sonra materialize etmek "numerik olarak eşdeğer"dir.
   Kanıtlanmamış determinizm varsayımıyla yapılan cache, sessizce farklı sonuç üretir.
2. **Ölçmeden optimize etme, ölçtükten sonra da tahmine güvenme.** Tahmin ~250 s idi; probe 3 eval'de
   230/350/170 s verdi (ortalama 250 s, ama **contention'a göre 2 kat oynuyor**). Tek ölçüm alsaydık
   170 de 350 de yanıltıcı olurdu.
3. **Maliyeti kalıcı olarak logla.** `[val NNN] ... time=XXXs` eklendi — cadence kararı artık her
   run'da denetlenebilir, iş bittikten sonra duvar saatinden geri-hesaplanması gerekmiyor.
4. `recompute_bn_stats` zaten herhangi bir iterable kabul ediyordu; DataLoader yerine materialize
   edilmiş liste vermek imza değişikliği gerektirmedi. **Gevşek tip beklentisi burada işe yaradı.**

**Sonuç.** Eval 367 s → **250 s (%32)**. `val_every: 1` 15.1 h → **11.9 h/run**; 5 run için
76 → **59 GPU-saat**. Bellek: MaxRSS 46 GB / 120 GB (materialize edilen tensörler ~7 GB).
Maliyet: ~45 dk uygulama + probe.

**Uygulandı.** `train_simsiam.validate` (`eval_batches = list(eval_ld)`, `pin_memory=False`),
`utils.recompute_bn_stats` docstring, `[val]` satırına `time=`. Smoke exit 0.
**Kaynak.** L22, `logs/probe_evalcost_10087.out`.

## L27 — Ölçüm bütçesini bir probe'dan çıkaramazsın; ve bir koşuyu elerken gerekçe sonuçtan ÖNCE yazılır (2026-07-29)

**Bağlam.** `val_every: 1`'e geçme kararı, tek işlik bir probe'da ölçülen **250 s**'lik eval
maliyetine dayanıyordu (L26). Altı iş birden koşunca gerçek maliyet **493 s**'ye çıktı ve tahmin
11.9 h/run yerine 19.4 h/run oldu — deadline'ı 4 saat aşacak şekilde.

**Ders 1 — contention bir ölçüm parametresidir.** Aynı kod, aynı kohort, aynı node tipi:
ai01'de 2 ağır iş varken eval **355 s**, ai02'de 4 ağır iş varken **~650 s**. Fark tamamen paylaşımlı
NFS'ten gzipped NIfTI okuma yarışı. Yani bir probe'dan çıkarılan "per-run maliyet" ancak **probe'un
koştuğu yük altında** geçerlidir; üretim yükünü temsil etmiyorsa plan yanlış çıkar. Bütçe ölçümü
hedef eşzamanlılıkta yapılmalı.

**Ders 2 — ortak-epoch kesme, `last.pth`'ten daha kontrollüdür.** Koşular farklı hızlarda
ilerlediğinde `last.pth` her seed'de başka bir epoch demektir; onu "final/plateau" kolu diye
raporlamak, kolun içine seed'e göre değişen bir eğitim miktarı gömer. Ortak bir `ckpt_eNNN.pth`
seçmek hem bu karışıklığı kaldırır hem de erken kesmeyi meşrulaştırır — yeter ki plato gerçekten
oturmuş olsun. Burada oturmuştu: epoch 40–52 arası `val_loss` aralığı 0.0047 (ön-kayıtlı
`plateau_tol` = 0.01'in altında), RankMe 6.38–6.91.

**Ders 3 — bir koşuyu elerken sınırlayıcı büyüklük "ortak epoch"tur, ortalama değil.**
Ortak kesme epoch'unu **en geriden gelen koşu** belirler. seed47 (ai02'de 4 iş arasında sıkışmış,
e38) tutulursa ortak epoch 49 ve Pareto plato havuzu **~9 nokta**; çıkarılırsa 69 ve **~29 nokta**.
L23 havuz 13–16 nokta iken kuralın kendi hiperparametrelerine kararsız olduğunu ölçmüştü — 9 nokta
o eşiğin altındadır. Yani "6 zayıf seed" değil **"5 sağlam seed"** doğru seçim: seed sayısı tek
başına değil, her seed'in taşıdığı karar-verilebilirlikle birlikte anlamlı.

**Ders 4 — eleme gerekçesi sonuçtan önce yazılır, yoksa savunulamaz.** seed47 hatalı değil
(logundaki 1360 "hata" satırının hepsi bilinen zararsız `/tmp/pymp` temizlik gürültüsü); yalnızca
SLURM onu yoğun node'a yerleştirmiş. Ölçüt **node yerleşimi kaynaklı ilerleme hızı**, hiçbir metrik
ya da AUC değeri değil. Bu ayrım ancak **kayıt tarihi sonuçların üretim tarihinden önceyse**
denetlenebilir; bu yüzden karar `tasks/todo.md` §C.4'e downstream hiç koşulmadan yazıldı.
Sonuçlara bakıp seed elemek cherry-picking'dir; operasyonel eleme değildir — ama ikisini ayıran
tek şey kayıttır.

**Kaynak.** L23 (Pareto havuz büyüklüğü), L26 (probe ölçümü), `tasks/todo.md` §C.4.

## Kaynak etiketleri
`MODEL_SELECTION.md` ve `DESIGN_JUSTIFICATION.md` sonundaki listeyle aynı.

## L28 — Danışman dokümanı sonuç kataloğu değil, karar zinciri olmalıdır (2026-07-30)

**Bağlam.** İlk teknik doküman tabloları, policy tanımları ve figürleri içeriyordu; ancak okuyucuya
her deneyin hangi soruyu yanıtladığını ve şekillerden hangi sonucun çıkarılabileceğini sistematik
olarak göstermiyordu. Ayrıca Markdown/LaTeX yazımı PDF üreticisinde gerçek matematik olarak
işlenmediğinden bazı formüller görsel olarak bozuluyordu.

**Kural.** Danışman veya hakem için hazırlanan her sonuç bölümünde şu sıra korunur:
(1) deney sorusu, (2) sabit tutulanlar ve karşılaştırılanlar, (3) tablo/figürün nasıl okunacağı,
(4) verinin desteklediği sınırlı çıkarım ve (5) belirsizlik/sınır. Teknik terimler English kalır;
açıklama metni Türkçe yazılır. PDF motoru matematik render etmiyorsa LaTeX kaynak metni bırakmak
yerine Unicode ile basılabilen, açık eşdeğer ifade kullanılmalıdır.
