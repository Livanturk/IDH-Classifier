# Lessons — kararlar, düzeltmeler ve çıkarımlar

Her giriş: **Bağlam → Ne oldu → Çıkarım/Kural → Kaynak**. Amaç, aynı hatayı tekrarlamamak ve
bildiride her kararı gerekçeyle savunabilmek. Yeni bir düzeltme/karar geldikçe buraya eklenir.

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

## Kaynak etiketleri
`MODEL_SELECTION.md` ve `DESIGN_JUSTIFICATION.md` sonundaki listeyle aynı.
