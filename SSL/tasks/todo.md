# Bildiri Deney Planı — SSL Temsil-Seçim Stratejisi × Downstream IDH

## Aktif düzeltme — deterministic downstream evaluation ve final paired inference (2026-07-30)

## Secondary robustness analysis — training-fold-only SMOTE (2026-07-31)

- [x] **S.1** Frozen 256-D encoder feature'ları için, yalnız outer-CV training fold'unda çalışan
      deterministik bir SMOTE kolu ekle. Test fold'a, encoder pretraining'e veya label-free
      checkpoint selection'a sentetik örnek girmemeli; önce training-fold `StandardScaler`, sonra
      SMOTE uygulanmalı. **Doğrulama:** sentetik testte 15/5 sınıf sayısı 15/15'e çıktı; ilk 20
      özgün örnek bit düzeyinde değişmeden kaldı; OOF skor ve threshold'lar sonlu.
- [x] **S.2** SMOTE kolunda `class_weight=None` kullan; mevcut `class_weight="balanced"` kolunu
      primary referans olarak değiştirme ve iki dengeleme yöntemini aynı kolda birleştirme.
- [x] **S.3** Beş canonical seed ve common gated/SPS/Pareto için cache-only downstream skorlamayı,
      ardından seed-level paired AUC analizini üret. Çıktılar primary tablolardan ayrı tutuldu:
      `results/paper/smote_paired_selection_{per_seed,summary}.csv`. Exact two-sided Wilcoxon:
      SPS−gated $\Delta$AUC $=+0.00750$, $W=7$, $p=1.00$; Pareto−gated $=-0.01112$, $W=6$, $p=0.8125$.
- [x] **S.4** Leakage ve sınıf sayısı kontrolleriyle sentetik test doğrulaması yap; sonuçları
      primary değil secondary robustness analysis olarak raporla.

- [x] **R.1** `idh_probe.py` precise-BN loader'ını deterministic yap: sabit subject sırası ve
      `drop_last=False`; eski stochastic feature cache'lerini yeni bir cache sürümüyle geçersiz kıl.
- [x] **R.2** Aynı run içindeki aynı `pretrain_epoch` için ortak canonical feature cache kullan;
      böylece byte-identical RankMe/LiDAR/AlphaReQ checkpoint'leri aynı feature/prediction'ı verir.
- [x] **R.3** Beş canonical seed için SPS ile common gated checkpoint arasındaki farkı hesaplayan
      final analiz ekle: subject-level paired hierarchical bootstrap ile %95 CI; seed-level exact
      two-sided sign-flip permutation ve exact two-sided Wilcoxon signed-rank test ile p-value.
      Seed ve subject eşleşmesi korunacak.
- [x] **R.4** Yeni cache'lerle canonical downstream evaluation'ı yeniden çalıştır; ana sonuç ve
      paired-statistics CSV'lerini üret, aynı checkpoint policy'lerinin eşitlendiğini doğrula.
      **Doğrulama:** 15 adet `detbn-v2` cache üretildi; RankMe/LiDAR/AlphaReQ, beş seed'in her
      birinde aynı epoch ve tüm downstream metriklerde tam eşit çıktı.
- [x] **R.5** Methods/Results yazımında kullanılacak final prosedürü ve sayıları yalnız bu yeniden
      üretilmiş artefact'lardan al; eski per-checkpoint stochastic cache sonuçlarını kullanma.
- [x] **R.6** Threshold-dependent metrikleri (ACC/F1/Sensitivity/Specificity/BalAcc) yalnız ilk CV
      repeat'inden değil, tüm 5 repeated stratified CV repeat'i üzerinden ortala; deterministic
      feature cache'den CPU-only yeniden analiz ile kanonik tabloları güncelle. AUC ve paired
      delta-AUC değişmedi; yalnız threshold-dependent summary değerleri güncellendi.

**KAPSAM KİLİTLENDİ (2026-07-28)** — kaynak: `Proposed Experimental Framework` belgesi + kullanıcı kararı.
Bu bölüm önceki tüm kapsam kararlarını (2026-07-23 "manşet=yöntem, 3 backbone" dahil) **ezer**.

## Ana araştırma sorusu

> Önerdiğimiz etiketsiz temsil-seçim stratejisi (**Stable Plateau Selection, SPS**), mevcut
> SSL seçim yöntemlerinden (RankMe / LiDAR / AlphaReQ) downstream IDH sınıflandırması için
> daha bilgilendirici bir temsil seçiyor mu?

Karşılaştırma **tek değişkenli**: aynı veri, aynı backbone, aynı downstream classifier,
aynı CV split'leri, aynı protokol. Yalnız **checkpoint seçim kuralı** değişir.

## Kilitli tasarım

| Eksen | Karar |
|---|---|
| Backbone | **ResNet18 — tek.** (r34/densenet121 kapsam dışı; bu bildiride bahsedilmez.) |
| SSL yöntemi | SimSiam (değişmez), `feature_dim=256` (değişmez) |
| Pretraining kohortu | `splits/splits_pretrain_noucsf_v200.json` — 995 train / 200 val, UCSF **hariç** |
| Downstream kohortu | UCSF-PDGM, n=500 (1 bozuk hacim hariç, `splits/ucsf_excluded_subjects.txt`) |
| Downstream protokolü | **Donuk** encoder → precise-BN → 256-d feature → head; 5-fold × 5-repeat stratified CV |
| Head | **Linear SVM** — tek head, 256 özelliğin hepsi, feature selection yok |
| Seed | ResNet18 × **5 seed** (42–46); seed47 ön-kayıtlı operasyonel gerekçeyle dışlandı (§C.4) |
| Metrikler | AUC, Accuracy, Sensitivity, Specificity, F1, Balanced Accuracy |
| İstatistik | Seed-içi **paired** δAUC + %95 bootstrap CI; seed'ler üzerinden mean ± SD (asla en iyi seed) |

## Karşılaştırma merdiveni (6+1 kol) — ÖN-KAYITLI

Ön-kayıt gerekçeleri: `braintumor_ssl/CHECKPOINT_SELECTION_METHODOLOGY.md` §8.

| # | Kol | Tanım | Rolü |
|---|---|---|---|
| 0 | **Random-init** | eğitilmemiş ResNet18, aynı ön işleme + precise-BN | taban (floor) |
| 1a | **RankMe-naive** | kapı YOK, tüm eval epoch'ları üzerinde argmax RankMe | literatüre sadık uygulama → başarısızlık modu |
| 1b | **LiDAR-naive** | kapı YOK, argmax LiDAR | aynı |
| 1c | **AlphaReQ-naive** | kapı YOK, argmin \|α−1\| | aynı |
| 2a | **RankMe-selected** | yakınsama+çöküş kapısı + argmax | *steel-man* existing baseline |
| 2b | **LiDAR-selected** | aynı kapı + argmax | aynı |
| 2c | **AlphaReQ-selected** | aynı kapı + argmin \|α−1\| | aynı |
| 3 | **Proposed (SPS)** | kapı YOK; üç metrik de yerleşik değerinin %5'i içinde, 3 ardışık eval → ilk böyle epoch | **KATKI** (2026-07-29) |
| 3b | Pareto (ablasyon) | 2 + plato kapısı + SE-duyarlı Pareto + front-medyan | eski Proposed |
| 4 | **Last epoch** | seçim yok, eğitimin son epoch'u (99) | referans |

Neden hem naive hem gated: Proposed'ı yalnız naive'e karşı kıyaslamak straw-man olur;
yalnız gated'a karşı kıyaslamak ise kapının neden gerekli olduğunu göstermez.

---

## Faz A — Kapsam kilidi + ön-kayıt (kod yok)

- [x] **A.1** r18 harici koşular kapatıldı (`scancel 10079_3..8`); yalnız ResNet18 seed 42/43/44 koşuyor.
- [x] **A.2** `tasks/todo.md` bu belgeye göre yeniden yazıldı (bu dosya).
- [x] **A.3** `CLAUDE.md` — "IDH downstream kapsam dışı" ifadesi düzeltildi; kapsam + merdiven eklendi.
- [x] **A.4** `START_HERE.md` — güncelleme başlığı + §1 kapsam paragrafı yeni çerçeveye alındı.
- [x] **A.5** `CHECKPOINT_SELECTION_METHODOLOGY.md` §8 — merdiven ve front-medyan tie-break **sonuçlara
      bakmadan** ön-kayıt edildi.
- [x] **A.6** Doküman↔config sapmaları düzeltildi: `lidar_views` (8→4 fiili), `areq_r2_min` (0.95→0.80,
      gerekçesiyle), `val_every` ↔ `val_schedule`.

## Faz B — Kod (GPU gerekmez; feature'lar `features/idh/` altında cache'li)

- [x] **B.1** `select_epoch.py`: front > 1 → `last.pth` yerine **front'un medyan epoch'u**
      (`front_representative`, `median_low` → her zaman diskte var olan bir epoch). Karar alanı artık
      üç değerli: `selected` (front=1) / `deferred` (front>1, medyan seçildi) / `fallback` (havuz boş
      → `last.pth`). Gerekçe: eski davranış Proposed kolunu kısmen Last-epoch kolunun kopyası yapıyordu.
      **Doğrulandı:** 6 sentetik vaka (tek front, tek/çift geniş front, boş havuz, plato kapısı,
      R² tabanı) — hepsi geçti.
- [ ] **B.2** `select_epoch.py --write` → `bestPareto.pth`. **C.1'e BAĞLI (blocked).** Eski
      `_seed4?` run'ları yalnız her 10 epoch'ta checkpoint yazmış; kural e24/e34 diyor ama
      `ckpt_e024.pth` yok. Kod bunu temiz raporluyor ve **yanlış dosya yazmıyor** (doğrulandı).
      `_dense` run'larında her validate edilen epoch'a checkpoint yazılıyor → orada çalışacak.
- [x] **B.3** Naive (kapısız) argmax kolları: `select_epoch.py --baselines [--write]` →
      `bestNaive{RankMe,LiDAR,A-ReQ}.pth`. **Doğrulandı (bulgu):** üç r18 seed'inde de üç metrik de
      **epoch 4**'ü seçiyor (val_loss ≈ −0.13…−0.18, `converged=False`) — yani neredeyse eğitilmemiş
      ağırlıklar. Kapının neden gerekli olduğunun doğrudan kanıtı. (Materialize `_dense` run'larını
      bekliyor; `ckpt_e004.pth` yalnız orada var.)
- [x] **B.4** `idh_probe.py`: **linear SVM head** artık **varsayılan** (`--head svm`, `LinearSVC`);
      `positive_scores` proba varsa onu, yoksa decision function'ı kullanır; SVM marjinleri
      tekrarlar arası ortalama alınmadan önce rank-normalize edilir (olasılık head'lerinin sayıları
      **değişmedi**). Clinical baseline'lar da aynı head'i kullanıyor. `CHECKPOINTS` listesi 8 kola çıktı.
      **Doğrulandı:** cache'ten skorlama çalışıyor, sıralama logistic ile tutarlı
      (bestRankMe 0.692 / last 0.796 / age 0.903).
- [x] **B.5** `pretrain_epoch` artık doluyor (9/9/9/99 okundu) → `paper_auc_table --check` kuralların
      aynı epoch'u seçtiğini tespit edebiliyor. Tam yeniden skorlama Faz D'de.
- [x] **B.6** `scripts/smoke_test.sh` **exit 0** — regresyon yok.

> ⚠️ **Faz D için not (şimdi çözülmesin, unutulmasın).** `paper_tables` / `paper_auc_table`
> `results/idh_probe*.csv`'i glob'luyor ve satırları `(run, checkpoint, arm)` ile dedup ediyor.
> Eski `simsiam_r18_unified_seed42` ile yeni `simsiam_r18_unified_seed42_dense` **farklı run adları**
> olduğu için ikisi de sayılır → resnet18 hücresi n=3 yerine n=6 raporlar. Faz D'de ya `--csv`
> yalnız `_dense` sonuçlarına daraltılmalı ya da bir run filtresi eklenmeli.

## Faz C — Koşular (GPU) — **6 seed, `val_every: 1`, `SUFFIX=_full`**

Karar (2026-07-28): 42-eval'lik `val_schedule` yerine **her epoch validation**, ve 5 yerine **6 seed**
(6 GPU açık, ek duvar saati sıfır). `_dense` koşuları iptal edilip protokol tekdüze hale getirildi —
karışık protokollü bir seed setini "all other settings identical" iddiasıyla savunmak mümkün değil.

- [x] **C.0** Eval maliyeti %32 düşürüldü (L26): 367 s → **250 s**. `val_every: 1` 15.1 h → **11.9 h/run**.
      Doğrulama: determinizm testi (`torch.equal`, max fark 0.0) + smoke exit 0 + GPU probe (job 10087,
      MaxRSS 46 GB / 120 GB). `[val]` satırına kalıcı `time=` eklendi.
- [x] **C.0b** Launcher/zincir 6-seed r18'e göre güncellendi: `sbatch_pretrain.sh` (array 0-5%6,
      CONFIGS=r18, SEEDS=42–47), `watch_downstream.sh` (varsayılan liste + `SUFFIX`),
      `finalize.sh` (`RUNS` ve `--probe_glob` `SUFFIX`'e kilitli — eski/`_dense` koşuların
      seed sayımına sızmasını önler), `configs/simsiam_r18_unified.yaml` (`val_every: 1`).
- [x] **C.0c** `idh_probe --sweep_stride` (varsayılan 10): `val_every: 1` her epoch'a checkpoint
      yazdığı için strid'siz sweep 100 encoder × 500 denek = saatlerce GPU demekti.
- [x] **C.1 İPTAL EDİLMEDİ — gerekmedi.** `scancel` ajan tarafında permission classifier'a takıldı,
      kullanıcı da makine başında değildi. Çözüm: iptal etmek yerine 6'lık array **kuyruğa** gönderildi.
      3 boş GPU'da hemen başladı, kalan 3 task eski işler bitince devralıyor. Bedeli ~3 saat gecikme;
      karşılığında `_dense` koşuları da tamamlanıyor (42-eval protokolü, "ızgara sıklığı seçimi
      değiştiriyor mu?" karşılaştırması için ikinci veri noktası).
- [x] **C.2** `SUFFIX=_full sbatch scripts/sbatch_pretrain.sh` → **job array 10088**, 6 task.
      10088_0/1/2 (seed 42/43/44) ai02'de koşuyor, `val_every: 1` log'dan doğrulandı;
      10088_3/4/5 (seed 45/46/47) `PD (Resources)`.
- [x] **C.3** Watcher çalışıyor (`logs/watch_downstream.out`, 6 `_full` run izleniyor, 300 s poll).
      Her run bitince otomatik: naive+Pareto materialize → report_run → idh_probe (SVM,
      `--sweep_stride 10`) → logistic robustness; altısı da bitince `finalize.sh`.
### C.4 — ÖN-KAYIT: ortak epoch'ta kesme + seed47'nin dışlanması (2026-07-29 07:44)

> **Bu blok, bu koşuların HİÇBİR downstream/etiketli sonucu üretilmeden önce yazıldı.**
> Gerekçesi tamamen operasyoneldir; sonuçlara bakılarak alınmış bir karar değildir.

**Durum (07:44).** `val_every: 1` altında eval maliyeti node yüküne çok duyarlı çıktı:
ai01'de 2 ağır iş → eval 355 s; ai02'de 4 ağır iş → eval ~650 s. Ölçülen epoch'lar:

| seed | job | node | epoch | s/epoch |
|---|---|---|---|---|
| 42 | 10088_0 | ai02 | 57 | 872 |
| 43 | 10088_1 | ai02 | 55 | 877 |
| 44 | 10088_2 | ai02 | 57 | 812 |
| 45 | 10088_3 | **ai01** | 61 | **528** |
| 46 | 10088_4 | **ai01** | 61 | **528** |
| **47** | **10088_5** | ai02 | **38** | 810 |

**Karar 1 — koşular ortak bir epoch'ta kesilir.** Plato zaten oturmuş: seed42'de epoch 40–52
arası `val_loss` aralığı 0.0047, yani ön-kayıtlı `plateau_tol`un (0.01) altında; RankMe 6.38–6.91.
Eski koşularda e55→e99 arası RankMe 6.66→6.42 idi. Epoch 99'a kadar koşmanın hiçbir sonucu
değiştirmediği ölçülmüştür. Kesme sonrası **"final/plateau" kolu `last.pth` DEĞİL, ortak
`ckpt_eNNN.pth`** olur — bu daha kontrollüdür, çünkü `last.pth` her seed'de farklı epoch'a denk gelir.

**Karar 2 — seed47 analizden çıkarılır.** Ortak epoch en geriden gelen koşu tarafından belirlenir:

| | seed | ortak epoch | Pareto plato havuzu |
|---|---|---|---|
| seed47 dahil | 6 | 49 | **~9 nokta** |
| **seed47 hariç** | **5** | **69** | **~29 nokta** |

Havuz büyüklüğü kritiktir: L23'te plato havuzu 13–16 nokta iken Pareto kuralı kendi
hiperparametrelerine kararsız çıkmıştı (ızgarada 0/9–7/9 kesin seçim). 9 nokta bunun altındadır ve
bildirinin ana katkısını ölçülemez hale getirir; 29 nokta eski koşuların iki katıdır.

**Gerekçenin niteliği.** seed47 hatalı ya da "kötü sonuç veren" bir koşu değildir — sağlıklıdır
(loglarındaki 1360 "hata" satırı bilinen zararsız `/tmp/pymp` temizlik gürültüsüdür), yalnızca
SLURM tarafından ai02'ye, üç ağır işin yanına yerleştirilmiştir. Dışlama ölçütü **node yerleşimi
kaynaklı ilerleme hızıdır**, hiçbir metrik ya da AUC değeri değildir. Bildiride "6 koşu başlatıldı,
5'i analiz edildi" olarak, bu gerekçeyle raporlanır.

- [x] **C.4a** Karar kaydedildi (bu blok + lessons **L27**), sonuçlardan önce.
- [ ] **C.4b** `scancel 10088_5` — kullanıcı çalıştıracak.
- [ ] **C.4c** ~11:30'da kalan 5 koşu durdurulur; ortak epoch = min(erişilen epoch) belirlenir.
- [ ] **C.4d** Downstream: `last.pth` yerine ortak `ckpt_eNNN.pth` "final/plateau" kolu olarak
      geçirilir (`idh_probe --checkpoints`; `paper_tables.POLICY_NAME` eşlemesi güncellenir).
- [ ] **C.4e** 5 downstream işi paralel → `finalize.sh`.

## Faz D — Analiz ve çıktı

- [ ] **D.1** Tablo 1 (etiketsiz): kol × seçilen epoch + RankMe/LiDAR/α-ReQ, mean±SD over seeds.
- [ ] **D.2** Tablo 2 (ana): 9 kol × {AUC, Acc, Sens, Spec, F1, BalAcc}, mean±SD over seeds.
- [ ] **D.3** Tablo 3 (paired): her kol vs random-init ve vs RankMe-selected → δ ± %95 CI, seed-içi paired.
- [ ] **D.4** Context tablosu: age / grade / age+sex clinical baseline (belgede yok, hakem soracağı için zorunlu).
- [ ] **D.5** Figürler: (a) üç metriğin epoch yörüngesi + her kuralın seçtiği nokta, (b) downstream AUC
      vs pretraining epoch (`--epoch_sweep`), (c) kol × AUC bar + seed CI.
- [ ] **D.6** Seçim kararlılığı: `selection_stability.py` (henüz hiç koşulmadı — `finalize.sh STABILITY=1`).

## Faz E — Yazım

- [ ] **E.1** `PAPER_*` taslakları eski "backbone comparison" çerçevesinden bu çerçeveye alınır.
- [ ] **E.2** Limitations: clinical confound (age tek başına AUC ≈ 0.90), n=200 < d=256, tek kohort/tek görev,
      Pareto kuralının hiperparametre duyarlılığı (L23).

## Doküman — System Design (2026-07-30)

- [x] **SD.1** System Design şablonunun yapısı, referans dosyası ve mevcut danışman dokümanı incelendi.
- [x] **SD.2** Checkpoint-selection ve evaluation sistemi, kilitli deney planına göre şablon içinde belgelendi.
- [x] **SD.3** Referans değiştirilmeden yeni `.docx` çıktısı üretildi. Ortamda Word/LibreOffice renderer bulunmadığından eşlik eden PDF üretilemedi.
- [x] **SD.4** Metin, tablo, görsel, placeholder, `.docx` paket ve referans-layout bütünlüğü doğrulandı.

**SD review.** Çıktı: `docs/SPS_SYSTEM_DESIGN.docx`. Şablonun 12 bölümü, 9 tablosu, sayfa ölçüsü,
marjinleri ve mimari-görsel alanı korunmuştur. Sistem, selection kararının label-free olması,
exact replay (20/20) ve SPS stability limitation’ı ile birlikte; son deney planı da M1–M4 altında
yer alacak biçimde belgelenmiştir. Görsel PDF render, bu çalışma ortamında DOCX renderer eksikliği
nedeniyle yapılmamıştır; Word veya LibreOffice bulunan bir ortamda açılarak son tipografik kontrol
yapılmalıdır.

## Doküman — imbalance ve Spearman kapsamı (2026-07-30)

- [x] Downstream `class_weight=balanced`, stratified CV ve training-fold Youden threshold açıklaması proje anlatımına eklendi.
- [x] Spearman’ın canonical beş ResNet-18 seed’i ve üç metric çifti için bulunduğu; policy veya downstream AUC korelasyonu olmadığı açıklandı.
- [x] Aynı ayrım System Design belgesinin downstream contract’ına işlendi.

## Analiz — ROC ve encoder CKA (2026-07-30)

- [x] Beş canonical ResNet-18 seed için RankMe, LiDAR, AlphaReQ ve SPS feature cache’leri üzerinden repeated-CV ROC curves üretildi.
- [x] Aynı 500 UCSF subject’inin 256-dimensional features’ı ile seed-içi linear CKA hesaplandı; sonra seed mean ± SD heatmap olarak özetlendi.
- [x] RankMe/LiDAR/AlphaReQ encoder `model` weights’lerinin her seed’de birebir eşleştiği doğrulandı; CKA≈1 beklenen sonuç olarak belgelendi.
- [x] ROC figure’ü, CV/held-out ayrımı ve yorum sınırlarıyla proje anlatımına eklendi. CKA heatmap’i canonical dokümana dahil edilmedi: üç gated policy aynı encoder weights’ini seçtiği ve cache’e bağlı sayısal farklar içerdiği için ek bağımsız bilimsel kanıt sağlamıyor.

---

## Bilinen riskler (ölçülmüş, tahmin değil)

1. **Üç gated kural aynı epoch'u seçiyor** → `bestRankMe/bestLiDAR/bestA-ReQ` byte-identical (L21).
   Ana tablo bugün koşulsa 2a/2b/2c satırları aynı çıkar. Yoğun `val_schedule` (L22) bunu çözebilir
   ama garanti değil; çözmezse **bu bir bulgu olarak raporlanır** ve Proposed'ın ayrımı 1/2/3/4
   arasında ölçülür.
2. **Downstream AUC pretraining epoch'una duyarsız** (r18 sweep'te ~düz) → seçim kuralları arasında
   büyük fark beklenmemeli. Yedek iddia: seçim ortalamayı yükseltmese de **varyansı düşürüyor**
   (seçilmiş SD 0.024 vs `last.pth` SD 0.060).
3. **Pareto kuralı kendi hiperparametrelerine duyarlı** (L23). `plateau_tol=0.01`, `pareto_eps=1.0`
   ön-kayıtlı; duyarlılık ızgarası limitation olarak raporlanır, gizlenmez.

---

## Geçmiş (referans — kapsam dışı kalan işler)

Aşağıdakiler önceki kapsamda tamamlandı; yeni kapsamda **bildiriye girmiyor**, altyapı olarak duruyor.

- **Aşama 0 altyapı:** VICReg VC-reg (`models.vc_regularizer`), W-MSE whitening
  (`models.whitening_mse_loss`), büyük projector config'i, uzun-eğitim + güçlü-aug config'i,
  jackknife GA (`utils.jackknife_ci`), run-arası seçici (`select_model.py`). Hepsi doğrulandı,
  hepsi opsiyonel/kapalı. Gerekçe ve kalibrasyon uyarıları: lessons L10, L11.
- **Çok-backbone karşılaştırması:** 18 eski koşu (noucsf/withucsf) + 9 unified koşu. Bulgular
  lessons L13/L16'da; bu bildiride kullanılmıyor.
- **Kohort evrimi:** L8 → L12 → L14 → L17. Nihai hâl: tek kohort, UCSF pretraining'den tamamen hariç.
- **Ölçüm tabanı kalibrasyonu:** val 60 → 200, α fit penceresi, LiDAR view sayısı, val feature dump.
  Lessons L18; METHODOLOGY §6. Bu kısım yeni kapsamda da **geçerli ve gerekli**.
- **Seçim kuralı rafinasyonu:** trailing-mean + 1-SE jackknife eşiği, `aggregate_runs.py`,
  `selection_stability.py`. Lessons L19; METHODOLOGY §7. Yeni kapsamda da geçerli.


---

## Faz D — SONUÇLAR (2026-07-29, 5 seed × ResNet18, UCSF n=500, linear SVM, image kolu)

| Kol | AUC | Sens | F1 | BalAcc |
|---|---|---|---|---|
| ungated argmax × 3 | 0.660–0.661 | 0.30 | 0.32 | 0.58 |
| RankMe / LiDAR / AlphaReQ-selected | 0.730 ± 0.017 | 0.48 | 0.46 | 0.66 |
| **Proposed (SPS)** | **0.739 ± 0.024** | 0.56 | 0.49 | 0.69 |
| Pareto (ablasyon) | 0.722 ± 0.030 | 0.53 | 0.45 | 0.66 |
| last epoch (seçim yok) | 0.745 ± 0.021 | 0.57 | 0.50 | 0.69 |

**Paired δAUC** (`bestRankMe`'ye göre, 5 koşu): ungated **−0.064…−0.065 [tutarlı]** ·
LiDAR/AlphaReQ-selected −0.001 · SPS +0.005 ± 0.030 · Pareto −0.012 ± 0.043 · last +0.010 ± 0.030.

### Okuma

1. **Tek anlamlı etki yakınsama/plato kapısıdır: +0.065 AUC**, beş seed'in beşinde aynı işaret.
   Kapısız argmax her seed'de **epoch 0'ı** (val_loss ≈ −0.02, eğitilmemiş ağ) seçiyor.
2. **Kapıdan sonra hiçbir kural diğerinden ayrılmıyor** — SPS, Pareto, kapılı argmax ve
   "hiç seçim yapma" hepsi 0.72–0.75 bandında, seed SD'si ±0.02–0.03.
3. Dolayısıyla savunulabilir manşet **"daha iyi bir seçim kuralı"** değil:
   *3B tıbbi SSL'de etiketsiz rank metrikleri transferle ters korelasyonludur (RankMe 85 → AUC 0.66;
   RankMe 6.4 → AUC 0.745); onları maksimize eden her kural başarısız olur. Gereken karmaşık bir
   kural değil, bir plato/yakınsama kapısıdır.*
4. SPS'in Pareto'ya üstünlüğü performans değil **sadelik**: tek hiperparametre, kapı yok,
   yön bilgisi gerekmiyor, L23'teki duyarsızlık problemi yok.

- [x] **D.1–D.3** Tablolar üretildi (`results/paper/table{1,2}*.csv`), paired δAUC dahil.
- [x] **D.4** Clinical baseline opsiyonel; Limitations için tek sayı: yaş tek başına AUC ≈ 0.90.
- [ ] **D.5** Figürler (kol × AUC bar, metrik yörüngesi + seçim noktaları).
- [ ] **D.6** `selection_stability.py` henüz koşulmadı.

---

## Doküman revizyonu — danışman görüşmesi (2026-07-30)

- [x] Ana teknik doküman, sonuç listesinden deney-sorusu ve karşılaştırma mantığı odaklı anlatıya dönüştürüldü.
- [x] Veri kaynağı / cohort adı ayrıntıları dokümandan çıkarıldı; yalnızca deneysel ayrım ve leakage önlemi soyut düzeyde tutuldu.
- [x] Her canonical figure için “nasıl okunur?” bölümü eklendi; Spearman ρ’nin performance metric olmadığı açıklaştırıldı.
- [x] LaTeX-benzeri ve PDF’de bozulabilen formüller, Unicode matematik ile PDF-uyumlu biçime taşındı.
- [ ] Revize PDF’nin görsel ve metin doğrulaması.
