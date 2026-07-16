# Tasarım Kararları ve Gerekçeleri

Bu belge, SSL pretraining aşamasında verilen tasarım kararlarını bilimsel/metodolojik
olarak gerekçelendirir (makalenin *Methods / Rationale* bölümü için). Her karar için:
**Karar → Gerekçe → Alternatifler neden değil → Kaynak/kanıt** verilmiştir. Kaynak
etiketleri belgenin sonundaki listeye atıftır.

---

## 1. SSL yöntemi: SimSiam

**Karar.** Encoder, SimSiam ile pretrain edilir (simetrik negatif-kosinüs, stop-gradient + predictor).

**Gerekçe.**
- **Negatif çift / büyük batch / memory-bank gerektirmez.** Kontrastif yöntemler (SimCLR,
  MoCo) ayrıştırma için çok sayıda negatif örnek, dolayısıyla büyük batch ya da kuyruk ister.
  3D MRI hacimleri bellek-yoğun olduğundan büyük batch pratik değil; SimSiam küçük batch'te
  stabildir [ChenHe2021].
- **Momentum encoder yok.** BYOL'a göre daha az hiperparametre ve bellek; SimSiam esasen
  "momentum'suz BYOL"dur ve benzer başarı verir [ChenHe2021, Grill2020].
- **Ayrımcı (discriminative) temsil.** Downstream görev bir sınıflandırma (IDH); SimSiam
  gibi instance-discrimination yöntemleri sınıflandırma-dostu, lineer-ayrılabilir temsiller
  üretir. Generatif/rekonstrüksiyon SSL (MAE, Models Genesis) piksel-düzeyi detaya odaklanır
  ve ViT-tabanlı MAE veri-açtır; ~1.2k denek bunun için küçüktür [He2022MAE, Zhou2021].
- **Collapse teorik olarak kontrol altında.** Stop-gradient + predictor asimetrisi çöküşü
  önler; ablasyonla gösterilmiştir [ChenHe2021].
- **Alan uyumu.** BraTS üzerinde 3D Siamese SSL'in radiomics'i tamamlayan deep feature ürettiği
  daha önce gösterilmiştir (docx'teki HongweiLi MICCAI 2021), bu da yaklaşımın uygunluğunu destekler.


**Alternatifler neden değil.** SimCLR/MoCo (büyük batch/negatif maliyeti), BYOL (momentum
encoder karmaşıklığı), MAE (data-hungry, ViT gerektirir) — hepsi bu veri/bellek rejiminde daha
dezavantajlı. [Chen2020, He2020, Grill2020, He2022MAE]

---

## 2. Backbone: 3D CNN (ResNet), R18 ↔ R50 ablasyonu

**Karar.** MONAI 3D ResNet; ablasyon ekseni olarak ResNet-18 (512-d) ve ResNet-50 (2048-d).

**Gerekçe.**
- **CNN'in indüktif önyargısı küçük veride avantaj.** Yerellik/translation-eşdeğerlik, ~1.2k
  denekte ViT'ten daha veri-verimli. ViT büyük ölçekli veri ister. [Hara2018, Dosovitskiy2021]
- **Kanıtlanmış + hazır.** 3D ResNet medikal görüntülemede standart ve MONAI'de yerleşiktir;
  gereksiz yeni mimari eklemekten kaçınıyoruz [Hara2018, Cardoso2022].
- **R18 vs R50 ampirik bir sorudur.** Kapasite ↔ veri/overfit dengesi veriye bağlıdır: R18
  (~34M) az veride daha güvenli; R50 (~47M) daha zengin temsil ama daha çok veri/bellek ister.
  Hangisinin daha iyi *transfer* ettiğini ölçmek için ikisini de eğitip karşılaştırıyoruz.

**Alternatifler neden değil.** ViT/Swin (veri-aç), DenseNet (bellek), özel mimari (docx:
"katkı yeni mimari değil"). Ayrıntılı mimari karşılaştırması: `Backbone_Karsilastirma.xlsx`.

### 2.1 Res3DNet (iScience 2026) — görev-özel supervised baseline

**Ne.** Res3DNet, IDH mutasyonunu whole-brain MRI'dan **tümör segmentasyonu gerektirmeden**
tahmin eden, **supervised, uçtan-uca** bir 3D-residual modeldir [Res3DNet2026]. 2537 hasta / 11
veri seti; external test AUC **0.872**, TCGA **0.912**; vanilla ResNet, I3D, transformer ve 4
radyoloğu geçmiştir.

**Bizim encoder'ımızın yerine geçmez.** Res3DNet IDH etiketleriyle eğitilir → bir **SSL yöntemi
değildir**; bizim etiketsiz SimSiam pretraining'imizin backbone'unu ikame etmez. Bu belgede
mimari-seçim rakibi olarak değil, **downstream baseline ve tasarım kanıtı** olarak konumlanır.

**Üç açıdan projeye katkısı:**
- **(a) Seg-free whole-brain tasarımını doğrular.** Res3DNet, segmentasyon kullanmadan, seg-bağımlı
  yaklaşımları geçmiştir → bizim varsayılan `brain` crop_mode'umuzu (bkz. §9) destekler ve
  WT-masking'in başarı için *şart olmadığına* literatür kanıtı sağlar. Bu, `brain` ↔ `tumor_margin`
  ↔ `tumor+mask` ablasyonumuzu daha da anlamlı kılar.
- **(b) Supervised karşılaştırma baseline'ı.** docx'in katkısı SSL+radiomics komplementerliğidir;
  Res3DNet saf-derin-öğrenme state-of-the-art'ı olarak, füzyon modelimizin kıyaslanacağı referanstır.
- **(c) Backbone referansı.** Vanilla ResNet'i geçen 3D-residual tasarımı, encoder backbone
  seçimimize (R18/R50 ötesi ince-ayar) ampirik referans olabilir.

*(Tam blok yapısı makale erişimi gerektirir; paywall.)*

---

## 3. Temsil boyutu = 256

**Karar.** Transfer edilen deep feature `h` 256 boyutlu.

**Gerekçe.**
- docx bunu sabitliyor ("Hasta × 256 Features").
- **Füzyon dengesi.** Downstream'de deep (256) + radiomics (1440) + clinical (2) birleşecek;
  256, radiomics'i sayısal olarak ezmeyecek ama anlamlı bilgi taşıyacak makul bir sıkıştırmadır.
- **Küçük downstream kohort (~400-500) için boyut laneti.** 2048-d ham backbone temsili bu
  örneklem büyüklüğünde overfit'e açıktır; 256 boyut sınıflandırıcı varyansını azaltır.

---

## 4. Girdi: 4 modalite (kanal-stack), beyin-bbox kırpma, 96³ patch

**Karar.** T1/T1ce/T2/FLAIR 4 kanal olarak yığılır; skull-stripped beyin sınırlayıcı kutusuna
kırpılır; eğitim view'ı 96³ rastgele patch'tir.

**Gerekçe.**
- **Modaliteler tamamlayıcı.** T1ce enhancing tümörü, FLAIR/T2 ödem/infiltrasyonu, T1 anatomiyi
  vurgular; çok-kanal giriş BraTS literatüründe standarttır [Menze2015, Baid2021].
- **Beyin-bbox kırpma.** Hacmin ~%83'ü arka plan (sıfır); beyne odaklanmak hesap israfını
  önler (kendi veri incelememizde nonzero oran ≈ 0.17).
- **96³ patch.** Tam hacim (240×240×155) 3D'de bellek/hesap açısından pahalı; 96³ bağlam ↔
  bellek dengesidir. FoV (128³) ayrı bir ablasyon eksenidir.

---

## 5. Hacim-başı, kanal-başı z-score normalizasyon (nonzero maske)

**Karar.** Her hacimde, her modalite için, yalnızca beyin (nonzero) vokselleri üzerinden
z-score normalizasyon.

**Gerekçe (ampirik).** Kendi incelememizde yoğunluklar modaliteler arası ~30× (T1 maks ≈150k,
T2 maks ≈4.5M) ve denekler arası çok değişkendi — MRI yoğunluğu keyfi birimdir. Normalizasyon
olmadan ağ, biyolojik yapı yerine ölçek farklarını öğrenir. Nonzero maske arka planı istatistiğe
katmaz. BraTS'te yerleşik pratik [Menze2015, Baid2021].

---

## 6. Augmentasyonlar (view başına iki bağımsız artırma)

**Karar.** Rastgele crop, 3-eksen flip, 90° rotasyon, küçük affine, Gaussian noise/smooth,
intensity scale/shift, gamma-kontrast, coarse dropout — orta şiddette.

**Gerekçe.**
- **SSL'in başarısı augmentasyon-invaryansına dayanır.** Kontrastif/Siamese öğrenme, iki view'ı
  "aynı örnek" kabul ederek görünüm-değişmez ama yapı-koruyan temsiller öğrenir; augmentasyon
  seçimi en kritik faktörlerden biridir [Chen2020, ChenHe2021].
- **Seçimlerin biyolojik anlamı.** Flip/rotasyon → yön-değişmezlik; affine → küçük kayıt/poz
  farkları; intensity scale/shift/gamma → tarayıcı/protokol varyasyonu; noise/smooth → edinim
  gürültüsü; coarse dropout → occlusion robustluğu.
- **Sadece düzlem-içi (axes 0/1) flip; z (superior-inferior) flip YOK.** z ekseni axial yöndür;
  onu flip'lemek beyni baş-aşağı çevirmek demektir (anatomik olarak implausible). L-R/A-P flip'ler
  makuldür ama yön-önyargısını azaltmak için olasılık 0.3'e indirildi.
- **Neden "orta" şiddet.** Doğal görüntü SSL'inden alınan ağır fotometrik augmentasyonlar
  (aşırı gamma, büyük dropout) medikal kontrastı bozup performansı düşürebilir; bu nedenle
  augmentasyon şiddetini bir ablasyon ekseni olarak tutuyoruz [Taleb2020].
- **Girdi rejimine göre iki preset (`aug_preset`: standard | gentle | auto).** SSL güçlü
  augment ister, ama girdi **WT-masked / tümör-only ROI** olduğunda güvenli güçlü augment
  *intensity/appearance* (scanner/protokol varyasyonunu taklit eder, tümörü kesmez);
  agresif *geometri* ise küçük maskeli tümörü bozar. Bu yüzden `gentle` preset 90° rotasyon
  ve coarse-dropout'u kaldırır, affine'i ~5°'ye indirir, ama intensity augment'i güçlü tutar;
  `standard` (whole-brain/bağlam ROI) güçlü geometriyi korur. `auto` maskeliyse `gentle` seçer.
  Bu, "aynı tümör → farklı scanner/intensity/noise altında aynı temsil" hedefini korur;
  Models Genesis'in appearance-bozma mantığıyla uyumlu ama SimSiam'a uyarlanmıştır [Zhou2021].

---

## 7. Optimizasyon: SGD + lineer LR-scaling + cosine + warmup, predictor sabit LR

**Karar.** SGD (momentum 0.9, wd 1e-4); `lr = 0.05 × batch/256`; 10-epoch warmup + cosine
decay; predictor LR decay edilmez.

**Gerekçe.**
- **SGD.** SimSiam SGD ile stabildir ve LARS gibi özel optimizer gerektirmez [ChenHe2021].
- **Lineer LR-scaling.** Batch büyüklüğüne göre lr ölçekleme büyük-batch eğitiminin standart
  reçetesidir [Goyal2017].
- **Cosine + warmup.** Cosine decay yakınsamayı iyileştirir [Loshchilov2017]; warmup erken
  instabiliteyi azaltır [Goyal2017]. **Ampirik destek:** kendi smoke testimizde warmup'sız ilk
  iterasyonlarda `conv1` ağırlık normu ~58× şişti — warmup ihtiyacını doğrudan gözlemledik.
- **Predictor sabit LR.** SimSiam ekinde predictor LR'ını decay etmemenin daha iyi olduğu
  gösterilmiştir [ChenHe2021].

---

## 8. Kayıp: simetrik negatif-kosinüs benzerliği + stop-gradient

**Karar.** `L = -½[cos(p1, sg(z2)) + cos(p2, sg(z1))]`.

**Gerekçe.** SimSiam'ın tanımı. **Stop-gradient, çöküşü önlemenin anahtar bileşenidir**; onsuz
temsil dejenere olur (ablasyonla gösterilmiş) [ChenHe2021]. Simetri (her iki yön) örnek başına
sinyali daha verimli kullanır.

---

## 9. Segmentasyon modele GİRDİ olarak verilmez (yalnızca crop lokalizasyonu)

**Karar.** `seg` hiçbir modda ağa girdi/kayıp olarak girmez. Ağın girişi **her zaman** 4 kanal
yoğunluktur. Segmentasyon yalnızca `crop_mode ∈ {tumor, tumor_margin}` seçildiğinde, 96³
kutunun **nereden alınacağını** (WT centroid/bbox) belirlemek için kullanılır — birkaç sayı
üretir, voxelleri tensöre girmez. Varsayılan `crop_mode=brain`'de seg hiç okunmaz.

**Gerekçe.**
- **Öğrenme sinyali etiketsiz kalır.** Kontrastif kayıp yalnızca augmente view'lar arasındadır;
  moleküler (IDH) etiket kullanılmaz. Tümör maskesi ucuz, otomatikleştirilebilir (nnU-Net) bir
  yapısal annotation'dır ve ROI seçimi için kullanımı SSL literatüründe yaygındır. docx'in
  "etiketsiz kohort" iddiası "moleküler etiket kullanılmaz" olarak ifade edilerek korunur.
- **Seg girdi kanalı olsaydı** encoder inference'ta maskeye bağımlı olur ve "SSL" iddiası
  zayıflardı; bunu bilinçle reddettik.
- **Field-of-view bir ablasyon eksenidir (komplementerlik gerekçesi).** Radiomics zaten
  tümör-içidir; deep feature'ları da tümöre hapsetmek onları radiomics'e **daha redundant**
  yapabilir. Whole-brain/peritümöral bağlam (`brain`) radiomics'in görmediği bilgiyi taşıyıp
  docx'in Katkı-3 (komplementerlik) hipotezini destekleyebilir; `tumor_margin` ise IDH ham
  doğruluğunu artırabilir. Hangisinin daha iyi transfer ettiği VE daha komplementer olduğu
  ölçülecek ampirik bir sorudur → `brain` ↔ `tumor_margin` ablasyonu.
  **Literatür kanıtı:** Res3DNet (§2.1) whole-brain + seg-free ile seg-bağımlı yöntemleri geçmiştir
  [Res3DNet2026] → `brain` (maskesiz) kolunun güçlü bir aday olduğunu destekler.
- **Tutarlılık.** crop_mode checkpoint'e yazılır; feature çıkarımı aynı crop_mode'u kullanır,
  böylece encoder pretraining ile aynı FoV'u görür.

**Not (inference bağımlılığı).** Tümör modları her kohortta seg gerektirir (UPenn/UCSF-PDGM'de
mevcut; ham harici kohortta nnU-Net ile otomatik üretilir). `brain` modu seg gerektirmez.

---

## 10. UPenn dahil/hariç ablasyonu

**Karar.** İki encoder üret: (a) tüm koleksiyonlarla, (b) UPENN-GBM hariç.

**Gerekçe.**
- **Sorun.** UPENN-GBM (403 denek) hem BraTS SSL setinde hem downstream IDH kohortunda.
- **Dahil etmek geçerli:** SSL etiketsiz olduğundan *etiket* sızıntısı yoktur ve SSL veri
  ölçeğinden faydalanır (daha çok pretraining verisi ⇒ genelde daha iyi encoder).
- **İtiraz riski:** Bir hakem "image-level leakage" (encoder downstream görüntüleri görmüş)
  itirazı yapabilir. **İkisini de üretip farkı ölçmek** bu itirazı ampirik olarak kapatır ve
  bir robustness/ablasyon bulgusu sağlar.
- **Nihai iddia zaten temiz:** docx harici test için **UCSF** kullanır; genelleme iddiası
  görülmemiş merkez üzerinden yapılır. [AdaBN motivasyonu: Li2018]

---

## 11. Değerlendirme: etiketsiz metrikler (şimdi) + lineer probe (nihai)

**Karar.** Config'ler önce etiketsiz metriklerle (RankMe, alignment/uniformity, collapse-z_std,
participation ratio) + t-SNE ile karşılaştırılır; IDH etiketi gelince lineer-probe/k-NN AUC
karar-verici olur.

**Gerekçe.**
- **Pretraining loss kıyaslanamaz/yanıltıcıdır.** Farklı augmentasyon/mimaride ölçekler farklı
  ve düşük loss collapse'ı bile gizleyebilir. Bu yüzden loss'a göre seçim yapılmaz.
- **RankMe** etiketsizdir ve downstream başarısıyla korelasyonu gösterilmiştir → ana eleyici
  [Garrido2023].
- **Alignment & Uniformity** SSL'in iki hedefini (view-invaryansı ve küre-üzerinde yayılım)
  doğrudan ölçer [Wang2020].
- **Lineer/k-NN probe**, SSL temsillerini değerlendirmenin altın-standart protokolüdür
  [Chen2020, ChenHe2021]; downstream *füzyon* modeli değildir (radiomics/klinik/attention yok),
  yalnızca "feature'lar IDH için lineer ayrılabilir mi?" sorusunu yanıtlar.
- **t-SNE/UMAP** görselleştirmesi docx'in Katkı-4 (interpretability) gereksinimini karşılar.
- **Şu an etiket yok** (repoda IDH tablosu yok) → etiketsiz katman + t-SNE ile ilerlenir.

---

## 12. Precise-BN / kohort-başı BatchNorm yeniden hesaplama

**Karar.** Feature çıkarımında (varsayılan açık) BatchNorm running-stats hedef kohort üzerinde
yeniden hesaplanır.

**Gerekçe (ampirik + literatür).**
- **Ampirik kanıt.** SimSiam kaybı L2-normalize (ölçek-değişmez) olduğundan backbone aktivasyon
  ölçeği ve BN running-stats kısıtlanmaz; bunu doğruladık — eğitim (batch-stat) modunda çıktı
  sağlıklıyken (ort ≈0.9) eval (running-stat) modunda ölçek bozulabiliyordu. Precise-BN bunu
  düzeltir [Wu2021].
- **Çok-merkezli transfer.** BraTS→UPenn→UCSF arasında domain shift vardır; BN istatistiklerini
  hedef kohorta uyarlamak (AdaBN) transfer robustluğunu artırır [Li2018].

---

## Ek: Epoch / batch

~1.2k denek küçük bir kohorttur; 200–400 epoch SSL için makul başlangıçtır ve etiketsiz
metriklerin platosuna göre kesilir. Batch büyüklüğü GPU belleğine göre ayarlanır; SimSiam küçük
batch'e toleranslıdır (bkz. §1) ve LR batch ile otomatik ölçeklenir (§7).

---

## Kaynaklar

- **[ChenHe2021]** Chen X., He K. *Exploring Simple Siamese Representation Learning.* CVPR 2021.
- **[Chen2020]** Chen T. et al. *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* ICML 2020.
- **[He2020]** He K. et al. *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo).* CVPR 2020.
- **[Grill2020]** Grill J.-B. et al. *Bootstrap Your Own Latent (BYOL).* NeurIPS 2020.
- **[He2022MAE]** He K. et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.
- **[Zhou2021]** Zhou Z. et al. *Models Genesis.* Medical Image Analysis 2021 (MICCAI 2019).
- **[Taleb2020]** Taleb A. et al. *3D Self-Supervised Methods for Medical Imaging.* NeurIPS 2020.
- **[Hara2018]** Hara K. et al. *Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?* CVPR 2018.
- **[Dosovitskiy2021]** Dosovitskiy A. et al. *An Image is Worth 16x16 Words (ViT).* ICLR 2021.
- **[Cardoso2022]** Cardoso M.J. et al. *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701, 2022.
- **[Menze2015]** Menze B.H. et al. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI 2015.
- **[Baid2021]** Baid U. et al. *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification.* arXiv:2107.02314, 2021.
- **[Res3DNet2026]** *Fully automated Res3DNet model to predict IDH mutation of gliomas from whole-brain MRI free of tumor segmentation.* iScience 29(1), 2026. https://www.sciencedirect.com/science/article/pii/S2589004225026343
- **[Goyal2017]** Goyal P. et al. *Accurate, Large Minibatch SGD.* arXiv:1706.02677, 2017.
- **[Loshchilov2017]** Loshchilov I., Hutter F. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
- **[Garrido2023]** Garrido Q. et al. *RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by their Rank.* ICML 2023.
- **[Wang2020]** Wang T., Isola P. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020.
- **[Wu2021]** Wu Y., Johnson J. *Rethinking "Batch" in BatchNorm.* arXiv:2105.07576, 2021.
- **[Li2018]** Li Y. et al. *Adaptive Batch Normalization for Practical Domain Adaptation (AdaBN).* Pattern Recognition 2018.
