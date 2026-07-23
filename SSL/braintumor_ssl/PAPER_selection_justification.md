# Seçim Algoritması — Gerekçelendirme ve Karşıt Görüşler (Discussion/Rationale)

> Bildirinin Tartışma/Gerekçe bölümü için. Her tasarım kararı **Karar → Mantığımız → Literatür →
> Karşıt görüş → Yanıtımız** yapısıyla savunulur. Yöntem taslağı: `PAPER_model_selection_section.md`;
> algoritma spesifikasyonu: `MODEL_SELECTION.md`. Atıflar belge sonundadır.

---

## 1. Neden birincil metrik RankMe?

**Karar.** Etiketsiz aşamada backbone seçiminin *birincil* ölçütü RankMe (temsilin effective rank'ı).

**Mantığımız.** IDH etiketimiz yok; joint-embedding SSL (SimSiam) girdiyi yeniden kurmadığından
başarılı/başarısız eğitimin görsel bir işareti de yok. Elimizdeki en iyi *etiketsiz* vekil, downstream
lineer-prob başarısıyla korelasyonu ampirik olarak gösterilmiş bir metrik olmalı — RankMe tam da budur:
hiperparametresiz, ölçek-bağımsız, yalnızca SVD.

**Literatür.** RankMe [Garrido2023] birçok SSL yöntemi (SimCLR, VICReg, Barlow, DINO) ve veri kümesinde
(ImageNet, iNaturalist, Places, Food101…) lineer-prob doğruluğunu izler; etiketli ImageNet-val ile
hiperparametre seçmenin performansının çoğunu **hiç etiket kullanmadan** geri kazanır ve makale
etiketsiz alanları (tıbbi görüntüleme, uydu) açıkça hedef gösterir.

**Karşıt görüş.** RankMe artık tek seçenek değil ve bazı halefleri onu **geçtiğini** iddia eder:
- **LiDAR** [Thilak2024] — RankMe'yi rafine eder; augmentasyon-değişmez yapıyı LDA-tarzı kullanarak
  lineer-prob'u RankMe'den daha iyi izlediğini raporlar.
- **α-ReQ** [Agrawal2022] — özdeğer spektrumunun güç-yasası üsteliyle transfer edilebilirliği kestirir;
  OOD'de RankMe'den bir tık iyi olabilir.
- **IdEst** [IdEst2026] — intrinsic-dimension tabanlı; RankMe/VICReg ile rekabetçi.
- Ayrıca RankMe, **dengesiz/karmaşık** görevlerde (anomali tespiti) downstream ile korelasyonunu
  yitirebilir [Otero2024].

**Yanıtımız.** RankMe'yi *bilinçli ve sınırlarını kabul ederek* seçiyoruz: (i) **hiperparametresiz ve
en yerleşik** olan odur — LiDAR/α-ReQ/IdEst ek yapı (sınıf/örnek grafiği), ek varsayım veya çok yeni
olmaları nedeniyle daha az hakemli/yaygındır; (ii) Otero eleştirisini **tek-metrik kullanmayarak**
(üçgenleme, §5) ve **nihai kararı denetimli prob'a bırakarak** (§6) doğrudan karşılıyoruz; (iii) RankMe
bir *ana* metrik ama *tek* metrik değil. **LiDAR'ı gelecek-iş olarak açıkça öneriyoruz** — mimarimiz
metrik-agnostik olduğundan RankMe'yi LiDAR ile değiştirmek tek satırlık bir değişikliktir.

## 2. Neden ağırlıklı toplam değil, katmanlı/leksikografik karar?

**Karar.** Metrikleri tek bir ağırlıklı skorda birleştirmiyoruz; zorunlu kapılar + tek birincil metrik
+ sıralı tie-break kullanıyoruz.

**Mantığımız.** (i) Etiketsiz ortamda ağırlıklar (w₁·RankMe + w₂·PR + …) **keyfi** olur — onları
kalibre edecek bir doğrulama sinyali yok. (ii) Çökme ve yakınsamama **ödünleşilebilir büyüklükler
değildir**; yüksek bir başka skorla "telafi" edilemezler — bunlar kapı (hard-requirement), ödünleşim
değil. (iii) Metrikler farklı ölçeklerde; tek toplama sokmak yeni keyfi normalizasyon seçimleri ekler.

**Literatür.** RankMe [Garrido2023] zaten *tek* bir ölçütle (rank) seçim yapılmasını önerir, ağırlıklı
bir bileşke değil. Leksikografik/öncelikli karar, çok-kriterli karar kuramında (lexicographic
preferences) yerleşik bir yaklaşımdır ve *bazı kriterlerin sıralamada mutlak öncelikli* olduğu
durumlar için uygundur.

**Karşıt görüş.** Çok-kriterli karar yazınında yaygın eleştiri: leksikografik/eşik-tabanlı kurallar
**kırılgan** olabilir — eşiğin hemen kenarındaki küçük bir değişim kararı çevirebilir; ağırlıklı skorlar
küçük ödünleşimleri yumuşakça tartabilir. Ayrıca hard eşikler bilgi kaybettirir (0.149 ile 0.151'i
niteliksel olarak farklı sayar).

**Yanıtımız.** Kırılganlık riskini iki şekilde yönetiyoruz: (i) eşikleri metriğin *sınırına* değil,
iki rejim arasındaki **boşluğun ortasına** koyuyoruz (ör. yakınsama alignment eşiği ~0.5 random-init ile
~0.1 yakınsamış arasının ortası; eval protokolünde 0.30) — böylece gürültü kararı çeviremez. (ii) Kapılar
*gerçekten* hard-requirement olan şeyleri (collapse, yakınsamama) kodlar; bunlar için "yumuşak ödünleşim"
zaten yanlış olurdu. Sıralamadaki asıl karar (RankMe) süreklidir; hard kısım yalnızca eleme kapılarıdır.

## 3. Neden RankMe'den ÖNCE yakınsama kapısı?

**Karar.** Bir checkpoint RankMe ile sıralanmadan önce yakınsamış olmalı (val_loss ≤ τ, alignment ≤ τ).

**Mantığımız.** Gözlemimiz: RankMe **rastgele başlangıçta en yüksektir** ve eğitimle düşer. Eğitilmemiş
encoder feature'ları uzaya gürültü gibi saçtığından effective rank yapay olarak yüksek çıkar. Kapı
olmadan salt max-RankMe kuralı, neredeyse-eğitilmemiş erken bir epoch'u "en iyi" seçer (bizde epoch 4).

**Literatür.** Bu, **boyutsal collapse** [Jing2022] ile tutarlıdır: kontrastsız SSL rank'ı eğitimle bir
alt-uzaya sıkıştırır; başlangıçtaki yüksek rank temsil kalitesi değil, eğitilmemişliğin artefaktıdır.
Yakınsamayı alignment ile ölçmek Wang & Isola'nın [WangIsola2020] çerçevesine dayanır (düşük alignment =
öğrenilmiş invaryans).

**Karşıt görüş.** "RankMe zaten downstream ile korelasyonlu; neden ek kapı? Belki erken-yüksek-rank
gerçekten daha genel bir temsildir." Ayrıca eşik eklemek [Garrido2023]'ün saf reçetesinden sapmadır.

**Yanıtımız.** Garrido, RankMe'yi *yakınsamış* modeller arasında hiperparametre seçimi için doğrular —
random-init'le yakınsamış modeli kıyaslamaz. Bizim val_loss/alignment ölçümlerimiz epoch 4'te modelin
invaryansı *hiç* öğrenmediğini (val_loss≈0, alignment≈0.5) gösterir; yani yüksek rank "zengin temsil"
değil, "iki görünümü henüz eşleştirememiş" demektir. Kapı, RankMe'yi *geçerli olduğu rejimde* (yakınsamış)
kullanmanın önkoşuludur — sapma değil, doğru uygulamadır.

## 4. Neden collapse kapısı (z_std)?

**Karar.** Çökmüş (z_std → 0) veya sonlu-olmayan run diskalifiye edilir.

**Mantığımız.** SimSiam'ın L2-normalize kaybı BN istatistiklerini serbest bırakır ve non-kontrastif
yöntemler collapse'a eğilimlidir; çökmüş bir temsil hiçbir bilgi taşımaz, RankMe/diğer skorları anlamsız
kılar. Bu bir güvenlik kapısıdır, kalite skoru değil.

**Literatür.** Stop-gradient + predictor collapse'ı önler ama garanti etmez [ChenHe2021]; z_std ≈ 1/√d
sağlıklı referanstır. Batch-bağımsız collapse-önleme hâlâ aktif bir araştırma konusudur [IConE2026].

**Karşıt görüş.** z_std batch-bağımlı ve kaba bir göstergedir; boyutsal collapse'ı (kısmi) kaçırabilir —
tam collapse'ı yakalar ama rank düşüşünü değil.

**Yanıtımız.** Doğru; bu yüzden z_std'yi **yalnızca tam-collapse kapısı** olarak kullanıyoruz, kalite
sıralaması için değil. Kısmi/boyutsal collapse'ı **RankMe'nin kendisi** ve spektrum yakalar — yani iki
metrik farklı çökme türlerini kapsar (z_std: tam; RankMe/spektrum: boyutsal).

## 5. Neden RankMe'yi tek başına değil, PR + uniformity ile üçgenliyoruz?

**Karar.** Beraberlikte tie-break PR→uniformity ile; ayrıca spektrum + t-SNE ile görsel doğrulama.

**Mantığımız.** Tek bir metriğe (özellikle küçük kohortta gürültülü RankMe'ye) bağlı bir karar
kırılgandır. Bağımsız formüllerle aynı yönü gösteren metrikler kararı sağlamlaştırır.

**Literatür.** Otero [Otero2024] RankMe'nin **tek-başına-vekil** olarak başarısız olabileceğini gösterir;
alignment/uniformity ikilisi [WangIsola2020] temsil kalitesinin tamamlayıcı iki eksenidir; participation
ratio bağımsız bir effective-dim ölçüsüdür.

**Karşıt görüş.** PR ve RankMe ikisi de spektrumdan türer; "bağımsız" sayılmaları tartışmalı — aynı
bilgiyi iki kez saymak olabilir.

**Yanıtımız.** Kısmen haklı: PR ve RankMe spektrum-ilişkilidir (ikisi de aynı yönü gösterince güçlü sürpriz
değil). Ama **uniformity** farklı bir niceliktir (çift-mesafe dağılımı, spektrum değil) ve **alignment**
büsbütün ayrı eksendir (invaryans). Üçgenlemenin ağırlığı bu yüzden uniformity + alignment'tadır; PR yalnızca
RankMe'yi *doğrulayan* ikinci bir spektral okuma olarak raporlanır, bağımsız kanıt olarak değil.

## 6. Neden nihai söz denetimli prob'da (D3)?

**Karar.** Etiketler geldiğinde 256-d üzerinde lineer-prob/kNN AUC etiketsiz vekili ezer.

**Mantığımız.** Etiketsiz metrikler *vekildir*; "temsil IDH için iyi mi" sorusunun kesin cevabı ancak
göreve-özgü etiketli bir ölçümdür.

**Literatür.** Garrido'nun kendi konumu da budur: RankMe etiketsiz ön-eleme/seçim içindir, nihai ölçüt
lineer-prob'dur [Garrido2023]. Otero [Otero2024], proxy-görev uyumsuzluğunun gerçek olduğunu gösterir.

**Karşıt görüş.** "O zaman etiketsiz seçim neden yapılıyor? Doğrudan AUC'yi bekleyip ona göre seç." 

**Yanıtımız.** Etiketler bu aşamada yok ve pretraining kararları (backbone, recipe) şimdi verilmeli;
etiketsiz seçim, etiketli aşamaya **daha iyi bir başlangıç noktası** taşımanın tek yoludur. Ayrıca RankMe
ile AUC uyuşursa, vekil bu görev için **doğrulanmış** olur — bu da başlı başına bir katkıdır.

## 7. Belirsizlik: neden jackknife, neden tohum-arası CI?

**Karar.** RankMe/PR/uniformity için jackknife %95 GA; ayrıca ≥3 tohum ve tohum-arası %95 GA. Kazanan
ancak CI'sı ayrık olduğunda ilan edilir.

**Mantığımız.** Küçük kohortta (n≈99) RankMe gürültülüdür; ondalık farka göre "kazanan" ilan etmek
savunulamaz. Sıradan bootstrap, tekrarlı-satır örneklemesiyle **rank istatistiğini aşağı saptırır** (bunu
ampirik olarak gösterdik: CI nokta-tahmini içermiyordu). Jackknife (leave-one-out) tekrarsız alt-kümeler
kullanır → rank çökmez; RankMe düzgün fonksiyonel olduğundan geçerlidir.

**Literatür.** Bootstrap/jackknife ayrımı ve düzgün-olmayan istatistiklerde jackknife'ın geçerliliği
[Efron&Tibshirani1993]. Garrido büyük batch kullanır; biz küçük-kohort belirsizliğini nicelemekle bu
boşluğu kapatırız.

**Karşıt görüş.** Jackknife normal-yaklaşım CI verir; RankMe simetrik/normal dağılmayabilir. Bayesyen
bootstrap (Dirichlet ağırlıklandırma) tekrarsız *ve* percentil CI verir — daha uygun olabilir.

**Yanıtımız.** Doğru bir iyileştirme yönü; jackknife'ı **basit, deterministik ve düzgün-fonksiyonel için
geçerli** olduğu için seçtik. Bayesyen bootstrap gelecek-iş olarak açıkça not edilebilir; kararı
(ayrık-CI) değiştirmesi beklenmez çünkü marjlar (densenet vs resnet) büyüktür.

## 8. İlgili bir doğrulama: transfer edilebilir öznitelik neden H (encoder çıktısı), whitened/projektör değil?

**Karar.** 256-d transfer özniteliği encoder başının çıktısı H'dir; projektör/whitened çıktı değil.

**Literatür + karşıt-bulgu.** Whitening kaybı incelemesi [Huang2022], **whitened çıktıyı temsil olarak
kullanmanın H'den belirgin biçimde kötü** olduğunu ve H'nin tam whitenlanmadığını gösterir. Bu bulgu bizim
tercihimizi *destekler*: W-MSE'yi yalnızca bir eğitim regülarizatörü olarak kullanıp özniteliği H'den
çıkarıyoruz. Aynı bulgu, "whitening/tam-rank her zaman daha iyi temsil verir" varsayımına karşı bir uyarıdır
— bu yüzden yüksek rank'ı *tek başına* başarı saymıyoruz (§1, §6).

---

## Alternatif etiketsiz seçim metrikleri (özet)

| Metrik | Temel | RankMe'ye göre | Neden birincil almadık |
|---|---|---|---|
| **RankMe** [Garrido2023] | effective rank (entropi) | — | hiperparametresiz, en yerleşik → **birincil** |
| **LiDAR** [Thilak2024] | LDA-tarzı rank (aug yapısı) | lineer-prob'u daha iyi izler | ek yapı gerektirir; **gelecek-iş olarak öneriyoruz** |
| **α-ReQ** [Agrawal2022] | spektrum güç-yasası üsteli | OOD'de bir tık iyi | in-domain'de RankMe daha iyi |
| **IdEst** [IdEst2026] | intrinsic dimension | rekabetçi | çok yeni, az doğrulanmış |

**Sonuç.** RankMe'yi *birincil ama tek değil* olarak, sınırlarını (Otero) açıkça kabul edip üçgenleme +
convergence/collapse kapıları + belirsizlik nicelemesi + denetimli D3 ile sararak kullanıyoruz. Katkı,
tek bir "sihirli metrik" değil; **RankMe'nin bilinen tuzaklarını kapatan, deterministik ve savunulabilir
bir etiketsiz seçim prosedürüdür** — ve daha iyi bir metrik (LiDAR) çıktığında onu takmaya açıktır.

---

## Atıflar
- **[ChenHe2021]** Chen & He, *Exploring Simple Siamese Representation Learning*, CVPR 2021.
- **[WangIsola2020]** Wang & Isola, *Alignment and Uniformity on the Hypersphere*, ICML 2020.
- **[Garrido2023]** Garrido, Balestriero, Najman, LeCun, *RankMe*, ICML 2023 (arXiv:2210.02885).
- **[Jing2022]** Jing, Vincent, LeCun, Tian, *Understanding Dimensional Collapse in Contrastive SSL*, ICLR 2022.
- **[Thilak2024]** Thilak et al., *LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures*, 2024 (arXiv:2312.04000).
- **[Agrawal2022]** Agrawal et al., *α-ReQ: Assessing representation quality via eigenspectrum decay*, NeurIPS 2022.
- **[IdEst2026]** *IdEst: Assessing SSL Representations via Intrinsic Dimension*, 2026 (arXiv:2606.03338).
- **[Otero2024]** Otero et al., *Self-Supervised Anomaly Detection in the Wild* — RankMe'nin tek-vekil sınırları.
- **[Huang2022]** Huang et al., *An Investigation into Whitening Loss for Self-supervised Learning*, NeurIPS 2022 (arXiv:2210.03586).
- **[Bardes2022]** Bardes, Ponce, LeCun, *VICReg*, ICLR 2022.
- **[Ermolov2021]** Ermolov et al., *Whitening for Self-Supervised Representation Learning (W-MSE)*, ICML 2021.
- **[IConE2026]** *IConE: Batch-Independent Collapse Prevention for SSL*, 2026 (arXiv:2603.15263).
- **[Efron&Tibshirani1993]** Efron & Tibshirani, *An Introduction to the Bootstrap*.
