# Savunma Notları — Beklenen Hocalar/Hakem Soruları ve Cevapları

> Sunumda gelmesi muhtemel zorlu sorular ve savunulabilir, literatür-dayanaklı, dürüst cevaplar.
> Atıflar `PAPER_selection_justification.md` ile ortak.

---

## S1. "Bu literatürde karşılığı var mı? Neyle kıyaslayarak iyi iş yaptığını düşünüyorsun? Özgün mü?"

### Dürüst konumlandırma (aşırı iddia YOK)
- **Yeni bir metrik icat ETMİYORUZ.** RankMe [Garrido2023], LiDAR [Thilak2024], α-ReQ [Agrawal2022],
  IdEst [IdEst2026] zaten var. Etiketsiz bir metrikle seçim yapma fikri de yeni değil — Garrido bunu
  hiperparametre/checkpoint seçimi için önerdi.
- **Katkımız metrik değil, PROSEDÜR:** var olan metrikleri, iki *belgelenmiş başarısızlık modunu*
  kapatan **deterministik, tekrarlanabilir bir seçim algoritmasında** birleştiriyoruz — (i) RankMe'nin
  rastgele başlangıçta şişmesi (yakınsama kapısı), (ii) tek-vekil olarak dengesiz görevlerde çökmesi
  [Otero2024] (üçgenleme + denetimli D3). Ayrıca bunu **3D beyin-MRI SimSiam backbone seçimine** ilk kez
  uyguluyoruz. Tıbbi SSL literatüründe seçim hâlâ çoğunlukla **etiketli** validasyonla yapılır (top-k
  ensembling vb.) — tam etiketsiz bir prosedür bu alanda seyrektir.

### "Neyle kıyaslıyorsun?" — üç somut karşılaştırma
Bu, sorunun en önemli kısmı; cevabımız üç katmanlı:
1. **Naif seçim stratejilerine karşı** (Stage-B ablasyonu). Prosedürümüzü şu baseline'larla kıyaslıyoruz:
   - **Salt max-RankMe (kapısız):** epoch 4'ü (eğitilmemiş) seçer — gösteriyoruz ki başarısız.
   - **last.pth (son epoch):** yakınsamış ama daha çok collapse olmuş.
   - **min-val_loss:** rank'ı hiç dikkate almaz.
   Gate açık/kapalı ablasyonu bu farkı **niceliksel** gösterir — "kapı gerçekten daha iyi checkpoint
   seçiyor mu?" sorusunun deneysel cevabı.
2. **Metrik alternatiflerine karşı:** RankMe'yi birincil aldık ama LiDAR/α-ReQ/IdEst'i de aynı
   prosedürde çalıştırıp seçimin **metrik-agnostik** olduğunu (aynı kazananı verdiğini) gösterebiliriz.
3. **Nihai/altın-standart: etiketli oracle ile uyum (concordance).** Garrido'nun kendi başarı ölçütü
   budur: etiketsiz seçim, etiketli-validasyon seçiminin ne kadarını *geri kazanıyor*? Etiketler
   geldiğinde (D3) "label-free seçimimiz = AUC-optimal seçim mi?" diye ölçeceğiz. Uyum yüksekse,
   prosedür bu görev için **doğrulanmış** olur — "iyi iş yaptık"ın nicel kanıtı budur.

### Özgünlük — net cümle
> "Yeni metrik iddia etmiyoruz. Katkımız: (a) RankMe'nin iki bilinen tuzağını kapatan deterministik,
> belirsizlik-nicelenmiş bir seçim prosedürü; (b) bunun 3D beyin-MRI SSL backbone seçimine ilk
> uygulanışı; (c) naif seçim baseline'larına ve (etiket gelince) etiketli oracle'a karşı doğrulanması."

---

## S2. "256 feature çıkıyor ama RankMe ~7–15 diyor; bu normal mi? Daha fazla çıkamaz mı? RankMe çıktısını nasıl savunacaksın?"

### Önce iki KRİTİK düzeltme (yanlış-anlama tuzağı)
1. **RankMe "kaç feature bilgi tutuyor" DEĞİLDİR.** RankMe = tekil-değer dağılımının *üstel Shannon
   entropisi* — "kaç boyut *efektif olarak* kullanımda" [Garrido2023]. **7 çıkması, 256 boyutun 249'u
   boş demek değildir.** Boyutların çoğu bir miktar varyans taşır; dağılım *yoğunlaşmıştır*, o kadar.
   "Sadece 7 feature bilgi tutuyor" cümlesi metriğin yanlış okunmasıdır.
2. **RankMe varyans yayılımını ölçer, AYIRT EDİCİLİĞİ değil.** Çok küçük varyanslı bir yön, IDH için
   son derece ayırt edici olabilir. Yani düşük RankMe, "IDH için sadece 7 boyut kullanılabilir"
   anlamına gelmez — lineer-prob 256 boyutun tamamını kullanır. RankMe bir *vekildir*, kapasite tavanı
   değil.

### "Bu normal mi?" — Evet, beklenen; ama iyi değil, o yüzden zaten üzerine gidiyoruz
- **Boyutsal collapse belgelenmiş bir olgudur:** SOTA SSL temsili düşük-boyutlu bir alt-uzaya
  haritalar [Jing2022]; effective rank'ın embedding boyutundan çok küçük olması *beklenir*, özellikle
  **kontrastsız** yöntemlerde (SimSiam) çünkü rank-koruyucu terimleri yoktur.
- **Ama literatür bunun bir dezavantaj olduğunu da söyler:** düşük rank downstream başarısı için "bir
  darboğaz"dır ve VICReg/W-MSE/WERank gibi yöntemler tam da onu yükseltmek için vardır [Bardes2022,
  Ermolov2021, WERank2024]. **Bizim Stage-C ablasyonlarımız (VICReg, W-MSE, büyük head, uzun+aug)
  doğrudan bunu hedefliyor** — yani "7 düşük mü?" sorusunun cevabını *biz de arıyoruz ve yükseltmeye
  çalışıyoruz*. Bu bir kusur değil, çalışmanın bir ekseni.

### "Daha fazla çıkamaz mı?" — Evet, çıkabilir; nasıl artacağını da gösteriyoruz
Effective rank şunlarla artar: (i) rank-koruyucu kayıp terimi (VICReg/W-MSE) — Stage-C; (ii) daha çok/
çeşitli veri — küçük kohortumuz (556–889 denek) temsilin *ihtiyaç duyduğu* boyut sayısını sınırlar;
(iii) daha büyük batch/projektör. Yani 7 bir "tavan" değil, mevcut (kontrastsız + küçük-veri)
rejimin sonucu. densenet'in resnet'lerden yüksek rank'ı zaten "collapse'a en dirençli backbone" =
bizim seçim gerekçemiz.

### Ölçüm dürüstlüğü (hakem bunu soracak, önden söyle)
- **n=99'da RankMe'nin tavanı ~99'dur** (min(n, d)); değerimiz 7 bu tavanın çok altında → yani 7,
  örnek-sayısı tavanının değil, **gerçek yoğunlaşmanın** sonucu (temsil ~50 boyut kullansa 99 örnekte
  ~50 görürdük). Yine de küçük-n kestirimi gürültülüdür — bu yüzden **jackknife %95 GA** ile
  raporluyoruz ve Garrido'nun büyük-batch'ine kıyasla bu belirsizliği açıkça niceliyoruz.
- Effective rank'ı **transfer edilebilir öznitelik H (256-d) üzerinde** hesaplıyoruz (Garrido'nun
  reçetesi), projektör/whitened çıktı üzerinde değil — [Huang2022] whitened çıktının H'den kötü
  olduğunu gösterdiğinden bu tercih de savunulabilir.

### RankMe çıktısını nasıl savunacağız — net cümle
> "RankMe mutlak bir kalite sertifikası değil, **göreli** bir çökme/zenginlik vekilidir; onu tek
> başına değil, üçgenleyerek ve nihai kararı denetimli AUC'ye bırakarak kullanıyoruz. Düşük mutlak
> değer, kontrastsız SSL'nin belgelenmiş boyutsal-collapse davranışıyla tutarlıdır; bunu bir kusur
> olarak kabul edip Stage-C'de rank-koruyucu yöntemlerle yükseltmeyi deniyoruz. Değerimiz örnek-sayısı
> tavanının çok altında olduğundan gerçek bir yoğunlaşmayı yansıtır, artefakt değil; belirsizliğini de
> jackknife CI ile açıkça raporluyoruz."

---

## Ek atıflar
- **[WERank2024]** *WERank: Towards Rank Degradation Prevention for SSL Using Weight Regularization*, 2024 (arXiv:2402.09586).
- **[MatInfo2023]** *Matrix Information Theory for Self-Supervised Learning*, 2023 (arXiv:2305.17326).
- (Diğerleri: `PAPER_selection_justification.md` kaynakçası.)
