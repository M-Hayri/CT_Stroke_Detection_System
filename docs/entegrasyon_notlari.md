# C# tarafı için entegrasyon notları

`model/ai-servisi/app.py`, gerçek `unet_focal_tversky_fold0..4.pt` 5-fold
topluluğunu servis ediyor. Uç nokta aynı (`POST /api/stroke-segmentation`),
mevcut yanıt alanları korunuyor — ama yanıt gövdesine ek alanlar eklendi.
`System.Text.Json` bilinmeyen alanları varsayılan olarak yok sayar, bu yüzden
mevcut kod bozulmaz; ama yeni alanları kullanmak isterseniz aşağıdaki
değişiklikler öneriliyor (henüz uygulanmadı).

## 1. `StrokeApiResponse` sınıfına eklenebilecek alanlar

`backend/Controllers/HastaController.cs` içindeki `StrokeApiResponse` (dosya sonu):

```csharp
public class StrokeApiResponse
{
    public bool InmeVarMi { get; set; }
    public double GuvenSkoru { get; set; }
    public string? TeshisDetay { get; set; }
    public string? MaskeYolu { get; set; }

    // --- Henuz kullanilmayan alanlar ---
    public string? OverlayYolu { get; set; }
    public int LezyonAlanPx { get; set; }
    public int OdakSayisi { get; set; }
    public List<int>? OdakAlanlariPx { get; set; }
    public double MaskeIciOrtalamaOlasilik { get; set; }
    public double MaksimumOlasilik { get; set; }
    public string? ModelSurumu { get; set; }
    public List<string>? CheckpointSha { get; set; }
    public bool SonIslemUygulandi { get; set; }
    public GirdiKontrolDto? GirdiKontrol { get; set; }
    public SinirlamalarDto? Sinirlamalar { get; set; }
}

public class GirdiKontrolDto
{
    public string? Durum { get; set; }       // "ok" | "warn"
    public double? MedyanYogunluk { get; set; }
    public string? Sebep { get; set; }
}

public class SinirlamalarDto
{
    public string? Kapsam { get; set; }
    public string? Hacim { get; set; }
    public string? CokOdak { get; set; }
    public string? YanlisPozitif { get; set; }
    public string? Kalibrasyon { get; set; }
    public string? Uyari { get; set; }
}
```

`SegmentasyonSonucu` veritabanı modeli/migration'ı bu alanlar için henüz
güncellenmedi. Bunları kalıcı saklamak istenirse ayrı bir migration gerekir
(`OverlayYolu`, `LezyonAlanPx` vb. için); bu karar bilerek burada bırakıldı,
yalnız DTO tarafı hazır.

## 2. `girdiKontrol.durum == "warn"` — arayüzde uyarı bandı

`CekimAnalizEt` (`HastaController.cs`) her kesit için API'ye tek tek istek
atıyor. `apiSonuc.GirdiKontrol?.Durum == "warn"` olduğunda, ilgili kesit için
bir uyarı işareti taşınmalı (ör. `SegmentasyonSonucu`'na eklenecek bir
`GirdiUyarisi` alanı, ya da `ViewBag` üzerinden `CekimDetay`'a taşınan bir
liste). Görünümde (`Views/Hasta/CekimDetay.cshtml`) bu kesit için sarı bir
bant ile `girdiKontrol.sebep` metni gösterilmeli (ör. "renkli girdi (overlay
olabilir)" veya "medyan gri yoğunluk bant dışı"). En kritik durum: renkli/
overlay girdi tespiti — biri yanlışlıkla ham BT yerine zaten işaretlenmiş bir
overlay görüntü yüklerse model üzerinde sahte-mükemmel bir sonuç üretebilir;
bu uyarının gözden kaçmaması önemli.

## 3. `sinirlamalar` — sonuç ekranında katlanabilir panel

Her segmentasyon sonucunun yanında varsayılan kapalı, tıklanınca açılan bir
panel ("Bu sonuç hakkında sınırlamalar") önerilir; içeriği doğrudan
`sinirlamalar.*` alanlarından doldurulur. `sinirlamalar.kalibrasyon` ve
`sinirlamalar.uyari` alanları her zaman gösterilmeli (yanıttan hiçbir zaman
çıkarılmaz) — güven skorunun kalibre edilmediği ve aracın tanı amaçlı
olmadığı hekime her seferinde hatırlatılmalı.

## 4. Servisi ayağa kaldırma

```bash
cd model/ai-servisi
uvicorn app:app --host 127.0.0.1 --port 8000
```

`backend/Program.cs` zaten `http://localhost:8000/` adresini bekliyor —
port/host değişmedi. Not: `app.py` import sırasında 5 checkpoint'in
SHA-256'sını `model/agirliklar/unet_results.json` içindeki referansla
karşılaştırıyor; uyuşmazlık durumunda `uvicorn` başlamadan hata verip çıkar
(bu kasıtlı bir davranış, bozuk/değiştirilmiş bir checkpoint'in sessizce
servis edilmesini engeller).

Duman testi (servis ayaktayken, ayrı bir terminalde — orijinal TEKNOFEST veri
setine ihtiyaç duyduğu için bu depoda doğrudan çalışmaz, bkz. `smoke_test.py`
üstündeki not):

```bash
cd model/ai-servisi
python smoke_test.py
```

## 5. `GoruntuController.SegmentasyonYap` neden yok

Eskiden `Controllers/GoruntuController.cs` içinde böyle bir action vardı:
sabit `GuvenSkoru = 0.9450m` yazıp orijinal görüntüyü maske olarak kaydeden,
gerçek AI servisine hiç bağlanmayan ölü kod. Kaldırıldı — hiçbir view/JS
dosyasında buna bir link/form/AJAX çağrısı olmadığı için kaldırma herhangi
bir ekranı bozmadı. Yeni bir view/route eklerken bu action'ın artık var
olmadığını unutmayın.
