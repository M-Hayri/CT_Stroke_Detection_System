# BTİNME — BT Kesitlerinde İnme (Kanama) Tespiti için Klinik Karar Destek Sistemi

Hekimlerin hasta/muayene kayıtlarını tuttuğu, beyin BT kesitlerini yükleyip
bir U-Net segmentasyon modeliyle kanama şüpheli bölgeleri otomatik
işaretleyebildiği ve aynı hastanedeki meslektaşlarıyla görüntü üzerinden
konsültasyon yapabildiği bir web uygulaması.

> **Önemli — tıbbi sorumluluk reddi:** Bu sistem bir tanı aracı değildir.
> Model yalnız kanama türü lezyonlar üzerinde değerlendirilmiştir, güven
> skoru klinik olasılık olarak yorumlanamaz ve hiçbir çıktı hekim
> değerlendirmesinin yerine geçmez. Ayrıntılı sınırlamalar için
> `model/ai-servisi/app.py` içindeki `SINIRLAMALAR` sözlüğüne bakınız.

## İçindekiler

- [Mimari](#mimari)
- [Klasör yapısı](#klasör-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Model hakkında](#model-hakkında)
- [Bilinen sınırlamalar](#bilinen-sınırlamalar)

## Mimari

Sistem üç parçadan oluşuyor: ASP.NET Core MVC ile yazılmış bir web uygulaması
(backend + sunucu tarafı görünümler), bu uygulamanın sunduğu statik dosyalar
(frontend) ve ayrı bir süreç olarak çalışan Python/FastAPI tabanlı yapay
zeka servisi. Backend, bir BT kesiti için analiz istendiğinde bu servise
HTTP isteğiyle bağlanır.

### Genel akış

```mermaid
flowchart TD
    subgraph TARAYICI["Tarayıcı (Razor View / Bootstrap)"]
        A1["Giriş Sayfası"]
        A2["Hasta Yönetimi Listesi"]
        A3["Hasta Detayı\n(Muayene + Sağlık Geçmişi)"]
        A4["Radyoloji Sayfası\n(Çekim Klasörleri)"]
        A5["Kesit Detay Sayfası\n(Slice Slider + AI Sonucu)"]
        A6["Konsültasyon Paneli\n(Mesajlaşma)"]
    end

    subgraph BACKEND["backend/ — ASP.NET Core (C#)"]
        B1["AccountController.Login\nTC + Şifre doğrulama"]
        B2["Cookie Claim Oluştur\n(DoktorId, Hastane bilgisi)"]
        B3["HastaController.Index\nDoktorHasta üzerinden hasta listesi"]
        B4["HastaController.CekimYukle\nBT kesiti dosya kaydı"]
        B5["HastaController.CekimAnalizEt\nHttpClient ile AI isteği"]
        B6["SegmentasyonSonucu\nkaydet / güncelle"]
        B7["MesajController\nGonder / PanelIcerik"]
    end

    subgraph DB["PostgreSQL (btinmeprogrami)"]
        D1[("doktor / hasta / hastane")]
        D2[("goruntu / segmentasyon_sonucu")]
        D3[("muayene / saglik_gecmisi")]
        D4[("mesaj / mesaj_ek")]
    end

    subgraph AI["model/ai-servisi/app.py — Python FastAPI"]
        C1["Açılış Kontrolü:\n5 checkpoint SHA-256 doğrulama"]
        C2["POST /api/stroke-segmentation"]
        C3["Girdi Denetimi\n(kafa maskesi, gri yoğunluk bandı)"]
        C4["Ön İşleme\nload_input_1ch (224x224, 1 kanal)"]
        C5["5-Fold U-Net Ensemble\nolasılık ortalaması"]
        C6["Eşikleme (>0.5) +\nBağlı Bileşen Analizi"]
        C7["Lezyon Alanı >= 50px ?"]
        C8["Maske + Overlay PNG\nfrontend/wwwroot/uploads/masks"]
        C9["JSON Yanıt:\ninmeVarMi, guvenSkoru, overlayYolu, sinirlamalar"]
    end

    A1 -->|"TC/Sifre gonder"| B1
    B1 -->|"dogru ise"| B2 --> A2
    B1 -.->|"hatali ise"| A1
    B1 <-->|"SELECT"| D1

    A2 <--> B3 <-->|"SELECT"| D1
    A2 --> A3
    A3 <-->|"SELECT/INSERT"| D3
    A2 --> A4
    A4 -->|"kesit yukle"| B4 -->|"INSERT"| D2
    B4 --> A4

    A4 -->|"Yapay Zeka ile Analiz Et"| B5
    B5 -->|"multipart POST"| C2
    C1 -.->|"servis ayaga kalkarken bir kez"| C2
    C2 --> C3 --> C4 --> C5 --> C6 --> C7
    C7 -->|"evet"| C8 --> C9
    C7 -->|"hayir"| C9
    C9 -->|"HTTP 200 JSON"| B5
    B5 --> B6 -->|"INSERT/UPDATE"| D2
    B6 --> A5

    A5 <-->|"SELECT"| D2

    A6 <--> B7 <-->|"INSERT/SELECT"| D4
```

### Veritabanı ER diyagramı

```mermaid
erDiagram
    HASTANE ||--o{ DOKTOR : "doktor_calistirir"
    DOKTOR ||--o{ HASTANE : "bas_hekim_olarak_yonetir"
    DOKTOR ||--o{ DOKTOR_HASTA : "hasta_takip_eder"
    HASTA ||--o{ DOKTOR_HASTA : "doktor_tarafindan_takip_edilir"
    DOKTOR ||--o{ HASTA : "ilk_kaydi_olusturur"
    HASTA ||--o{ GORUNTU : "cekim_sahibidir"
    GORUNTU ||--o{ SEGMENTASYON_SONUCU : "analiz_sonucu_uretir"
    HASTA ||--o{ MUAYENE : "muayene_olur"
    DOKTOR ||--o{ MUAYENE : "muayene_yapar"
    HASTA ||--o{ SAGLIK_GECMISI : "gecmis_kaydina_sahiptir"
    DOKTOR ||--o{ MESAJ : "gonderen_olarak_yollar"
    DOKTOR ||--o{ MESAJ : "alici_olarak_alir"
    MESAJ ||--o{ MESAJ_EK : "ek_dosya_icerir"

    HASTANE {
        int hastane_id PK
        string ad
        string lokasyon
        string adres
        int bas_hekim_id FK
    }
    DOKTOR {
        int doktor_id PK
        string tc
        string sifre
        string isim
        string soyad
        string uzmanlik_alani
        string telefon
        string email
        int hastane_id FK
    }
    HASTA {
        int hasta_id PK
        string tc
        string isim
        string soyad
        date dogum_tarihi
        string cinsiyet
        string telefon
        string adres
        int doktor_id FK
    }
    DOKTOR_HASTA {
        int id PK
        int doktor_id FK
        int hasta_id FK
        timestamp eklenme_tarihi
    }
    GORUNTU {
        int goruntu_id PK
        int hasta_id FK
        string goruntu_turu
        string dosya_yolu
        string aciklama
        timestamp cekim_tarihi
    }
    SEGMENTASYON_SONUCU {
        int sonuc_id PK
        int goruntu_id FK
        string model_versiyonu
        string maske_dosya_yolu
        decimal guven_skoru
        timestamp islem_tarihi
    }
    MUAYENE {
        int muayene_id PK
        int doktor_id FK
        int hasta_id FK
        timestamp tarih
        string teshis
        string notlar
    }
    SAGLIK_GECMISI {
        int gecmis_id PK
        int hasta_id FK
        string hastalik_adi
        date tani_tarihi
        string aciklama
        boolean kronik_mi
    }
    MESAJ {
        int mesaj_id PK
        int gonderen_doktor_id FK
        int alici_doktor_id FK
        string icerik
        timestamp gonderim_tarihi
        boolean okundu_mu
    }
    MESAJ_EK {
        int ek_id PK
        int mesaj_id FK
        string dosya_yolu
        string dosya_adi
    }
```

### Use case diyagramı

```mermaid
flowchart LR
    Hekim(["👤 Hekim (Doktor)"])

    subgraph SISTEM["İNME — Klinik Karar Destek Sistemi"]
        UC1(["Giriş Yap"])
        UC2(["Çıkış Yap"])
        UC3(["Hasta Listesini Görüntüle"])
        UC4(["Yeni Hasta Ekle / TC ile Sorgula"])
        UC5(["Hasta Detayını Görüntüle"])
        UC6(["Muayene Kaydı Ekle"])
        UC7(["Sağlık Geçmişi Ekle"])
        UC8(["BT Kesiti / Klasör Yükle"])
        UC9(["Yapay Zeka ile Analiz Et"])
        UC10(["Analiz Sonucunu Görüntüle"])
        UC11(["Muayene Raporu Yazdır"])
        UC12(["Meslektaşa Mesaj Gönder"])
        UC13(["Görüntü Klasörünü Konsültasyona Paylaş"])
        UC14(["Okunmamış Mesajları Takip Et"])
    end

    AIServis(["⚙️ Python AI Servisi\n(U-Net Ensemble)"])

    Hekim --> UC1
    Hekim --> UC2
    Hekim --> UC3
    Hekim --> UC4
    Hekim --> UC5
    Hekim --> UC6
    Hekim --> UC7
    Hekim --> UC8
    Hekim --> UC9
    Hekim --> UC10
    Hekim --> UC11
    Hekim --> UC12
    Hekim --> UC13
    Hekim --> UC14

    UC9 -.->|"include"| AIServis
    UC13 -.->|"include"| UC12
    UC4 -.->|"include"| UC3
```

Yukarıdaki üç diyagramın düzenlenebilir (diagrams.net / draw.io) kaynak
dosyaları `docs/diagrams/` klasöründe bulunur:

- [docs/diagrams/hasta_akis_diyagrami.drawio](docs/diagrams/hasta_akis_diyagrami.drawio)
- [docs/diagrams/er_diyagrami.drawio](docs/diagrams/er_diyagrami.drawio)
- [docs/diagrams/use_case_diyagrami.drawio](docs/diagrams/use_case_diyagrami.drawio)

Düzenlemek için [app.diagrams.net](https://app.diagrams.net) üzerinden
*Dosya > Diyagram Aç* ile açabilirsiniz.

## Klasör yapısı

```
CTStrokeDetectionSystem/
├── frontend/
│   └── wwwroot/            # css, js, images, lib, uploads (statik dosyalar)
├── backend/                # ASP.NET Core MVC (Controllers, Models, Views, Migrations)
├── model/
│   ├── ai-servisi/         # FastAPI cikarim servisi (app.py) + duman testi
│   ├── cekirdek/           # U-Net mimarisi, kayip fonksiyonlari, on-isleme (tek kaynak)
│   ├── egitim/             # Egitim betigi (train_unet.py) + dataset_manifest.json
│   └── agirliklar/         # Egitilmis checkpoint'ler (.pt), unet_results.json, log
└── docs/
    ├── diagrams/            # .drawio diyagram kaynaklari (akis, ER, use case)
    └── entegrasyon_notlari.md
```

## Kurulum

### Gereksinimler

- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- [Python 3.10+](https://www.python.org/downloads/)
- [PostgreSQL](https://www.postgresql.org/download/) (yerelde çalışan bir sunucu)

### 1. Veritabanı

PostgreSQL üzerinde `btinmeprogrami` adında bir veritabanı oluşturun ve
`backend/appsettings.json` içindeki `ConnectionStrings:DefaultConnection`
değerini kendi kullanıcı adı/şifrenize göre güncelleyin (depodaki değer
yalnız yerel geliştirme için bırakılmış bir örnektir, gerçek bir ortamda
kullanılmamalıdır):

```bash
dotnet ef database update --project backend
```

### 2. Backend (ASP.NET Core)

```bash
cd backend
dotnet restore
dotnet run
```

Uygulama varsayılan olarak `http://localhost:5072` adresinde açılır ve
doğrudan giriş sayfasına yönlendirir.

### 3. Yapay zeka servisi (Python/FastAPI)

```bash
cd model/ai-servisi
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Backend, `http://localhost:8000` adresindeki bu servise istek atacak şekilde
yapılandırılmıştır (`backend/Program.cs`). Servis açılırken 5 checkpoint
dosyasının SHA-256 bütünlüğünü otomatik doğrular; uyuşmazlık varsa servis
başlamadan hata verir.

> Backend ve AI servisi **ayrı süreçlerdir**; ikisinin de aynı anda çalışır
> durumda olması gerekir. AI servisi kapalıyken BT analiz isteği gönderilirse
> ilgili kesit sessizce atlanır (bkz. `HastaController.CekimAnalizEt`).

## Çalıştırma

1. PostgreSQL ve AI servisini ayağa kaldırın.
2. Backend'i çalıştırın, tarayıcıda giriş sayfası açılır.
3. Sisteme kayıtlı bir doktor (TC + şifre) ile giriş yapın.
4. "Hasta Yönetimi"nden bir hasta seçin → Radyoloji sekmesinden BT kesiti
   yükleyin → "Yapay Zeka ile Analiz Et" ile segmentasyonu çalıştırın.
5. Sonucu Kesit Detay sayfasında (maske + güven skoru + sınırlamalar) görün.
6. Sağ üstteki "Konsültasyon & Mesajlar" panelinden aynı hastanedeki bir
   meslektaşınıza görüntü klasörünü paylaşabilirsiniz.

## Model hakkında

- **Mimari:** 4 seviyeli U-Net (encoder-decoder), BatchNorm, bilinear
  upsample — `model/cekirdek/unet_model.py`.
- **Kayıp fonksiyonu:** Focal Tversky Loss (alpha=0.7, beta=0.3, gamma=0.75)
  — küçük/dengesiz lezyon alanlarını standart BCE+Dice'tan daha güçlü
  cezalandırır.
- **Topluluk (ensemble):** 5-fold çapraz doğrulamadan çıkan 5 checkpoint'in
  olasılık tahminlerinin ortalaması alınır (tek bir fold değil).
- **Girdi:** 224x224 tek kanal (gri) BT kesiti; kafa dışı bölge Otsu eşiği +
  en büyük bağlı bileşen yöntemiyle maskelenip sıfırlanır.
- **Eğitim verisi:** TEKNOFEST beyin BT veri seti. Boyut ve gizlilik
  nedeniyle ham görüntüler bu depoda yer almaz; `model/egitim/` içinde
  yalnız eğitim kodu ve fold/etiket atamalarını içeren
  `dataset_manifest.json` bulunur.

## Bilinen sınırlamalar

- Yalnız **kanama** türü lezyonlar için doğrulanmıştır; iskemik inme için
  model performansı ölçülmemiştir.
- İşaretlenen alan gerçek lezyondan sistematik olarak küçüktür (ölçülen
  oran ~%72-76).
- Birden fazla odak varsa bir kısmı yakalanamayabilir (ikincil odak
  yakalama oranı ~%63).
- Orta hat ve damar yapılarında yanlış pozitif işaretleme gözlenmiştir.
- Güven skoru **kalibre edilmemiştir**; klinik olasılık olarak
  yorumlanamaz.
- Bu liste `model/ai-servisi/app.py` içindeki `SINIRLAMALAR` sözlüğüyle
  birebir aynıdır ve her analiz sonucunda hekime aynen gösterilir.
