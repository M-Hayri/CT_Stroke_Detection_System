using BTİNME.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using System.Text.Json;

namespace BTİNME.Controllers
{
    [Authorize]
    public class HastaController : Controller
    {
        private readonly AppDbContext _context;
        private readonly IHttpClientFactory _httpClientFactory; // yapay zeka servisine HTTP istegi atmak icin

        public HastaController(AppDbContext context, IHttpClientFactory httpClientFactory)
        {
            _context = context;
            _httpClientFactory = httpClientFactory;
        }

        // GET: /Hasta/Index
        public async Task<IActionResult> Index()
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            if (claim == null || !int.TryParse(claim.Value, out int aktifDoktorId))
            {
                return RedirectToAction("Login", "Account");
            }

            // Aktif doktorun takip ettiği tüm hastaları doktor_hasta ilişki tablosundan çekiyoruz
            var doktorunHastalari = await _context.DoktorHastas
                .Where(dh => dh.DoktorId == aktifDoktorId)
                .Include(dh => dh.Hasta)
                .Select(dh => dh.Hasta)
                .ToListAsync();

            return View(doktorunHastalari);
        }

        // 1. SADECE KLİNİK DOSYA (Muayene ve Sağlık Geçmişi)
        [HttpGet]
        public async Task<IActionResult> Detay(int id)
        {
            var hasta = await _context.Hastas
                .Include(h => h.Muayenes)
                    .ThenInclude(m => m.Doktor)
                        .ThenInclude(d => d.Hastane)
                .Include(h => h.SaglikGecmisis)
                .FirstOrDefaultAsync(h => h.HastaId == id);

            if (hasta == null) return NotFound();

            return View(hasta);
        }

        // 2. RADYOLOJİ KLASÖRLERİ SAYFASI (Hastanın Çekim Klasörleri)
        [HttpGet]
        public async Task<IActionResult> Radyoloji(int id)
        {
            var hasta = await _context.Hastas
                .Include(h => h.Goruntus)
                    .ThenInclude(g => g.SegmentasyonSonucus)
                .FirstOrDefaultAsync(h => h.HastaId == id);

            if (hasta == null) return NotFound();

            return View(hasta);
        }

        // 3. KLASÖR İÇERİĞİ / SLICE SLIDER & AI DETAY SAYFASI
        [HttpGet]
        public async Task<IActionResult> CekimDetay(int id)
        {
            var goruntu = await _context.Goruntus
                .Include(g => g.Hasta)
                .Include(g => g.SegmentasyonSonucus)
                .FirstOrDefaultAsync(g => g.GoruntuId == id);

            if (goruntu == null) return NotFound();

            var tumKesitler = await _context.Goruntus
                .Include(g => g.SegmentasyonSonucus)
                .Where(g => g.HastaId == goruntu.HastaId
                         && g.GoruntuTuru == goruntu.GoruntuTuru
                         && g.CekimTarihi == goruntu.CekimTarihi)
                .OrderBy(g => g.GoruntuId)
                .ToListAsync();

            ViewBag.TumKesitler = tumKesitler;
            ViewBag.AnaGoruntu = goruntu;

            return View(goruntu.Hasta);
        }

        // 4. ÇOKLU KESİT / KLASÖR YÜKLEME (POST)
        [HttpPost]
        public async Task<IActionResult> CekimYukle(int hastaId, string goruntuTuru, string aciklama, List<IFormFile> dosyalar)
        {
            if (dosyalar != null && dosyalar.Count > 0)
            {
                var uploadsFolder = Path.Combine(Directory.GetCurrentDirectory(), "..", "frontend", "wwwroot", "uploads");
                if (!Directory.Exists(uploadsFolder))
                {
                    Directory.CreateDirectory(uploadsFolder);
                }

                DateTime cekimTarihi = DateTime.Now;

                foreach (var dosya in dosyalar)
                {
                    if (dosya.Length > 0)
                    {
                        var uniqueFileName = Guid.NewGuid().ToString() + "_" + Path.GetFileName(dosya.FileName);
                        var filePath = Path.Combine(uploadsFolder, uniqueFileName);

                        using (var stream = new FileStream(filePath, FileMode.Create))
                        {
                            await dosya.CopyToAsync(stream);
                        }

                        var yeniGoruntu = new Goruntu
                        {
                            HastaId = hastaId,
                            GoruntuTuru = goruntuTuru ?? "BT",
                            Aciklama = aciklama,
                            DosyaYolu = "/uploads/" + uniqueFileName,
                            CekimTarihi = cekimTarihi
                        };

                        _context.Goruntus.Add(yeniGoruntu);
                    }
                }

                await _context.SaveChangesAsync();
            }

            return RedirectToAction("Radyoloji", new { id = hastaId });
        }

        // 5. STROKENEXT AI ANALİZİNİ ÇALIŞTIRMA — ARTIK GERÇEK PYTHON API'SİNE BAĞLANIYOR
        [HttpPost]
        public async Task<IActionResult> CekimAnalizEt(int hastaId, string goruntuTuru, DateTime cekimTarihi)
        {
            var kesitler = await _context.Goruntus
                .Include(g => g.SegmentasyonSonucus)
                .Where(g => g.HastaId == hastaId
                         && g.GoruntuTuru == goruntuTuru
                         && g.CekimTarihi == cekimTarihi)
                .ToListAsync();

            var httpClient = _httpClientFactory.CreateClient("StrokeApi");
            var wwwrootPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "frontend", "wwwroot");

            foreach (var kesit in kesitler)
            {
                var fizikselYol = Path.Combine(wwwrootPath, kesit.DosyaYolu!.TrimStart('/'));

                if (!System.IO.File.Exists(fizikselYol))
                {
                    continue; // dosya bulunamazsa bu kesiti atla
                }

                StrokeApiResponse? apiSonuc;

                try
                {
                    using var fileStream = System.IO.File.OpenRead(fizikselYol);
                    using var content = new MultipartFormDataContent();
                    using var streamContent = new StreamContent(fileStream);
                    streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
                    content.Add(streamContent, "file", Path.GetFileName(fizikselYol));

                    var response = await httpClient.PostAsync("api/stroke-segmentation", content);
                    response.EnsureSuccessStatusCode();

                    var json = await response.Content.ReadAsStringAsync();
                    apiSonuc = JsonSerializer.Deserialize<StrokeApiResponse>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                }
                catch (Exception)
                {
                    // Python API'ye ulaşılamazsa (kapalıysa, hata verirse) bu kesiti atla
                    continue;
                }

                if (apiSonuc == null) continue;

                var mevcutSonuc = kesit.SegmentasyonSonucus?.FirstOrDefault();

                if (apiSonuc.InmeVarMi)
                {
                    // model_versiyonu kolonu varchar(20) -- apiSonuc.ModelSurumu (ör.
                    // "unet_focal_tversky_ensemble_v1") bundan uzun olabilir, kesiyoruz.
                    string modelEtiketi = string.IsNullOrEmpty(apiSonuc.ModelSurumu)
                        ? "StrokeNext-CNN v1.0"
                        : (apiSonuc.ModelSurumu.Length > 20 ? apiSonuc.ModelSurumu[..20] : apiSonuc.ModelSurumu);

                    if (mevcutSonuc == null)
                    {
                        var yeniSonuc = new SegmentasyonSonucu
                        {
                            GoruntuId = kesit.GoruntuId,
                            ModelVersiyonu = modelEtiketi,
                            // NOT: MaskeDosyaYolu alanina, ekranda dogrudan gosterilebilir olan
                            // OverlayYolu (gri arka plan + macenta kontur) yaziliyor -- duz ikili
                            // maske (MaskeYolu) degil. CekimDetay.cshtml bu alani hazir bir
                            // gorsel olarak render ediyor, piksel-renk filtrelemesi yapmiyor.
                            MaskeDosyaYolu = apiSonuc.OverlayYolu ?? apiSonuc.MaskeYolu,
                            GuvenSkoru = (decimal)apiSonuc.GuvenSkoru,
                            IslemTarihi = DateTime.Now
                        };
                        _context.SegmentasyonSonucus.Add(yeniSonuc);
                    }
                    else
                    {
                        // Yeniden analiz edildiyse mevcut sonucu güncelle
                        mevcutSonuc.MaskeDosyaYolu = apiSonuc.OverlayYolu ?? apiSonuc.MaskeYolu;
                        mevcutSonuc.ModelVersiyonu = modelEtiketi;
                        mevcutSonuc.GuvenSkoru = (decimal)apiSonuc.GuvenSkoru;
                        mevcutSonuc.IslemTarihi = DateTime.Now;
                    }
                }
                else
                {
                    if (mevcutSonuc != null)
                    {
                        _context.SegmentasyonSonucus.Remove(mevcutSonuc);
                    }
                }
            }

            await _context.SaveChangesAsync();

            var ilkGoruntu = kesitler.FirstOrDefault();
            return RedirectToAction("CekimDetay", new { id = ilkGoruntu?.GoruntuId ?? 0 });
        }

        // 6. YENİ HASTA EKLEME SAYFASI (GET)
        [HttpGet]
        public IActionResult Ekle()
        {
            return View();
        }

        // 6.1 AJAX İLE TC SORGULAMA METODU
        [HttpGet]
        public async Task<IActionResult> TcKontrolEt(string tc)
        {
            if (string.IsNullOrWhiteSpace(tc)) return Json(new { varMi = false });

            var hasta = await _context.Hastas
                .FirstOrDefaultAsync(h => h.Tc == tc.Trim());

            if (hasta != null)
            {
                return Json(new
                {
                    varMi = true,
                    hastaId = hasta.HastaId,
                    isim = hasta.Isim,
                    soyad = hasta.Soyad,
                    cinsiyet = hasta.Cinsiyet,
                    dogumTarihi = hasta.DogumTarihi?.ToString("yyyy-MM-dd"),
                    telefon = hasta.Telefon
                });
            }

            return Json(new { varMi = false });
        }

        // 7. YENİ HASTA KAYIT İŞLEMİ / MEVCUT HASTAYI DOKTORUN LİSTESİNE BAĞLAMA (POST)
        [HttpPost]
        public async Task<IActionResult> Ekle(Hasta hasta)
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            if (claim == null || !int.TryParse(claim.Value, out int aktifDoktorId))
            {
                return RedirectToAction("Login", "Account");
            }

            // 1. Veritabanında bu TC ile kayıtlı hasta var mı?
            var mevcutHasta = await _context.Hastas.FirstOrDefaultAsync(h => h.Tc == hasta.Tc);

            if (mevcutHasta != null)
            {
                // Doktor bu hastayı daha önce takip listesine eklemiş mi kontrol et
                bool zatenTakipte = await _context.DoktorHastas
                    .AnyAsync(dh => dh.DoktorId == aktifDoktorId && dh.HastaId == mevcutHasta.HastaId);

                if (!zatenTakipte)
                {
                    // Doktor-Hasta ilişkisini ara tabloya ekliyoruz
                    _context.DoktorHastas.Add(new DoktorHasta
                    {
                        DoktorId = aktifDoktorId,
                        HastaId = mevcutHasta.HastaId,
                        EklenmeTarihi = DateTime.Now
                    });
                    await _context.SaveChangesAsync();
                }

                // Doğrudan HASTA YÖNETİM PANELİNE (INDEX) yönlendiriyoruz
                return RedirectToAction("Index");
            }

            // 2. Hasta sistemde hiç yoksa SIFIRDAN KAYDET
            if (ModelState.IsValid)
            {
                hasta.DoktorId = aktifDoktorId; // İlk oluşturan doktor
                _context.Hastas.Add(hasta);
                await _context.SaveChangesAsync();

                // İlişki tablosuna da kaydet
                _context.DoktorHastas.Add(new DoktorHasta
                {
                    DoktorId = aktifDoktorId,
                    HastaId = hasta.HastaId,
                    EklenmeTarihi = DateTime.Now
                });
                await _context.SaveChangesAsync();

                return RedirectToAction("Index");
            }

            return View(hasta);
        }

        // 8. MUAYENE RAPORU YAZDIRMA EKRANI (GET)
        [HttpGet]
        public async Task<IActionResult> Rapor(int id)
        {
            var muayene = await _context.Muayenes
                .Include(m => m.Hasta)
                .Include(m => m.Doktor)
                .FirstOrDefaultAsync(m => m.MuayeneId == id);

            if (muayene == null) return NotFound();

            return View(muayene);
        }

        // POST: /Hasta/MuayeneEkle
        [HttpPost]
        public async Task<IActionResult> MuayeneEkle(int hastaId, string teshis, string notlar)
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            if (claim == null || !int.TryParse(claim.Value, out int doktorId))
            {
                return RedirectToAction("Login", "Account");
            }

            if (!string.IsNullOrWhiteSpace(teshis))
            {
                var yeniMuayene = new Muayene
                {
                    HastaId = hastaId,
                    DoktorId = doktorId,
                    Tarih = DateTime.Now,
                    Teshis = teshis,
                    Notlar = notlar
                };

                _context.Muayenes.Add(yeniMuayene);
                await _context.SaveChangesAsync();
            }

            return RedirectToAction("Detay", new { id = hastaId });
        }

        // POST: /Hasta/SaglikGecmisiEkle
        [HttpPost]
        public async Task<IActionResult> SaglikGecmisiEkle(int hastaId, string hastalikAdi, string aciklama, bool kronikMi, DateOnly? taniTarihi)
        {
            if (!string.IsNullOrWhiteSpace(hastalikAdi))
            {
                var yeniGecmis = new SaglikGecmisi
                {
                    HastaId = hastaId,
                    HastalikAdi = hastalikAdi,
                    Aciklama = aciklama,
                    KronikMi = kronikMi,
                    TaniTarihi = taniTarihi ?? DateOnly.FromDateTime(DateTime.Now)
                };

                _context.SaglikGecmisis.Add(yeniGecmis);
                await _context.SaveChangesAsync();
            }

            return RedirectToAction("Detay", new { id = hastaId });
        }
    }

    // Python API'sinden dönen JSON'u karşılayan yardımcı sınıf
    public class StrokeApiResponse
    {
        public bool InmeVarMi { get; set; }
        public double GuvenSkoru { get; set; }
        public string? TeshisDetay { get; set; }
        public string? MaskeYolu { get; set; }
        public string? OverlayYolu { get; set; }
        public string? ModelSurumu { get; set; }
    }
}