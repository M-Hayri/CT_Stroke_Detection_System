using BTİNME.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;

namespace BTİNME.Controllers
{
    [Authorize]
    public class MesajController : Controller
    {
        private readonly AppDbContext _context;
        public MesajController(AppDbContext context)
        {
            _context = context;
        }

        [HttpGet]
        public async Task<IActionResult> PanelIcerik(int? aliciId)
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            int aktifDoktorId = claim != null ? int.Parse(claim.Value) : 0;

            // Mesajlaşma yalnız aynı hastanedeki doktorlar arasında yapılabiliyor
            var aktifDoktor = await _context.Doktors.FindAsync(aktifDoktorId);
            int? aktifHastaneId = aktifDoktor?.HastaneId;

            // aktifHastaneId null ise (doktor henüz bir hastaneye atanmamışsa) liste boş döner
            var doktorlar = aktifHastaneId.HasValue
                ? await _context.Doktors
                    .Where(d => d.DoktorId != aktifDoktorId && d.HastaneId == aktifHastaneId)
                    .ToListAsync()
                : new List<Doktor>();

            List<Mesaj> mesajGecmisi = new List<Mesaj>();
            Doktor? seciliDoktor = null;

            if (aliciId.HasValue)
            {
                seciliDoktor = await _context.Doktors.FindAsync(aliciId.Value);

                var okunmamisMesajlar = await _context.Mesajs
                    .Where(m => m.GonderenDoktorId == aliciId.Value && m.AliciDoktorId == aktifDoktorId && (m.OkunduMu == false || m.OkunduMu == null))
                    .ToListAsync();

                if (okunmamisMesajlar.Any())
                {
                    foreach (var m in okunmamisMesajlar)
                    {
                        m.OkunduMu = true;
                    }
                    await _context.SaveChangesAsync();
                }

                mesajGecmisi = await _context.Mesajs
                    .Include(m => m.Ekler)
                    .Where(m => (m.GonderenDoktorId == aktifDoktorId && m.AliciDoktorId == aliciId.Value) ||
                                (m.GonderenDoktorId == aliciId.Value && m.AliciDoktorId == aktifDoktorId))
                    .OrderBy(m => m.GonderimTarihi)
                    .ToListAsync();
            }

            var doktorOzetleri = new List<MesajDoktorOzet>();
            foreach (var doc in doktorlar)
            {
                var unreadCount = await _context.Mesajs
                    .CountAsync(m => m.GonderenDoktorId == doc.DoktorId && m.AliciDoktorId == aktifDoktorId && (m.OkunduMu == false || m.OkunduMu == null));

                var sonMesaj = await _context.Mesajs
                    .Include(m => m.Ekler)
                    .Where(m => (m.GonderenDoktorId == aktifDoktorId && m.AliciDoktorId == doc.DoktorId) ||
                                (m.GonderenDoktorId == doc.DoktorId && m.AliciDoktorId == aktifDoktorId))
                    .OrderByDescending(m => m.GonderimTarihi)
                    .FirstOrDefaultAsync();

                string onizleme = "Henüz mesaj yok";
                DateTime? sonTarih = null;

                if (sonMesaj != null)
                {
                    sonTarih = sonMesaj.GonderimTarihi;
                    bool benimMesajimMi = sonMesaj.GonderenDoktorId == aktifDoktorId;
                    string onek = benimMesajimMi ? "Siz: " : "";

                    if (!string.IsNullOrWhiteSpace(sonMesaj.Icerik))
                    {
                        var kisaltilmis = sonMesaj.Icerik.Length > 34 ? sonMesaj.Icerik.Substring(0, 34) + "..." : sonMesaj.Icerik;
                        onizleme = onek + kisaltilmis;
                    }
                    else if (sonMesaj.Ekler != null && sonMesaj.Ekler.Any())
                    {
                        int ekSayisi = sonMesaj.Ekler.Count;
                        onizleme = onek + "📷 " + (ekSayisi == 1 ? "1 görsel" : ekSayisi + " görsel");
                    }
                }

                doktorOzetleri.Add(new MesajDoktorOzet
                {
                    DoktorId = doc.DoktorId,
                    Isim = doc.Isim ?? "",
                    Soyad = doc.Soyad ?? "",
                    UzmanlikAlani = doc.UzmanlikAlani,
                    OkunmamisSayisi = unreadCount,
                    SonMesajOnizleme = onizleme,
                    SonMesajTarihi = sonTarih
                });
            }

            ViewBag.DoktorOzetleri = doktorOzetleri
                .OrderByDescending(d => d.SonMesajTarihi ?? DateTime.MinValue)
                .ToList();

            ViewBag.AktifDoktorId = aktifDoktorId;
            ViewBag.AliciId = aliciId;
            ViewBag.SeciliDoktor = seciliDoktor;
            ViewBag.AktifDoktorHastaneVarMi = aktifHastaneId.HasValue; // view'da bilgilendirme mesajı için

            return PartialView("_MesajPanelPartial", mesajGecmisi);
        }

        [HttpPost]
        [RequestSizeLimit(50_000_000)]
        public async Task<IActionResult> Gonder(int aliciDoktorId, string? icerik, List<IFormFile>? dosyalar)
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            int aktifDoktorId = claim != null ? int.Parse(claim.Value) : 0;

            bool metinVar = !string.IsNullOrWhiteSpace(icerik);
            bool dosyaVar = dosyalar != null && dosyalar.Any(d => d.Length > 0);

            if ((metinVar || dosyaVar) && aktifDoktorId > 0)
            {
                var yeniMesaj = new Mesaj
                {
                    GonderenDoktorId = aktifDoktorId,
                    AliciDoktorId = aliciDoktorId,
                    Icerik = metinVar ? icerik!.Trim() : null,
                    GonderimTarihi = DateTime.Now,
                    OkunduMu = false
                };

                _context.Mesajs.Add(yeniMesaj);
                await _context.SaveChangesAsync();

                if (dosyaVar)
                {
                    var uploadsFolder = Path.Combine(Directory.GetCurrentDirectory(), "..", "frontend", "wwwroot", "uploads", "mesajlar");
                    if (!Directory.Exists(uploadsFolder))
                    {
                        Directory.CreateDirectory(uploadsFolder);
                    }

                    foreach (var dosya in dosyalar!)
                    {
                        if (dosya.Length <= 0) continue;

                        var uniqueFileName = Guid.NewGuid().ToString() + "_" + Path.GetFileName(dosya.FileName);
                        var filePath = Path.Combine(uploadsFolder, uniqueFileName);

                        using (var stream = new FileStream(filePath, FileMode.Create))
                        {
                            await dosya.CopyToAsync(stream);
                        }

                        _context.MesajEks.Add(new MesajEk
                        {
                            MesajId = yeniMesaj.MesajId,
                            DosyaYolu = "/uploads/mesajlar/" + uniqueFileName,
                            DosyaAdi = dosya.FileName
                        });
                    }

                    await _context.SaveChangesAsync();
                }
            }

            return await PanelIcerik(aliciDoktorId);
        }

        [HttpGet]
        public async Task<IActionResult> OkunmamisSayisi()
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            if (claim == null || !int.TryParse(claim.Value, out int aktifDoktorId))
            {
                return Json(0);
            }

            var sayi = await _context.Mesajs
                .CountAsync(m => m.AliciDoktorId == aktifDoktorId && (m.OkunduMu == false || m.OkunduMu == null));

            return Json(sayi);
        }

        // Radyoloji sayfasındaki "Mesaj Gönder" modalında listelenecek doktorlar.
        // Sadece aktif doktorla aynı hastanedeki doktorlar dönüyor.
        [HttpGet]
        public async Task<IActionResult> DoktorlariGetir()
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            int aktifDoktorId = claim != null ? int.Parse(claim.Value) : 0;

            var aktifDoktor = await _context.Doktors.FindAsync(aktifDoktorId);
            int? aktifHastaneId = aktifDoktor?.HastaneId;

            if (!aktifHastaneId.HasValue)
            {
                return Json(new List<object>()); // hastanesi olmayan doktor için boş liste
            }

            var doktorlar = await _context.Doktors
                .Where(d => d.DoktorId != aktifDoktorId && d.HastaneId == aktifHastaneId)
                .ToListAsync();

            var sonuc = new List<object>();
            foreach (var d in doktorlar)
            {
                var sonMesaj = await _context.Mesajs
                    .Where(m => (m.GonderenDoktorId == aktifDoktorId && m.AliciDoktorId == d.DoktorId) ||
                                (m.GonderenDoktorId == d.DoktorId && m.AliciDoktorId == aktifDoktorId))
                    .OrderByDescending(m => m.GonderimTarihi)
                    .FirstOrDefaultAsync();

                sonuc.Add(new
                {
                    doktorId = d.DoktorId,
                    isim = d.Isim,
                    soyad = d.Soyad,
                    uzmanlikAlani = d.UzmanlikAlani,
                    sonMesaj = sonMesaj?.Icerik
                });
            }

            return Json(sonuc);
        }

        [HttpPost]
        public async Task<IActionResult> KlasorGonder(int hastaId, string goruntuTuru, DateTime cekimTarihi, List<int> aliciDoktorIdler)
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier);
            int aktifDoktorId = claim != null ? int.Parse(claim.Value) : 0;

            if (aktifDoktorId <= 0 || aliciDoktorIdler == null || !aliciDoktorIdler.Any())
            {
                return Json(new { success = false, message = "Geçersiz doktor bilgisi." });
            }

            // GÜVENLİK: seçilen alıcıların gerçekten aynı hastaneden olduğunu doğruluyoruz
            // (istemci tarafındaki liste manipüle edilse bile sunucu tarafında kural korunuyor)
            var aktifDoktor = await _context.Doktors.FindAsync(aktifDoktorId);
            if (aktifDoktor?.HastaneId == null)
            {
                return Json(new { success = false, message = "Bir hastaneye atanmadığınız için mesaj gönderemezsiniz." });
            }

            var geciliAliciIdler = await _context.Doktors
                .Where(d => aliciDoktorIdler.Contains(d.DoktorId) && d.HastaneId == aktifDoktor.HastaneId)
                .Select(d => d.DoktorId)
                .ToListAsync();

            if (!geciliAliciIdler.Any())
            {
                return Json(new { success = false, message = "Seçilen doktorlar aynı hastaneden değil." });
            }

            var kesitler = await _context.Goruntus
                .Where(g => g.HastaId == hastaId && g.GoruntuTuru == goruntuTuru && g.CekimTarihi == cekimTarihi)
                .OrderBy(g => g.GoruntuId)
                .ToListAsync();

            if (!kesitler.Any())
            {
                return Json(new { success = false, message = "Klasörde kesit bulunamadı." });
            }

            var hasta = await _context.Hastas.FindAsync(hastaId);
            var klasorAdi = kesitler.First().Aciklama ?? "Çekim Serisi";
            var mesajMetni = $"📁 {hasta?.Isim} {hasta?.Soyad} — {klasorAdi} ({kesitler.Count} kesit) paylaşıldı. Bu bölgede inme var mı incelermisiniz?";

            foreach (var aliciId in geciliAliciIdler.Distinct())
            {
                var yeniMesaj = new Mesaj
                {
                    GonderenDoktorId = aktifDoktorId,
                    AliciDoktorId = aliciId,
                    Icerik = mesajMetni,
                    GonderimTarihi = DateTime.Now,
                    OkunduMu = false
                };

                _context.Mesajs.Add(yeniMesaj);
                await _context.SaveChangesAsync();

                foreach (var kesit in kesitler)
                {
                    _context.MesajEks.Add(new MesajEk
                    {
                        MesajId = yeniMesaj.MesajId,
                        DosyaYolu = kesit.DosyaYolu ?? "",
                        DosyaAdi = kesit.Aciklama
                    });
                }

                await _context.SaveChangesAsync();
            }

            return Json(new { success = true, gonderilenSayisi = geciliAliciIdler.Distinct().Count() });
        }
    }
}