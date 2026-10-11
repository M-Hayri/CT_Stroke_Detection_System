using BTİNME.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace BTİNME.Controllers
{
    public class GoruntuController : Controller
    {
        private readonly AppDbContext _context;
        private readonly IWebHostEnvironment _environment;

        public GoruntuController(AppDbContext context, IWebHostEnvironment environment)
        {
            _context = context;
            _environment = environment;
        }

        // GET: /Goruntu/Yukle/1
        [HttpGet]
        public IActionResult Yukle(int id)
        {
            var hasta = _context.Hastas
                .Include(h => h.Goruntus)
                    .ThenInclude(g => g.SegmentasyonSonucus)
                .FirstOrDefault(h => h.HastaId == id);

            if (hasta == null)
            {
                return NotFound();
            }

            return View(hasta);
        }

        // POST: /Goruntu/Yukle
        [HttpPost]
        public async Task<IActionResult> Yukle(int hastaId, IFormFile goruntuDosyasi, string goruntuTuru, string aciklama)
        {
            if (goruntuDosyasi != null && goruntuDosyasi.Length > 0)
            {
                string uploadsFolder = Path.Combine(_environment.WebRootPath, "uploads");
                if (!Directory.Exists(uploadsFolder))
                {
                    Directory.CreateDirectory(uploadsFolder);
                }

                string uniqueFileName = Guid.NewGuid().ToString() + "_" + Path.GetFileName(goruntuDosyasi.FileName);
                string filePath = Path.Combine(uploadsFolder, uniqueFileName);

                using (var fileStream = new FileStream(filePath, FileMode.Create))
                {
                    await goruntuDosyasi.CopyToAsync(fileStream);
                }

                var goruntu = new Goruntu
                {
                    HastaId = hastaId,
                    GoruntuTuru = string.IsNullOrEmpty(goruntuTuru) ? "BT" : goruntuTuru,
                    DosyaYolu = "/uploads/" + uniqueFileName,
                    Aciklama = aciklama,
                    CekimTarihi = DateTime.Now
                };

                _context.Goruntus.Add(goruntu);
                await _context.SaveChangesAsync();
            }

            return RedirectToAction("Yukle", new { id = hastaId });
        }

        // Not: burada eskiden sabit GuvenSkoru = 0.9450m yazan ve orijinal
        // goruntuyu maske olarak kaydeden bir "SegmentasyonYap" action'i vardi;
        // gercek AI servisine hic baglanmadigi icin kaldirildi. Gercek
        // segmentasyon akisi HastaController.CekimAnalizEt icinde, gercek
        // FastAPI servisine HTTP istegiyle calisir.
    }
}
