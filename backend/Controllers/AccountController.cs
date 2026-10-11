using BTİNME.Models;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;

namespace BTİNME.Controllers
{
    public class AccountController : Controller
    {
        private readonly AppDbContext _context;

        public AccountController(AppDbContext context)
        {
            _context = context;
        }

        // GET: /Account/Login
        [HttpGet]
        public IActionResult Login()
        {
            if (User.Identity != null && User.Identity.IsAuthenticated)
            {
                return RedirectToAction("Index", "Hasta");
            }
            return View();
        }

        // POST: /Account/Login
        [HttpPost]
        public async Task<IActionResult> Login(string tc, string sifre)
        {
            if (string.IsNullOrEmpty(tc) || string.IsNullOrEmpty(sifre))
            {
                ViewBag.Error = "Lütfen TC Kimlik No ve Şifre giriniz.";
                return View();
            }

            // Hastane ve Başhekim bilgilerini de Include ederek doktoru getiriyoruz
            var doktor = await _context.Doktors
                .Include(d => d.Hastane)
                    .ThenInclude(h => h.BasHekim) // Eğer Hastane.cs içinde BasHekim ilişkisi tanımlıysa
                .FirstOrDefaultAsync(d => d.Tc == tc);

            bool sifreDogru = false;
            if (doktor != null)
            {
                if (doktor.Sifre.StartsWith("$2"))
                {
                    // bcrypt ile hash'lenmiş şifre
                    sifreDogru = BCrypt.Net.BCrypt.Verify(sifre, doktor.Sifre);
                }
                else
                {
                    // eski kayıtlardan kalma düz metin şifre: bu girişte doğrula,
                    // doğruysa veritabanında hash'e yükselt (tek seferlik geçiş,
                    // ayrı bir migration betiğine gerek kalmadan)
                    sifreDogru = doktor.Sifre == sifre;
                    if (sifreDogru)
                    {
                        doktor.Sifre = BCrypt.Net.BCrypt.HashPassword(sifre);
                        await _context.SaveChangesAsync();
                    }
                }
            }

            if (doktor != null && sifreDogru)
            {
                var claims = new List<Claim>
        {
            new Claim(ClaimTypes.NameIdentifier, doktor.DoktorId.ToString()),
            new Claim(ClaimTypes.Name, $"Dr. {doktor.Isim} {doktor.Soyad}"),
            new Claim("TC", doktor.Tc ?? ""),
            
            // --- HASTANE BİLGİLERİ EKLENDİ ---
            new Claim("HastaneAd", doktor.Hastane?.Ad ?? "Hastane Tanımsız"),
            new Claim("HastaneLokasyon", doktor.Hastane?.Lokasyon ?? "Belirtilmemiş"),
            new Claim("HastaneAdres", doktor.Hastane?.Adres ?? "Belirtilmemiş"),
            new Claim("HastaneBasHekim", doktor.Hastane?.BasHekim != null
                ? $"Dr. {doktor.Hastane.BasHekim.Isim} {doktor.Hastane.BasHekim.Soyad}"
                : "Başhekim Atanmamış")
        };

                var claimsIdentity = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);

                await HttpContext.SignInAsync(
                    CookieAuthenticationDefaults.AuthenticationScheme,
                    new ClaimsPrincipal(claimsIdentity));

                return RedirectToAction("Index", "Hasta");
            }

            ViewBag.Error = "TC Kimlik No veya Şifre hatalı!";
            return View();
        }

        // GET: /Account/Logout
        public async Task<IActionResult> Logout()
        {
            await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            return RedirectToAction("Login");
        }
    }
}