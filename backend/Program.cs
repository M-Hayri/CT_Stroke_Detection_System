using BTİNME.Models;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;

// Statik dosyalar (css/js/images/lib/uploads) proje klasorunun degil, repo
// kokundeki frontend/wwwroot klasorunun icinden sunulur (backend+frontend
// ayri klasorlerde oldugu icin).
var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    WebRootPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "frontend", "wwwroot"),
});

// 1. MVC Servislerini Ekle
builder.Services.AddControllersWithViews();

// 2. PostgreSQL DbContext Bağlantısı
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

// 3. Cookie (Çerez) Tabanlı Kimlik Doğrulama Servisi
builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.LoginPath = "/Account/Login";
        options.LogoutPath = "/Account/Logout";
        options.ExpireTimeSpan = TimeSpan.FromHours(8);
    });

// 4. Yapay zeka servisine (Python/FastAPI) istek atmak için HttpClient kaydı
builder.Services.AddHttpClient("StrokeApi", client =>
{
    client.BaseAddress = new Uri("http://localhost:8000/");
    client.Timeout = TimeSpan.FromSeconds(60); // segmentasyon işlemi zaman alabilir
});

var app = builder.Build();

// 5. HTTP İstek Hattı (Pipeline)
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();

// 6. Kimlik Doğrulama ve Yetkilendirme (Sıralama Önemli!)
app.UseAuthentication();
app.UseAuthorization();

// 7. Varsayılan Rota (Dogrudan Giriş Sayfasına)
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Account}/{action=Login}/{id?}");

app.Run();
