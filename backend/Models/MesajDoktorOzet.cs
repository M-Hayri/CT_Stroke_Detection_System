using System;

namespace BTİNME.Models;

// Bu bir veritabanı tablosu DEĞİL — sadece konsültasyon panelindeki
// doktor listesini (WhatsApp tarzı: son mesaj + saat + okunmamış sayacı) 
// view'a taşımak için kullanılan yardımcı bir görünüm modelidir.
public class MesajDoktorOzet
{
    public int DoktorId { get; set; }
    public string Isim { get; set; } = string.Empty;
    public string Soyad { get; set; } = string.Empty;
    public string? UzmanlikAlani { get; set; }
    public int OkunmamisSayisi { get; set; }
    public string SonMesajOnizleme { get; set; } = string.Empty;
    public DateTime? SonMesajTarihi { get; set; }
}