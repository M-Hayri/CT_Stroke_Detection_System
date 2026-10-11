using System;
using System.Collections.Generic;
namespace BTİNME.Models;
public partial class Doktor
{
    public int DoktorId { get; set; }
    public string Tc { get; set; } = null!;
    public string Sifre { get; set; } = null!;
    public string Isim { get; set; } = null!;
    public string Soyad { get; set; } = null!;
    public string? UzmanlikAlani { get; set; }
    public string? Telefon { get; set; }
    public string? Email { get; set; }

    // Doktorun bağlı olduğu hastane
    public int? HastaneId { get; set; }
    public virtual Hastane? Hastane { get; set; }

    public virtual ICollection<Mesaj> MesajAliciDoktors { get; set; } = new List<Mesaj>();
    public virtual ICollection<Mesaj> MesajGonderenDoktors { get; set; } = new List<Mesaj>();
    public virtual ICollection<Muayene> Muayenes { get; set; } = new List<Muayene>();
}