using System;
using System.Collections.Generic;
namespace BTİNME.Models;
public partial class Mesaj
{
    public int MesajId { get; set; }
    public int GonderenDoktorId { get; set; }
    public int AliciDoktorId { get; set; }
    public string? Icerik { get; set; } // boş olabilir: yalnız resim/dosya da gönderilebilir
    public DateTime? GonderimTarihi { get; set; }
    public bool? OkunduMu { get; set; }
    public virtual Doktor AliciDoktor { get; set; } = null!;
    public virtual Doktor GonderenDoktor { get; set; } = null!;
    public virtual ICollection<MesajEk> Ekler { get; set; } = new List<MesajEk>();
}