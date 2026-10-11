using System;
using System.Collections.Generic;

namespace BTİNME.Models;

public partial class Hastane
{
    public int HastaneId { get; set; }
    public string Ad { get; set; } = null!;
    public string? Lokasyon { get; set; }
    public string? Adres { get; set; }
    public int? BasHekimId { get; set; }

    public virtual Doktor? BasHekim { get; set; }
    public virtual ICollection<Doktor> Doktorlar { get; set; } = new List<Doktor>();
}