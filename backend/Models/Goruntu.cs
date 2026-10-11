using System;
using System.Collections.Generic;

namespace BTİNME.Models;

public partial class Goruntu
{
    public int GoruntuId { get; set; }

    public int? HastaId { get; set; }

    public string? GoruntuTuru { get; set; }

    public string? DosyaYolu { get; set; }

    public string? Aciklama { get; set; }

    public DateTime? CekimTarihi { get; set; }

    public virtual Hasta? Hasta { get; set; }

    public virtual ICollection<SegmentasyonSonucu> SegmentasyonSonucus { get; set; } = new List<SegmentasyonSonucu>();
}