using System;
using System.Collections.Generic;

namespace BTİNME.Models;

public partial class SegmentasyonSonucu
{
    public int SonucId { get; set; }

    public int GoruntuId { get; set; }

    public string? ModelVersiyonu { get; set; }

    public string? MaskeDosyaYolu { get; set; }

    public decimal? GuvenSkoru { get; set; }

    public DateTime? IslemTarihi { get; set; }

    public virtual Goruntu Goruntu { get; set; } = null!;
}
