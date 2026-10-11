using System;
using System.Collections.Generic;

namespace BTİNME.Models;

public partial class MesajEk
{
    public int EkId { get; set; }
    public int MesajId { get; set; }
    public string DosyaYolu { get; set; } = null!;
    public string? DosyaAdi { get; set; }

    public virtual Mesaj Mesaj { get; set; } = null!;
}