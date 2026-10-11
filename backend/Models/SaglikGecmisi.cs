using System;
using System.Collections.Generic;

namespace BTİNME.Models;

public partial class SaglikGecmisi
{
    public int GecmisId { get; set; }

    public int HastaId { get; set; }

    public string HastalikAdi { get; set; } = null!;

    public DateOnly? TaniTarihi { get; set; }

    public string? Aciklama { get; set; }

    public bool? KronikMi { get; set; }

    public virtual Hasta Hasta { get; set; } = null!;
}
