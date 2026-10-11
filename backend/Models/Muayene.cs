using System;
using System.Collections.Generic;

namespace BTİNME.Models;

public partial class Muayene
{
    public int MuayeneId { get; set; }

    public int? DoktorId { get; set; }

    public int? HastaId { get; set; }

    public DateTime? Tarih { get; set; }

    public string? Teshis { get; set; }

    public string? Notlar { get; set; }

    public virtual Doktor? Doktor { get; set; }

    public virtual Hasta? Hasta { get; set; }
}