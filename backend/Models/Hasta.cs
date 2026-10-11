using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;

namespace BTİNME.Models;

public partial class Hasta
{
    public int HastaId { get; set; }

    public string Tc { get; set; } = null!;

    public string Isim { get; set; } = null!;

    public string Soyad { get; set; } = null!;

    public DateOnly? DogumTarihi { get; set; }

    public string? Cinsiyet { get; set; }

    public string? Telefon { get; set; }

    public string? Adres { get; set; }

    // POSTGRESQL SÜTUN EŞLEŞTİRMESİ
    [Column("doktor_id")]
    public int? DoktorId { get; set; }

    public virtual Doktor? Doktor { get; set; }

    public virtual ICollection<Goruntu> Goruntus { get; set; } = new List<Goruntu>();

    public virtual ICollection<Muayene> Muayenes { get; set; } = new List<Muayene>();

    public virtual ICollection<SaglikGecmisi> SaglikGecmisis { get; set; } = new List<SaglikGecmisi>();
}