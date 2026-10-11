using System;

namespace BTİNME.Models
{
    public class DoktorHasta
    {
        public int Id { get; set; }

        public int DoktorId { get; set; }
        public virtual Doktor Doktor { get; set; }

        public int HastaId { get; set; }
        public virtual Hasta Hasta { get; set; }

        public DateTime EklenmeTarihi { get; set; } = DateTime.Now;
    }
}