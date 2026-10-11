using System;
using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;

namespace BTİNME.Models;

public partial class AppDbContext : DbContext
{
    public AppDbContext()
    {
    }

    public AppDbContext(DbContextOptions<AppDbContext> options)
        : base(options)
    {
    }

    public virtual DbSet<Doktor> Doktors { get; set; }

    public virtual DbSet<Goruntu> Goruntus { get; set; }

    public virtual DbSet<Hasta> Hastas { get; set; }

    public virtual DbSet<Hastane> Hastanes { get; set; }

    public virtual DbSet<DoktorHasta> DoktorHastas { get; set; }

    public virtual DbSet<Mesaj> Mesajs { get; set; }

    public virtual DbSet<MesajEk> MesajEks { get; set; }

    public virtual DbSet<Muayene> Muayenes { get; set; }

    public virtual DbSet<SaglikGecmisi> SaglikGecmisis { get; set; }

    public virtual DbSet<SegmentasyonSonucu> SegmentasyonSonucus { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        => optionsBuilder.UseNpgsql("Host=localhost;Database=btinmeprogrami;Username=postgres;Password=12345");

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder
            .HasPostgresEnum("kan_grubu", new[] { "A+", "A-", "B+", "B-", "AB+", "AB-", "0+", "0-" })
            .HasPostgresEnum("segmentasyon_durum", new[] { "bekliyor", "tamamlandi", "hata" });

        modelBuilder.Entity<DoktorHasta>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("doktor_hasta_pkey");

            entity.ToTable("doktor_hasta");

            entity.Property(e => e.Id).HasColumnName("id");
            entity.Property(e => e.DoktorId).HasColumnName("doktor_id");
            entity.Property(e => e.HastaId).HasColumnName("hasta_id");
            entity.Property(e => e.EklenmeTarihi)
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasColumnType("timestamp without time zone")
                .HasColumnName("eklenme_tarihi");

            entity.HasOne(d => d.Doktor)
                .WithMany()
                .HasForeignKey(d => d.DoktorId)
                .HasConstraintName("doktor_hasta_doktor_id_fkey");

            entity.HasOne(d => d.Hasta)
                .WithMany()
                .HasForeignKey(d => d.HastaId)
                .HasConstraintName("doktor_hasta_hasta_id_fkey");
        });

        modelBuilder.Entity<Doktor>(entity =>
        {
            entity.HasKey(e => e.DoktorId).HasName("doktor_pkey");

            entity.ToTable("doktor");

            entity.HasIndex(e => e.Email, "doktor_email_key").IsUnique();

            entity.HasIndex(e => e.Tc, "doktor_tc_key").IsUnique();

            entity.Property(e => e.DoktorId).HasColumnName("doktor_id");
            entity.Property(e => e.Email)
                .HasMaxLength(100)
                .HasColumnName("email");
            entity.Property(e => e.Isim)
                .HasMaxLength(50)
                .HasColumnName("isim");
            entity.Property(e => e.Sifre)
                .HasMaxLength(255)
                .HasColumnName("sifre");
            entity.Property(e => e.Soyad)
                .HasMaxLength(30)
                .HasColumnName("soyad");
            entity.Property(e => e.Tc)
                .HasMaxLength(11)
                .HasColumnName("tc");
            entity.Property(e => e.Telefon)
                .HasMaxLength(15)
                .HasColumnName("telefon");
            entity.Property(e => e.UzmanlikAlani)
                .HasMaxLength(50)
                .HasColumnName("uzmanlik_alani");

            entity.Property(e => e.HastaneId).HasColumnName("hastane_id");

            entity.HasOne(d => d.Hastane)
                .WithMany(h => h.Doktorlar)
                .HasForeignKey(d => d.HastaneId)
                .OnDelete(DeleteBehavior.SetNull)
                .HasConstraintName("doktor_hastane_id_fkey");
        });

        modelBuilder.Entity<Goruntu>(entity =>
        {
            entity.HasKey(e => e.GoruntuId).HasName("goruntu_pkey");

            entity.ToTable("goruntu");

            entity.Property(e => e.GoruntuId).HasColumnName("goruntu_id");
            entity.Property(e => e.GoruntuTuru)
                .HasMaxLength(10)
                .HasColumnName("goruntu_turu");
            entity.Property(e => e.Aciklama).HasColumnName("aciklama");
            entity.Property(e => e.CekimTarihi)
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasColumnType("timestamp without time zone")
                .HasColumnName("cekim_tarihi");
            entity.Property(e => e.DosyaYolu).HasColumnName("dosya_yolu");
            entity.Property(e => e.HastaId).HasColumnName("hasta_id");

            entity.HasOne(d => d.Hasta).WithMany(p => p.Goruntus)
                .HasForeignKey(d => d.HastaId)
                .HasConstraintName("goruntu_hasta_id_fkey");
        });

        modelBuilder.Entity<Hasta>(entity =>
        {
            entity.HasKey(e => e.HastaId).HasName("hasta_pkey");

            entity.ToTable("hasta");

            entity.HasIndex(e => e.Tc, "hasta_tc_key").IsUnique();

            entity.Property(e => e.HastaId).HasColumnName("hasta_id");
            entity.Property(e => e.Adres).HasColumnName("adres");
            entity.Property(e => e.Cinsiyet)
                .HasMaxLength(10)
                .HasColumnName("cinsiyet");
            entity.Property(e => e.DogumTarihi).HasColumnName("dogum_tarihi");
            entity.Property(e => e.Isim)
                .HasMaxLength(50)
                .HasColumnName("isim");
            entity.Property(e => e.Soyad)
                .HasMaxLength(30)
                .HasColumnName("soyad");
            entity.Property(e => e.Tc)
                .HasMaxLength(11)
                .HasColumnName("tc");
            entity.Property(e => e.Telefon)
                .HasMaxLength(15)
                .HasColumnName("telefon");

            entity.Property(e => e.DoktorId).HasColumnName("doktor_id");

            entity.HasOne(d => d.Doktor)
                .WithMany()
                .HasForeignKey(d => d.DoktorId)
                .HasConstraintName("hasta_doktor_id_fkey");
        });

        modelBuilder.Entity<Hastane>(entity =>
        {
            entity.HasKey(e => e.HastaneId).HasName("hastane_pkey");

            entity.ToTable("hastane");

            entity.Property(e => e.HastaneId).HasColumnName("hastane_id");
            entity.Property(e => e.Ad)
                .HasMaxLength(150)
                .HasColumnName("ad");
            entity.Property(e => e.Lokasyon)
                .HasMaxLength(150)
                .HasColumnName("lokasyon");
            entity.Property(e => e.Adres).HasColumnName("adres");
            entity.Property(e => e.BasHekimId).HasColumnName("bas_hekim_id");

            entity.HasOne(d => d.BasHekim)
                .WithMany()
                .HasForeignKey(d => d.BasHekimId)
                .OnDelete(DeleteBehavior.SetNull)
                .HasConstraintName("hastane_bas_hekim_id_fkey");
        });

        modelBuilder.Entity<Mesaj>(entity =>
        {
            entity.HasKey(e => e.MesajId).HasName("mesaj_pkey");

            entity.ToTable("mesaj");

            entity.Property(e => e.MesajId).HasColumnName("mesaj_id");
            entity.Property(e => e.AliciDoktorId).HasColumnName("alici_doktor_id");
            entity.Property(e => e.GonderenDoktorId).HasColumnName("gonderen_doktor_id");
            entity.Property(e => e.GonderimTarihi)
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasColumnType("timestamp without time zone")
                .HasColumnName("gonderim_tarihi");
            entity.Property(e => e.Icerik).HasColumnName("icerik");
            entity.Property(e => e.OkunduMu)
                .HasDefaultValue(false)
                .HasColumnName("okundu_mu");

            entity.HasOne(d => d.AliciDoktor).WithMany(p => p.MesajAliciDoktors)
                .HasForeignKey(d => d.AliciDoktorId)
                .HasConstraintName("mesaj_alici_doktor_id_fkey");

            entity.HasOne(d => d.GonderenDoktor).WithMany(p => p.MesajGonderenDoktors)
                .HasForeignKey(d => d.GonderenDoktorId)
                .HasConstraintName("mesaj_gonderen_doktor_id_fkey");
        });

        modelBuilder.Entity<MesajEk>(entity =>
        {
            entity.HasKey(e => e.EkId).HasName("mesaj_ek_pkey");

            entity.ToTable("mesaj_ek");

            entity.Property(e => e.EkId).HasColumnName("ek_id");
            entity.Property(e => e.MesajId).HasColumnName("mesaj_id");
            entity.Property(e => e.DosyaYolu)
                .HasColumnName("dosya_yolu");
            entity.Property(e => e.DosyaAdi)
                .HasMaxLength(255)
                .HasColumnName("dosya_adi");

            entity.HasOne(d => d.Mesaj).WithMany(p => p.Ekler)
                .HasForeignKey(d => d.MesajId)
                .HasConstraintName("mesaj_ek_mesaj_id_fkey")
                .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<Muayene>(entity =>
        {
            entity.HasKey(e => e.MuayeneId).HasName("muayene_pkey");

            entity.ToTable("muayene");

            entity.Property(e => e.MuayeneId).HasColumnName("muayene_id");
            entity.Property(e => e.DoktorId).HasColumnName("doktor_id");
            entity.Property(e => e.HastaId).HasColumnName("hasta_id");
            entity.Property(e => e.Notlar).HasColumnName("notlar");
            entity.Property(e => e.Tarih)
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasColumnType("timestamp without time zone")
                .HasColumnName("tarih");
            entity.Property(e => e.Teshis)
                .HasMaxLength(255)
                .HasColumnName("teshis");

            entity.HasOne(d => d.Doktor).WithMany(p => p.Muayenes)
                .HasForeignKey(d => d.DoktorId)
                .HasConstraintName("muayene_doktor_id_fkey");

            entity.HasOne(d => d.Hasta).WithMany(p => p.Muayenes)
                .HasForeignKey(d => d.HastaId)
                .HasConstraintName("muayene_hasta_id_fkey");
        });

        modelBuilder.Entity<SaglikGecmisi>(entity =>
        {
            entity.HasKey(e => e.GecmisId).HasName("saglik_gecmisi_pkey");

            entity.ToTable("saglik_gecmisi");

            entity.Property(e => e.GecmisId).HasColumnName("gecmis_id");
            entity.Property(e => e.Aciklama).HasColumnName("aciklama");
            entity.Property(e => e.HastaId).HasColumnName("hasta_id");
            entity.Property(e => e.HastalikAdi)
                .HasMaxLength(100)
                .HasColumnName("hastalik_adi");
            entity.Property(e => e.KronikMi)
                .HasDefaultValue(false)
                .HasColumnName("kronik_mi");
            entity.Property(e => e.TaniTarihi).HasColumnName("tani_tarihi");

            entity.HasOne(d => d.Hasta).WithMany(p => p.SaglikGecmisis)
                .HasForeignKey(d => d.HastaId)
                .HasConstraintName("saglik_gecmisi_hasta_id_fkey");
        });

        modelBuilder.Entity<SegmentasyonSonucu>(entity =>
        {
            entity.HasKey(e => e.SonucId).HasName("segmentasyon_sonucu_pkey");

            entity.ToTable("segmentasyon_sonucu");

            entity.Property(e => e.SonucId).HasColumnName("sonuc_id");
            entity.Property(e => e.GoruntuId).HasColumnName("goruntu_id");
            entity.Property(e => e.GuvenSkoru)
                .HasPrecision(5, 4)
                .HasColumnName("guven_skoru");
            entity.Property(e => e.IslemTarihi)
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasColumnType("timestamp without time zone")
                .HasColumnName("islem_tarihi");
            entity.Property(e => e.MaskeDosyaYolu).HasColumnName("maske_dosya_yolu");
            entity.Property(e => e.ModelVersiyonu)
                .HasMaxLength(20)
                .HasColumnName("model_versiyonu");

            entity.HasOne(d => d.Goruntu).WithMany(p => p.SegmentasyonSonucus)
                .HasForeignKey(d => d.GoruntuId)
                .HasConstraintName("segmentasyon_sonucu_goruntu_id_fkey");
        });

        OnModelCreatingPartial(modelBuilder);
    }

    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}