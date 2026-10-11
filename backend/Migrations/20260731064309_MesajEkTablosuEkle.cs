using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace BTİNME.Migrations
{
    /// <inheritdoc />
    public partial class MesajEkTablosuEkle : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "mesaj_ek",
                columns: table => new
                {
                    ek_id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    mesaj_id = table.Column<int>(type: "integer", nullable: false),
                    dosya_yolu = table.Column<string>(type: "text", nullable: false),
                    dosya_adi = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("mesaj_ek_pkey", x => x.ek_id);
                    table.ForeignKey(
                        name: "mesaj_ek_mesaj_id_fkey",
                        column: x => x.mesaj_id,
                        principalTable: "mesaj",
                        principalColumn: "mesaj_id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_mesaj_ek_mesaj_id",
                table: "mesaj_ek",
                column: "mesaj_id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "mesaj_ek");
        }
    }
}