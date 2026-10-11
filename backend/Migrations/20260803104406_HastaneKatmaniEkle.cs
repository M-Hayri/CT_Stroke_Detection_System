using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace BTİNME.Migrations
{
    /// <inheritdoc />
    public partial class HastaneKatmaniEkle : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "hastane_id",
                table: "doktor",
                type: "integer",
                nullable: true);

            migrationBuilder.CreateTable(
                name: "hastane",
                columns: table => new
                {
                    hastane_id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    ad = table.Column<string>(type: "character varying(150)", maxLength: 150, nullable: false),
                    lokasyon = table.Column<string>(type: "character varying(150)", maxLength: 150, nullable: true),
                    adres = table.Column<string>(type: "text", nullable: true),
                    bas_hekim_id = table.Column<int>(type: "integer", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("hastane_pkey", x => x.hastane_id);
                    table.ForeignKey(
                        name: "hastane_bas_hekim_id_fkey",
                        column: x => x.bas_hekim_id,
                        principalTable: "doktor",
                        principalColumn: "doktor_id",
                        onDelete: ReferentialAction.SetNull);
                });

            migrationBuilder.CreateIndex(
                name: "IX_doktor_hastane_id",
                table: "doktor",
                column: "hastane_id");

            migrationBuilder.CreateIndex(
                name: "IX_hastane_bas_hekim_id",
                table: "hastane",
                column: "bas_hekim_id");

            migrationBuilder.AddForeignKey(
                name: "doktor_hastane_id_fkey",
                table: "doktor",
                column: "hastane_id",
                principalTable: "hastane",
                principalColumn: "hastane_id",
                onDelete: ReferentialAction.SetNull);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "doktor_hastane_id_fkey",
                table: "doktor");

            migrationBuilder.DropTable(
                name: "hastane");

            migrationBuilder.DropIndex(
                name: "IX_doktor_hastane_id",
                table: "doktor");

            migrationBuilder.DropColumn(
                name: "hastane_id",
                table: "doktor");
        }
    }
}
