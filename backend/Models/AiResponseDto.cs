namespace BTİNME.Models
{
    public class AiResponseDto
    {
        public bool InmeVarMi { get; set; }
        public double GuvenSkoru { get; set; }
        public string TeshisDetay { get; set; } = string.Empty;
        public string MaskeYolu { get; set; } = string.Empty;
    }
}
