# 2. Teknoloji yığını

- **Durum:** Kabul edildi
- **Tarih:** 2026-11-11

## Bağlam

Kısıtlar: sıfır bütçe (açık kaynak, yerel veya ücretsiz katman), tek geliştirici,
Windows 11 + WSL2, RTX 5060 Ti 8 GB VRAM, 32 GB RAM. Amaç, senkron ajan mimarisine kıyasla
predictive prefetching'in gecikme kazancını ölçülebilir biçimde göstermek.

## Karar

| Alan | Seçim | Gerekçe | Değerlendirilen alternatifler |
|---|---|---|---|
| Dil | Python 3.12 | LLM/ajan ekosistemi, asyncio, tek dil | Go (ekosistem zayıf), TypeScript |
| Paket yönetimi | uv | Hızlı, lock dosyası, Python sürümünü de yönetir | Poetry, pip-tools |
| Cache | Valkey | BSD-3, Redis protokol uyumlu, pub/sub | Redis 8 (AGPL/SSPL), Dragonfly (BSL) |
| Event bus | NATS JetStream | Tek binary, düşük gecikme, kalıcılık, Apache 2.0 | Kafka (ağır), Redpanda (BSL), Valkey Streams |
| Ajan çatısı | LangGraph | Durum grafiği, araç düğümleri, MIT | Saf asyncio (çok iş), CrewAI |
| LLM sunucusu | Ollama | 8 GB VRAM'e uygun, Windows'ta GPU desteği, OpenAI uyumlu API | vLLM (8 GB'ta sıkışık, Windows desteği yok) |
| Ana model | qwen3.5:9b-q4_K_M, düşünme kapalı | %100 GPU, ~61 tok/s | 14B (VRAM'e sığmıyor) |
| Gözlemlenebilirlik | OpenTelemetry + Jaeger | Vendor bağımsız, kritik yol analizi | Langfuse (sonra eklenebilir) |
| Gecikme simülasyonu | Toxiproxy | Dış API'lere deterministik gecikme/hata enjeksiyonu | tc/netem |
| Yük testi | Locust | Python, tek dil | k6 |

LLM erişimi yalnızca OpenAI uyumlu bir istemci üzerinden yapılır; sağlayıcı değişimi
(`PA_LLM_BASE_URL`, `PA_LLM_MODEL`) konfigürasyonla olur.

## Sonuçlar

- Ollama Windows'ta yerel çalışır; WSL2 `networkingMode=mirrored` ile `localhost` üzerinden erişilir.
- Düşünme modu ajan için kapalıdır: açıkken tek cümlelik yanıt ~3.100 token ve ~50 sn sürdü.
- vLLM, M5'te kiralık GPU üzerinde ayrıca değerlendirilecek.
