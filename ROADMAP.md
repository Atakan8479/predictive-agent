# ROADMAP — Predictive Agent Orchestration

Event-driven, asenkron bir LLM ajan orkestrasyon altyapısı. Router, ana ajan başlamadan önce
ihtiyaç duyulacak dış verileri tahmin edip arka planda çeker (predictive prefetching).
Ana ajan bu veriyi ağ gecikmesi yaşamadan in-memory cache'ten okur.

**Kısıtlar:** Sıfır bütçe (açık kaynak + yerel + ücretsiz katman) · Windows 11 + WSL2 ·
RTX 5060 Ti 8 GB · 32 GB RAM

**Temel ilke:** Prefetch saf bir optimizasyondur. `PREFETCH_ENABLED=false` iken sistem
senkron modda doğru çalışmalıdır. Tüm ölçümler bu iki modun karşılaştırmasına dayanır.

---

## M0 — Geliştirme ortamı

| Görev | Çıktı |
|---|---|
| WSL2 + Ubuntu, Docker, Ollama, uv, git, gh, VS Code kurulumu | Tüm versiyon kontrolleri geçer |
| Ollama'da ana model (7–8B Q4) ve küçük model (3–4B Q4) | `ollama run` ile GPU'da yanıt alınır |
| GitHub reposu, branch koruması, PR şablonu | `main`'e doğrudan push kapalı |
| `pyproject.toml`: uv, ruff, pyright, pytest ayarları; pre-commit | `uv run pytest` ve `ruff check` temiz çalışır |
| GitHub Actions CI: lint + type check + test | Her PR'da yeşil CI |
| `docs/adr/0001-record-architecture-decisions.md` | ADR süreci başlar |

**Bitti sayılır:** Boş proje CI'dan geçiyor, Ollama GPU'da çalışıyor, WSL'den erişilebiliyor.

---

## M1 — Temel ve senkron baseline

| Görev | Çıktı |
|---|---|
| `docker-compose.yml`: Valkey, NATS, Toxiproxy, Jaeger, mock servisler | `docker compose up` ile her şey ayağa kalkar |
| Mock dış API'ler (FastAPI): OMS, kargo, CRM, iade politikası | Deterministik sahte veri döner |
| Toxiproxy profilleri: 50 / 200 / 800 ms + jitter + hata oranı | Profil tek komutla değiştirilir |
| Mock LLM (OpenAI uyumlu, gecikmesi ve token hızı ayarlanabilir, streaming) | Token harcamadan yük testi yapılabilir |
| LLM katmanı: `base_url` / `model` ortam değişkeninden; Ollama, Groq, mock | Sağlayıcı değişimi sadece konfigürasyon |
| Tool Registry (Pydantic): `idempotent`, `freshness_ttl`, `auth_scope`, `depends_on` | Araç sözleşmeleri tek yerde |
| Gateway (FastAPI) + senkron LangGraph ajanı | Uçtan uca çalışan senkron sistem |
| OpenTelemetry → Jaeger; tüm span'lerde `request_id` | Kritik yol görünür |
| İlk ölçüm: gerçek modelin token/sn değeri, senkron gecikme dökümü | `docs/results/m1-baseline.md` |

**Bitti sayılır:** Araç beklemesinin toplam gecikmedeki payı ölçülmüş ve raporlanmış.

---

## M2 — Prefetch çekirdeği

| Görev | Çıktı |
|---|---|
| Valkey durum makinesi: `PENDING / READY / ERROR`, araç başına TTL | Anahtar şeması: `tenant:user:scope:tool:sha256(args)` |
| `get_or_fetch(key, deadline)`: READY → döndür, PENDING → bekle, MISS/ERROR/timeout → senkron çek | Tek giriş noktası |
| Single-flight: `SET NX PX` + pub/sub ile hazır bildirimi | Aynı veri için tek dış çağrı |
| NATS JetStream: `prefetch.<tool>` konuları, retry, DLQ | Kalıcı event akışı |
| Fetch worker'lar (FastStream + httpx): kullanıcı kapsamıyla çağrı, semaphore ile limit | Idempotent tüketiciler |
| Oracle router: tahminleri veri seti etiketlerinden alır | Teorik üst sınır ölçülebilir |
| Ön-bağlam: deadline'a kadar hazır veriler prompt'a gömülür | Araç çağrısı token'ları azalır |
| İptal: `request_id` bazlı iptal event'i | Yarım kalan işler temizlenir |
| Testler: 100 eşzamanlı istek → 1 dış çağrı; PENDING sırasında okuma; TTL aşımı; Valkey kapalıyken fallback | Race condition'lar test altında |

**Bitti sayılır:** Oracle router ile sync ve predictive karşılaştırması yapılmış, üst sınır raporlanmış.

---

## M3 — Gerçek router, prompt'lar ve değerlendirme

| Görev | Çıktı |
|---|---|
| Etiketli veri seti: 300–500 Türkçe sorgu → beklenen araçlar | `bench/dataset/` |
| Embedding router: çok dilli küçük embedding modeli + sınıflandırıcı (CPU) | Gecikme < 20 ms |
| Güven eşiği ve istek başına prefetch bütçesi | Boşa çekim kontrol altında |
| Router metrikleri: precision, recall, confusion matrix | `docs/results/m3-router.md` |
| Prompt'ların versiyonlanmış dosyalarda tutulması | `prompts/` + changelog |
| Kalite eval seti (yerel model yargıç olarak) | Prompt değişiklikleri ölçülerek yapılır |
| Opsiyonel: argüman çıkarımı için küçük LLM (structured output) | Router + LLM hibrit |

**Bitti sayılır:** Gerçek router'ın oracle'a göre kazancın yüzde kaçını yakaladığı biliniyor.

---

## M4 — Benchmark ve rapor

| Görev | Çıktı |
|---|---|
| Gecikme matrisi: {API gecikmesi} × {eşzamanlılık} × {sync / oracle / gerçek router} | Ham veri `bench/results/` |
| Locust yük testleri (mock LLM ile) | Throughput ve kırılma noktası |
| Gerçek modelle kalite paritesi | Kalite düşüşü yok kanıtı |
| Analiz notebook'u: p50/p95/p99, bootstrap CI, Mann-Whitney U | Grafikler |
| Ağır yükte prefetch'i otomatik kapatan devre kesici | Yük altında güvenli davranış |

**Bitti sayılır:** README'de ölçülmüş sonuçlar ve grafikler var.

---

## M5 — Deployment (sıfır bütçe)

| Görev | Çıktı |
|---|---|
| Servis başına Dockerfile (multi-stage, non-root) | Küçük, güvenli imajlar |
| GitHub Actions: imaj build → GHCR | Her tag'de yayınlanan imajlar |
| k3d ile yerel Kubernetes cluster | Üretime benzer ortam |
| Helm chart: tüm servisler, ConfigMap/Secret, probe'lar, kaynak limitleri | `helm install` ile kurulum |
| Ajan ve Valkey için pod affinity, NetworkPolicy | Prod ağ kaygıları ele alınmış |
| Feature flag ile shadow → kanarya senaryosu | Deployment stratejisi uygulanmış |
| Opsiyonel: kiralık GPU'da vLLM ile tek seferlik benchmark (~10–30 $) | GPU sunucusunda sonuçlar |

**Bitti sayılır:** Sistem k3d üzerinde Helm ile kuruluyor, CI imajları yayınlıyor.

---

## Çalışma kuralları

- Her değişiklik feature branch → PR → CI yeşil → review → merge.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Önemli her teknik karar için bir ADR.
- Ölçülmeyen iyileştirme iyileştirme sayılmaz: Her milestone bir sonuç raporuyla kapanır.
