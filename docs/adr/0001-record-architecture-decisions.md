# 1. Mimari kararları kayıt altına almak

- **Durum:** Kabul edildi
- **Tarih:** 2026-11-11

## Bağlam

Proje birçok bileşen (router, event bus, cache, ajan, LLM sunucusu) içeriyor ve her biri için
birden fazla makul seçenek var. Kararların gerekçesi zamanla unutulur; "neden X?" sorusu hem
gelecekteki değişikliklerde hem de projenin anlatımında tekrar tekrar sorulacak.

## Karar

Önemli her teknik karar, Michael Nygard formatında kısa bir ADR olarak `docs/adr/` altında
numaralı dosyalarla kaydedilir: Bağlam → Karar → Sonuçlar. Kabul edilmiş bir ADR değiştirilmez;
karar değişirse yeni bir ADR yazılır ve eskisi "Yerini aldı: ADR-N" olarak işaretlenir.

## Sonuçlar

- Kararların gerekçesi kodla birlikte versiyonlanır.
- Küçük bir yazma maliyeti var; sadece geri dönmesi pahalı kararlar için yazılır.
