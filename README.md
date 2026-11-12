# Predictive Agent

Event-driven, asenkron bir LLM ajan orkestrasyon altyapısı. Hızlı bir router kullanıcının
niyetini tahmin eder ve ana ajan çalışmaya başlamadan **önce** gereken dış verileri arka planda
çekip in-memory cache'e yazar. Ana ajan bu veriyi ağ gecikmesi yaşamadan okur
(predictive tool prefetching).

> **Durum:** M0 tamamlandı; M1 geliştirmesi devam ediyor. Yol haritası için [ROADMAP.md](ROADMAP.md),
> mimari kararlar için [docs/adr/](docs/adr/).

## Gereksinimler

- WSL2 (Ubuntu 24.04), `networkingMode=mirrored`
- [uv](https://docs.astral.sh/uv/), Docker, [Ollama](https://ollama.com) (Windows'ta)

## Kurulum

```bash
cp .env.example .env
uv sync
uv run pre-commit install
```

## Ortam kontrolü ve LLM baseline

```bash
uv run pa-doctor --runs 5
uv run pa-doctor --runs 5 --cached --json doctor-cached.json
```

Ollama'ya erişimi, modelin kurulu olduğunu kontrol eder; modeli ısıtıp sıcak üretim hızını ölçer.
Varsayılan benchmark her çalıştırmada benzersiz bir prompt ile cache'siz prefill'i ölçer;
`--cached` aynı promptu tekrarlayarak cache'li koşulu ayrı raporlar.

## Geliştirme

```bash
uv run pytest            # testler
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pyright           # tip kontrolü
```

Her değişiklik: feature branch → PR → CI yeşil → merge. Commit mesajları
[Conventional Commits](https://www.conventionalcommits.org/) formatında.
