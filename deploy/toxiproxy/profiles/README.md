# Toxiproxy profiles

All toxics in this directory operate on the `downstream` response stream.

The `flaky` profile's `0.05` reset and `0.02` timeout toxicity values are Toxiproxy
connection-level activation probabilities. They are not guaranteed HTTP request failure rates.
HTTP keep-alive, connection reuse, response chunking, and interactions between toxics can change
the empirical request-level results.

Use independent TCP connections when sampling the profile, record the observed outcomes, and treat
the results as measurements rather than acceptance thresholds.

The opt-in diagnostic compares a clean baseline, `typical` latency, and `flaky` using identical
independent connections. Its 12-second read timeout deliberately exceeds the configured 10-second
timeout toxic: a long injected stall is recorded in request latency instead of being automatically
reported as an undifferentiated client timeout. HTTPX connect, read, write, and pool timeouts and
transport errors are reported separately. Run it with:

```bash
PA_RUN_INTEGRATION=1 PA_RUN_FLAKY_EXPERIMENT=1 \
  uv run pytest tests/integration/test_toxiproxy_profiles.py -k controlled -s
```

`PA_FLAKY_DIAGNOSTIC_ATTEMPTS` and `PA_FLAKY_DIAGNOSTIC_CONCURRENCY` may be set for bounded
comparisons. Each condition verifies the intended proxy state before measuring and verifies an
empty state afterward. Reported RTT begins immediately before the request, so it includes TCP
connection establishment and the complete response but excludes client construction and shutdown.
