# Toxiproxy profiles

All toxics in this directory operate on the `downstream` response stream.

The `flaky` profile's `0.05` reset and `0.02` timeout toxicity values are Toxiproxy
connection-level activation probabilities. They are not guaranteed HTTP request failure rates.
HTTP keep-alive, connection reuse, response chunking, and interactions between toxics can change
the empirical request-level results.

Use independent TCP connections when sampling the profile, record the observed outcomes, and treat
the results as measurements rather than acceptance thresholds.
