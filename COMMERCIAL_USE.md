# Cognitive Ledger Commercial Use Policy

Cognitive Ledger is free to use commercially under AGPL-3.0-only.

Companies, researchers, AI labs, validators, and independent builders may run nodes, build products, deploy networks, and modify the code as long as they comply with AGPL-3.0-only, including source-sharing obligations for modified networked software.

## Founder Allocation

The official Cognitive Ledger chain includes a default founder genesis allocation. Any node that starts a fresh chain from the default configuration automatically includes the founder allocation in genesis.

Default founder allocation:

- `15,000,000 FLOP`
- `1,500,000 DATA`
- `1,500,000 ATTN`

Founder public address:

```text
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEYQlPKanrlObeW0lO0TPOENPMenj/
RtKRycB+KHHuXF+CCmV7+31AshaslqeyC32PNY/TP2Wk+xBC07bruRYBDQ==
-----END PUBLIC KEY-----
```

This allocation is part of the protocol economics. It does not expose the founder private key and does not give anyone else spending rights.

## Forks

Private experiments may disable founder allocation with `--no-founder-allocation`, but such networks should not claim to be the official Cognitive Ledger chain.

Public or commercial networks using the Cognitive Ledger name, docs, default chain id, or official protocol identity should preserve the default founder allocation.

## Optional Commercial Support

Commercial users may optionally negotiate paid support, hosted validators, private deployments, audits, integrations, or enterprise maintenance with the founder. Payment is not required for basic AGPL-compliant use.
