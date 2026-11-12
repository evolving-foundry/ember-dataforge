# DataForge -> Gym terminal profile example

This example freezes a terminal-profile environment entirely from DataForge
semantic inputs, without Curator documents:

```text
terminal DomainAdapter
  -> EnvironmentModel
  -> registered terminal ProfileCompiler
  -> PackageBuilder + gates + Publisher
  -> Gym load_package_zip + TerminalProfile + ReplayEngine
```

Run it from the repository root:

```bash
PYTHONPATH=/path/to/DataForge \
DATA_FORGE_GYM=/path/to/AscendAgent-Gym \
  python examples/envsynth_terminal_e2e/run_example.py \
    --work-dir /tmp/dd-terminal-e2e \
    --gym-root /path/to/AscendAgent-Gym
```

Outputs:

- `domain.json` / `draft.json`: semantic inputs and rich DD tasks;
- `release/package/`: the immutable Gym-native package directory;
- `release/gym-env-package-v1.zip`: the exact package Gym loads;
- `example_summary.json`: fingerprints, gate verdict, and Gym runtime result.

The package contains `env_spec.json`, `fs/snapshot.jsonl`,
`replay/replay.jsonl` and the strict six-field `tasks.jsonl`. It intentionally
does not contain office files such as `state/seed.jsonl` or `rules.yaml`.
