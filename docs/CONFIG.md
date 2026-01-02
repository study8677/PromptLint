# Suite Configuration

`promptlint` runs from a single YAML suite file. The file contains providers, a model ladder, sampling configs, and prompts.

## Top-level fields
- `name`: suite name
- `providers`: provider API configuration list
- `ladder`: model ladder definition
- `sampling`: list of sampling configs
- `prompts`: prompt definitions and constraints

## Providers
```yaml
providers:
  - name: "openai"
    kind: "openai_compatible"
    api_base: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    timeout_s: 60
    max_retries: 4
    price_per_1k_prompt: 0.0
    price_per_1k_completion: 0.0
```

Fields:
- `name`: provider identifier used by models
- `kind`: provider implementation key (default `openai_compatible`)
- `api_base`: base URL for API requests
- `api_key_env`: environment variable name with API key
- `timeout_s`: request timeout
- `max_retries`: retry attempts for transient failures
- `price_per_1k_prompt`: optional pricing for prompt tokens (USD)
- `price_per_1k_completion`: optional pricing for completion tokens (USD)
- `metadata`: optional dict for extra headers or settings (`headers`, `params` are supported)

## Model ladder
```yaml
ladder:
  name: "default"
  models:
    - provider: "openai"
      name: "gpt-5"
      tier: 5
```

Fields:
- `provider`: must match a `providers[].name`
- `name`: model name passed to the API
- `tier`: larger number = larger model
- `context_window`: optional context length
- `metadata`: optional per-model overrides (e.g. `price_per_1k_prompt`)

## Sampling
```yaml
sampling:
  - temperature: 0.0
    top_p: 1.0
    max_tokens: 256
```

## Run settings
```yaml
run:
  concurrency: 8
  cache_path: ".promptlint/cache.sqlite"
  use_cache: true
  embeddings:
    provider: "openai"
    model: "text-embedding-3-small"
    batch_size: 16
```

Fields:
- `concurrency`: number of concurrent API calls
- `cache_path`: SQLite cache file path
- `use_cache`: enable/disable caching
- `embeddings`: optional embedding model config for semantic similarity

Embedding requests use the provider listed in `run.embeddings.provider`. The provider must be
configured under `providers`. You can override the embeddings endpoint via provider metadata:
```yaml
providers:
  - name: "openai"
    metadata:
      embedding_endpoint: "embeddings"
```

## Prompts and constraints
```yaml
prompts:
  - id: "summarize-bullets"
    text: |
      Summarize the text in 4 bullets.
    metadata:
      expected_format: "bullets"
    constraints:
      - name: "format-bullets"
        description: "Output exactly 4 bullets"
        kind: "format"
        weight: 1.0
        rules:
          type: "count"
          pattern: "^\\s*[-*]\\s+"
          exact: 4
```

Supported `expected_format` values: `bullets`, `numbered_list`, `json`, `code`, `table`.

Constraint rule types:
- `count`: counts regex matches (supports `exact`, `min`, `max`)
- `all_lines_match`: every non-empty line matches regex
- `regex`: regex must match at least once
- `json`: must be valid JSON (`expect: object|array`)
- `length`: length check (`unit: chars|words|lines`)
- `contains` / `not_contains`: substring checks
