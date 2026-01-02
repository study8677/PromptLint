# Contributing

Thanks for helping improve PromptLint. Keep changes small, clear, and testable.

## Development setup
```bash
pip install -e .
```

## Running a suite locally
```bash
export OPENAI_API_KEY=your_key
promptlint --suite examples/suite.yaml --report report.md --report-format markdown
```

## Tests
```bash
python -m unittest discover -s tests
```

## Guidelines
- Keep interfaces stable (`providers`, `evaluators`, `aggregators`).
- Avoid heavy dependencies unless necessary.
- Include a small example or update `examples/suite.yaml` if you add new config options.
- Prefer clear docs over clever abstractions.
