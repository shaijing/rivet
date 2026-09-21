# Crate architecture diagrams

The architecture diagrams are maintained one per workspace crate:

- `rivet-core.lini` — tensor storage, layouts, dtypes, and operations.
- `rivet-data.lini` — datasets, storage, sampling, and runtime primitives.
- `rivet-vision.lini` — image datasets, transforms, batching, and loaders.
- `rivet-python.lini` — PyO3 bindings and Python interoperability.

Regenerate the SVG outputs in `docs/static/arch-svg` with:

```bash
./docs/arch-lini/generate.sh
```
