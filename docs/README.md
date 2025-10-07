# PettingLLMs Documentation

This directory contains the documentation for PettingLLMs, built using [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## 🚀 Quick Start

### Building the Documentation

To build the documentation:

```bash
./build_docs.sh
```

Or from project root:

```bash
bash build_docs.sh
```

### Serving the Documentation Locally

To build and serve the documentation with live reload:

```bash
./build_docs.sh serve
```

The documentation will be available at `http://localhost:8000`.

## 📁 Structure

```
.
├── mkdocs.yml              # MkDocs configuration (in root)
├── build_docs.sh           # Build script (in root)
├── docs/                   # Documentation content
│   ├── index.md            # Homepage
│   ├── getting-started/    # Getting started guides
│   ├── core-concepts/      # Core concepts
│   ├── training/           # Training guides
│   ├── evaluation/         # Evaluation guides
│   ├── results/            # Benchmark results
│   ├── api/                # API reference
│   ├── contributing.md     # Contribution guide
│   ├── stylesheets/        # Custom CSS
│   ├── javascripts/        # Custom JS
│   └── requirements.txt    # Documentation dependencies
└── site/                   # Generated static site (after build)
```

## 🔧 Features

### Documentation Features
- **Material Design**: Modern, responsive theme
- **Code highlighting**: Syntax highlighting for multiple languages
- **Navigation**: Automatic navigation generation
- **Search**: Full-text search functionality
- **Mobile-friendly**: Responsive design for all devices
- **Math support**: LaTeX math rendering with MathJax

### Customizations
- Custom color scheme matching PettingLLMs branding
- Enhanced tables for benchmark results
- Code block improvements
- Responsive image handling

## ✍️ Writing Documentation

### Adding New Pages

1. Create a new `.md` file in the appropriate `docs/` subdirectory
2. Add the page to the `nav` section in `mkdocs.yml` (in root)
3. Use Markdown syntax for content

### Code Examples

Use fenced code blocks with language specification:

```python
from pettingllms.trainer import train

# Train agent
train(config, num_iterations=2000)
```

### Math Equations

Use LaTeX syntax:

- Inline: `\( E = mc^2 \)`
- Block: `\[ \frac{-b \pm \sqrt{b^2-4ac}}{2a} \]`

### Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

## 📝 Dependencies

Documentation dependencies are in `docs/requirements.txt`:

```bash
pip install -r docs/requirements.txt
```

Required packages:
- `mkdocs`: Static site generator
- `mkdocs-material`: Material Design theme
- `mkdocstrings[python]`: API documentation from docstrings
- `mkdocs-autorefs`: Cross-references
- `pymdown-extensions`: Enhanced Markdown extensions

## 🚀 Deployment

### Build Static Site

```bash
./build_docs.sh build
```

Output in `site/` directory.

### Deploy to GitHub Pages

```bash
./build_docs.sh deploy
```

### Clean Build Artifacts

```bash
./build_docs.sh clean
```

## 🐛 Troubleshooting

### Import errors when building

Ensure PettingLLMs is installed:
```bash
pip install -e .
```

### Missing dependencies

Install documentation dependencies:
```bash
pip install -r docs/requirements.txt
```

### Build fails

Check that:
- All Markdown files are valid
- `mkdocs.yml` syntax is correct
- All linked files exist

## 📚 Resources

- [MkDocs documentation](https://www.mkdocs.org/)
- [Material theme documentation](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings documentation](https://mkdocstrings.github.io/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## 🤝 Contributing

To contribute to the documentation:

1. Follow the structure outlined above
2. Use clear, concise language
3. Include code examples where appropriate
4. Test your changes locally before submitting
5. See [Contributing Guide](contributing.md) for more details

---

For questions or issues, please open a GitHub issue.

