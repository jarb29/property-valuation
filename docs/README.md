# MkDocs Documentation for Property Valuation ML System

This directory contains the documentation for the Property Valuation ML System, configured for MkDocs.

## About MkDocs

[MkDocs](https://www.mkdocs.org/) is a fast, simple, and beautiful static site generator designed for building project documentation. The documentation source files are written in Markdown and configured with a single YAML configuration file.

We're using the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme for enhanced features and a modern look.

## Structure

- `mkdocs.yml`: MkDocs configuration file (in the project root)
- `docs/`: Documentation source files
  - `index.md`: Main landing page
  - `getting-started.md`: Quick start guide
  - `installation-guide.md`: Detailed installation instructions
  - `user-manual.md`: Comprehensive user guide
  - `api-documentation.md`: API reference
  - `github-pages-deployment.md`: Guide for deploying to GitHub Pages
  - `Challenge.md`: Original project requirements
  - `assets/css/extra.css`: Custom CSS styles

## Working with MkDocs

### Installation

To work with MkDocs, you need to install it first:

```bash
# Install MkDocs and the Material theme
pip install mkdocs mkdocs-material

# Install additional plugins
pip install mkdocs-minify-plugin
```

### Local Development

To preview the documentation locally:

```bash
# Start the MkDocs development server
mkdocs serve
```

This will start a local server at `http://127.0.0.1:8000/`. The server automatically reloads when you make changes to the documentation.

### Building the Documentation

To build the static site:

```bash
# Build the documentation
mkdocs build
```

This will create a `site` directory with the built HTML files.

### Deploying to GitHub Pages

To deploy the documentation to GitHub Pages:

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```

See the [GitHub Pages Deployment Guide](github-pages-deployment.md) for more details.

## Maintaining Documentation

### Adding New Pages

1. Create a new Markdown file in the `docs` directory
2. Add your content using Markdown
3. Update the navigation in `mkdocs.yml` to include your new page

### Updating Existing Pages

1. Edit the Markdown file directly
2. Save your changes
3. The local server will automatically reload to show your changes

### Styling

The documentation uses a custom CSS file located at `docs/assets/css/extra.css`. To modify the styling:

1. Edit the CSS file
2. Save your changes
3. The local server will automatically reload to show your changes

## MkDocs Features

MkDocs with the Material theme provides many useful features:

- **Search**: Built-in search functionality
- **Navigation**: Automatically generated navigation
- **Responsive Design**: Works well on mobile devices
- **Code Highlighting**: Syntax highlighting for code blocks
- **Admonitions**: Special callout blocks for notes, warnings, etc.
- **Tables of Contents**: Automatically generated for each page
- **Customization**: Extensive theme customization options

## Advantages Over Jekyll

MkDocs offers several advantages over Jekyll for technical documentation:

1. **Simplicity**: Easier to set up and maintain
2. **Speed**: Faster build times
3. **Focus**: Specifically designed for project documentation
4. **Material Theme**: Modern, responsive design with many features
5. **Python-based**: Integrates well with Python projects
6. **Navigation**: Better handling of navigation structure
7. **Search**: Superior built-in search functionality

## Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs Documentation](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)