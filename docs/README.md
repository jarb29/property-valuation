# GitHub Pages Documentation

This directory contains the documentation for the Property Valuation ML System, configured for GitHub Pages.

## Structure

- `index.html`: Main landing page
- `_config.yml`: GitHub Pages configuration
- `_layouts/default.html`: Default layout template for all pages
- `assets/css/style.css`: Custom CSS styles
- Markdown documentation files:
  - `getting-started.md`: Quick start guide
  - `installation-guide.md`: Detailed installation instructions
  - `user-manual.md`: Comprehensive user guide
  - `api-documentation.md`: API reference
  - `github-pages-deployment.md`: Guide for deploying to GitHub Pages
  - `Challenge.md`: Original project requirements

## Maintaining Documentation

### Adding New Pages

1. Create a new Markdown file in the `docs` directory
2. Add front matter at the beginning of the file:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   ---
   ```
3. Add your content using Markdown
4. Update the navigation in `_layouts/default.html` to include your new page

### Updating Existing Pages

1. Edit the Markdown file directly
2. Commit and push your changes
3. GitHub Pages will automatically rebuild the site

### Styling

The documentation uses a custom CSS file located at `assets/css/style.css`. To modify the styling:

1. Edit the CSS file
2. Commit and push your changes
3. GitHub Pages will apply the new styles

## Local Development

To test the documentation locally:

1. Install [Jekyll](https://jekyllrb.com/docs/installation/)
2. Navigate to the `docs` directory
3. Run `bundle install`
4. Run `bundle exec jekyll serve`
5. Open `http://localhost:4000` in your browser

## GitHub Pages Configuration

The documentation is configured to use the `jekyll-theme-minimal` theme with custom styling. The configuration is in `_config.yml`.

Key settings:
- Theme: `jekyll-theme-minimal`
- Markdown processor: GitHub Flavored Markdown (GFM)
- Plugins: SEO tags and sitemap

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. The site is available at:

```
https://username.github.io/property-valuation/
```

Replace `username` with your GitHub username or organization name.
