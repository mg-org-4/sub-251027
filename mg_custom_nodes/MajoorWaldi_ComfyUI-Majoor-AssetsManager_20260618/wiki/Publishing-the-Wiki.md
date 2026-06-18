# Publishing This Wiki

GitHub wikis are stored in a separate repository ending with `.wiki.git`.

This folder provides wiki-ready Markdown pages you can publish there.

## Typical Publish Flow

```bash
git clone https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager.wiki.git
cd ComfyUI-Majoor-AssetsManager.wiki
```

Then copy the contents of this repository's `wiki/` folder into that cloned wiki repository and commit the pages.

## Exact Local Sequence

If you keep the main repository and the wiki clone side by side locally, a practical sequence is:

```powershell
# From the main repository root
git clone https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager.wiki.git .wiki-publish
Copy-Item -Path .\wiki\*.md -Destination .\.wiki-publish\ -Force

# Review pending changes
git -C .wiki-publish status --short

# Commit and publish
git -C .wiki-publish add .
git -C .wiki-publish commit -m "Update wiki from repository docs"
git -C .wiki-publish push origin master
```

If the wiki repository already exists locally, skip the clone step and start from the copy command.

## Minimum Files To Publish

- `Home.md`
- `_Sidebar.md`

The rest of the pages are optional but recommended.

## Suggested Page Set

- Home
- Installation and Setup
- Using the Assets Manager
- Search, Ratings, Tags, and Collections
- Viewer and MFV
- AI Features
- Privacy and Offline
- Configuration and Security
- Plugin System
- Maintenance and Testing
- Development and Architecture

## Practical Recommendation

When publishing updates, prioritize syncing these pages whenever the underlying docs change:

- Home
- AI Features
- Privacy and Offline
- Configuration and Security
- Plugin System
- Maintenance and Testing

## Maintenance Approach

Treat the repository docs as the canonical, detailed reference and use the wiki for:
- project onboarding
- navigation
- short conceptual summaries
- discoverability for GitHub visitors

That keeps the wiki stable while the repo docs continue to evolve.

## Source Of Truth Strategy

Use the wiki for concise navigation and onboarding. Keep detailed procedures, command references, and long-form technical design in the repository docs.
