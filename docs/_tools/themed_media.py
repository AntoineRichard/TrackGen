"""Pair published images/videos with true dark renders without duplicate players.

Sphinx extension: add this module to extensions and put a format-1 registry at
``_static/media-dark/media-manifest.json``. Rendering stays an optional task.
The command-line inventory reads an existing HTML build; it never generates art.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

_MEDIA = {'.png', '.svg', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm'}
_VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}


def _basename(uri):
    return Path(unquote(urlsplit(uri).path)).name


def _local(uri):
    return bool(uri) and not urlsplit(uri).scheme and not uri.startswith('//')


def _tag(name, attrs):
    fields = [key if value is None else f'{key}="{escape(str(value), quote=True)}"'
              for key, value in attrs.items()]
    return '<' + name + (' ' + ' '.join(fields) if fields else '') + '>'


class _Videos(HTMLParser):
    def __init__(self, body):
        super().__init__(convert_charrefs=False)
        self.records = []
        self.current = None
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'video':
            self.current = dict(src=attrs.get('src'), poster=attrs.get('poster'))
            self.records.append(self.current)
        elif tag == 'source' and self.current is not None and not self.current['src']:
            self.current['src'] = attrs.get('src')

    def handle_endtag(self, tag):
        if tag == 'video':
            self.current = None


class _Rewrite(HTMLParser):
    def __init__(self, body, assets, pathto):
        super().__init__(convert_charrefs=False)
        self.assets, self.pathto = assets, pathto
        self.videos = iter(_Videos(body).records)
        self.parts, self.stack = [], []
        self.feed(body)
        self.close()

    def _asset(self, uri, *, required=False):
        if not _local(uri):
            return None
        row = self.assets.get(_basename(uri))
        if row is None and required:
            raise ValueError(f'Published media has no paired theme variant: {uri}')
        return row

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        original = self.get_starttag_text()
        if tag == 'a':
            uri = values.get('href', '')
            row = self._asset(uri)
            if row is not None:
                values['data-theme-light-href'] = uri
                values['data-theme-dark-href'] = self.pathto(row['dark'])
                original = _tag(tag, values)
            self.parts.append(original)
        elif tag == 'img' and 'picture' not in self.stack:
            uri = values.get('src', '')
            row = self._asset(uri, required='_images/' in uri)
            if row is not None:
                dark = self.pathto(row['dark'])
                self.parts.append(_tag('picture', {'class':'themed-media-picture',
                    'data-theme-light-src':uri, 'data-theme-dark-src':dark}))
                self.parts.append(_tag('source', {'data-theme-dark':'', 'srcset':dark,
                    'media':'(prefers-color-scheme: dark)'}))
                self.parts.extend((original, '</picture>'))
            else:
                self.parts.append(original)
        elif tag == 'video':
            record = next(self.videos)
            uri = record['src'] or ''
            row = self._asset(uri, required=_local(uri))
            if row is not None:
                classes = values.get('class', '').split()
                if 'themed-media-video' not in classes:
                    classes.append('themed-media-video')
                values['class'] = ' '.join(classes)
                values['data-theme-light-src'] = uri
                values['data-theme-dark-src'] = self.pathto(row['dark'])
                if record['poster']:
                    poster = self._asset(record['poster'], required=_local(record['poster']))
                    if poster is not None:
                        values['data-theme-light-poster'] = record['poster']
                        values['data-theme-dark-poster'] = self.pathto(poster['dark'])
                original = _tag(tag, values)
            self.parts.append(original)
        else:
            self.parts.append(original)
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')
        if tag in self.stack:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]

    def handle_data(self, data): self.parts.append(data)
    def handle_entityref(self, name): self.parts.append('&' + name + ';')
    def handle_charref(self, name): self.parts.append('&#' + name + ';')
    def handle_comment(self, data): self.parts.append('<!--' + data + '-->')
    def handle_decl(self, decl): self.parts.append('<!' + decl + '>')


def rewrite_body(body, assets, pathto):
    """Keep one image/player and its accessible content; only add theme choices."""
    if assets is None:
        return body  # explicit bootstrap build before paired media exist
    return ''.join(_Rewrite(body, assets, pathto).parts)


def load_registry(source_dir, manifest):
    """Validate the complete checked-in pair registry before writing an HTML page."""
    if manifest is None or manifest is False or manifest == '' or manifest == {}:
        return None
    root = Path(source_dir).resolve()
    manifest = Path(manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    data = json.loads(manifest.read_text())
    if data.get('format') != 1 or not isinstance(data.get('assets'), dict):
        raise ValueError(f'Invalid themed media registry: {manifest}')
    for name, row in data['assets'].items():
        if Path(name).name != name or row.get('kind') not in ('image', 'video'):
            raise ValueError(f'Invalid themed media entry: {name}')
        for key in ('light', 'dark'):
            uri = row.get(key, '')
            path = (root / uri).resolve()
            if (not _local(uri) or not path.is_relative_to(root) or not path.is_file()):
                raise ValueError(f'Missing {key} media variant for {name}: {uri}')
            expected = row.get(key + '_sha256')
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Changed {key} media hash for {name}: {uri}')
    return data['assets']


def _load(app):
    if app.builder.format != 'html':
        return
    app._trackgen_themed_media = load_registry(app.srcdir, app.config.themed_media_registry)


def _html_context(app, pagename, templatename, context, doctree):
    if context.get('body'):
        context['body'] = rewrite_body(context['body'], app._trackgen_themed_media,
                                       lambda uri: context['pathto'](uri, 1))


def setup(app):
    app.add_config_value('themed_media_registry', '_static/media-dark/media-manifest.json', 'html',
                         types=[str, type(None), bool, dict])
    app.connect('builder-inited', _load)
    app.connect('html-page-context', _html_context)
    app.add_js_file('themed-media.js', defer='defer')
    app.add_css_file('themed-media.css')
    return {'version':'1', 'parallel_read_safe':True, 'parallel_write_safe':True}


class _References(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.uris=[]; self.feed(html)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ('img', 'video', 'source'):
            self.uris.extend(a[k] for k in ('src', 'poster') if k in a and _local(a[k]))
        elif tag == 'a' and _local(a.get('href', '')):
            if Path(urlsplit(a['href']).path).suffix.lower() in _MEDIA:
                self.uris.append(a['href'])


def inventory(html_dir, source_dir, *, require_pairs=False):
    """Map actual built reader media to source files, excluding internal records."""
    source = Path(source_dir).resolve()
    available = {}
    excluded = {'library-readiness', 'superpowers', 'media-dark', '_build', '_ext', '_tools'}
    for path in source.rglob('*'):
        if (path.is_file() and path.suffix.lower() in _MEDIA
                and not set(path.relative_to(source).parts) & excluded):
            available.setdefault(path.name, []).append(path)
    html_root = Path(html_dir).resolve()
    used = {}
    for page in html_root.rglob('*.html'):
        for uri in _References(page.read_text()).uris:
            used.setdefault(_basename(uri), []).append((page, uri))
    assets = {}
    for name in sorted(used):
        candidates = available.get(name, [])
        if not candidates:
            continue  # Sphinx chrome, such as its generated minus icon
        if len(candidates) == 1:
            path = candidates[0]
        else:
            hashes = {candidate: hashlib.sha256(candidate.read_bytes()).hexdigest()
                      for candidate in candidates}
            selected = set()
            for page, uri in used[name]:
                local = unquote(urlsplit(uri).path)
                built = ((html_root / local.lstrip('/')) if local.startswith('/')
                         else (page.parent / local)).resolve()
                if not built.is_relative_to(html_root):
                    raise ValueError(f'Published media is outside HTML root: {uri}')
                if not built.is_file():
                    raise ValueError(f'Missing built media for {name}: {uri}')
                digest = hashlib.sha256(built.read_bytes()).hexdigest()
                matches = [candidate for candidate, value in hashes.items() if value == digest]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous published media basename: {name}: {candidates}')
                selected.add(matches[0])
            if len(selected) != 1:
                raise ValueError(f'Ambiguous published media basename: {name}: {sorted(selected)}')
            path = selected.pop()
        assets[name] = dict(kind='video' if path.suffix in ('.mp4', '.webm') else 'image',
                            light=str(path.relative_to(source)),
                            dark='_static/media-dark/' + name,
                            light_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        dark = source / assets[name]['dark']
        if dark.is_file():
            assets[name]['dark_sha256'] = hashlib.sha256(dark.read_bytes()).hexdigest()
        elif require_pairs:
            raise ValueError(f'Missing dark media variant for {name}: {dark}')
    return {'format':1, 'assets':assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-html', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path('docs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-pairs', action='store_true',
                        help='fail if any published asset lacks its real dark counterpart')
    args = parser.parse_args()
    data = inventory(args.inventory_html, args.source_dir, require_pairs=args.require_pairs)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(f"Inventoried {len(data['assets'])} published media files")


if __name__ == '__main__':
    main()
