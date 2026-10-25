"""Published media keep one accessible element while following the page theme."""
from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT/'docs/_tools/themed_media.py'
    assert path.exists(), 'Published media need a theme-aware HTML integration'
    spec = importlib.util.spec_from_file_location('trackgen_themed_media_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.tags=[]; self.feed(html)
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag,dict(attrs)))


def _assets():
    return {'curve.png': {'kind':'image', 'light':'assets/curve.png',
                           'dark':'_static/media-dark/curve.png'},
            'story.mp4': {'kind':'video', 'light':'assets/story.mp4',
                           'dark':'_static/media-dark/story.mp4'}}


def test_image_uses_one_picture_and_preserves_accessible_content():
    tool = _tool()
    body='<figure id="curve"><a class="image-reference" href="../_images/curve.png"><img src="../_images/curve.png" alt="A &amp; B" width="640" class="plot"></a><figcaption>Caption stays.</figcaption></figure>'
    html=tool.rewrite_body(body,_assets(),lambda path:'../'+path)
    nodes=Elements(html).tags
    assert len([x for x in nodes if x[0]=='picture'])==1
    images=[a for t,a in nodes if t=='img'];assert len(images)==1
    assert images[0]['alt']=='A & B' and images[0]['width']=='640'
    source=next(a for t,a in nodes if t=='source')
    assert source['srcset']=='../_static/media-dark/curve.png'
    assert source['media']=='(prefers-color-scheme: dark)'
    assert 'Caption stays.' in html and 'id="curve"' in html
    assert tool.rewrite_body(html,_assets(),lambda path:'../'+path)==html


def test_video_has_one_player_and_one_original_source_with_paired_metadata():
    tool=_tool()
    body='<video controls preload="none" poster="../_images/curve.png" aria-label="Actual states"><source src="../story.mp4" type="video/mp4"></video>'
    html=tool.rewrite_body(body,_assets(),lambda path:'../'+path)
    nodes=Elements(html).tags
    videos=[a for t,a in nodes if t=='video'];assert len(videos)==1
    assert len([x for x in nodes if x[0]=='source'])==1
    v=videos[0]
    assert v['data-theme-light-src']=='../story.mp4'
    assert v['data-theme-dark-src']=='../_static/media-dark/story.mp4'
    assert v['data-theme-light-poster']=='../_images/curve.png'
    assert v['data-theme-dark-poster']=='../_static/media-dark/curve.png'
    assert v['preload']=='none' and 'controls' in v and v['aria-label']=='Actual states'
    assert tool.rewrite_body(html,_assets(),lambda path:'../'+path)==html


def test_missing_published_variant_is_an_error_but_chrome_and_remote_media_stay():
    tool=_tool()
    with pytest.raises(ValueError,match='new.png'):
        tool.rewrite_body('<img src="../_images/new.png" alt="new">',_assets(),str)
    body='<img src="../_static/minus.png"><img src="https://example.org/picture.png">'
    assert tool.rewrite_body(body,_assets(),str)==body


def test_registry_checks_both_real_variant_files_and_optional_hashes(tmp_path):
    tool=_tool();assets=_assets()
    for row in assets.values():
        for key in ('light','dark'):
            p=tmp_path/row[key];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'actual file')
    registry=tmp_path/'registry.json';registry.write_text(json.dumps({'format':1,'assets':assets}))
    assert tool.load_registry(tmp_path,registry)==assets
    assets['curve.png']['dark_sha256']='incorrect'
    registry.write_text(json.dumps({'format':1,'assets':assets}))
    with pytest.raises(ValueError,match='Changed dark media hash'):
        tool.load_registry(tmp_path,registry)
    del assets['curve.png']['dark_sha256']
    registry.write_text(json.dumps({'format':1,'assets':assets}))
    (tmp_path/assets['curve.png']['dark']).unlink()
    with pytest.raises(ValueError,match='curve.png'):
        tool.load_registry(tmp_path,registry)


def test_bootstrap_is_explicit_and_default_registry_remains_required(tmp_path):
    tool=_tool();body='<img src="../_images/not-yet-paired.png">'
    assert tool.load_registry(tmp_path,'') is None
    assert tool.rewrite_body(body,None,str)==body
    with pytest.raises(FileNotFoundError):
        tool.load_registry(tmp_path,'_static/media-dark/media-manifest.json')


def test_download_link_uses_one_anchor_with_paired_theme_urls():
    tool=_tool()
    body='<a class="reference download internal" download="" href="../_downloads/abc123/story.mp4"><span>Download MP4</span></a>'
    html=tool.rewrite_body(body,_assets(),lambda path:'../'+path)
    anchors=[a for t,a in Elements(html).tags if t=='a'];assert len(anchors)==1
    assert anchors[0]['href']=='../_downloads/abc123/story.mp4'
    assert anchors[0]['data-theme-light-href']=='../_downloads/abc123/story.mp4'
    assert anchors[0]['data-theme-dark-href']=='../_static/media-dark/story.mp4'
    assert 'download' in anchors[0] and '<span>Download MP4</span>' in html
    assert tool.rewrite_body(html,_assets(),lambda path:'../'+path)==html
    external='<a href="https://example.org/story.mp4">External</a>'
    assert tool.rewrite_body(external,_assets(),str)==external


def test_inventory_final_mode_hashes_both_variants_and_includes_download_only(tmp_path):
    tool=_tool();source=tmp_path/'docs';html=tmp_path/'html';html.mkdir()
    (source/'assets').mkdir(parents=True)
    (source/'_static/media-dark').mkdir(parents=True)
    (source/'assets/curve.png').write_bytes(b'light pixels')
    (source/'assets/story.mp4').write_bytes(b'light frames')
    (source/'_static/media-dark/curve.png').write_bytes(b'dark pixels')
    (source/'_static/media-dark/story.mp4').write_bytes(b'dark frames')
    (source/'assets/unpublished.png').write_bytes(b'not referenced')
    (html/'index.html').write_text('<img src="_images/curve.png"><a href="_downloads/hash/story.mp4">MP4</a><img src="_static/minus.png">')
    result=tool.inventory(html,source,require_pairs=True)
    assert set(result['assets'])=={'curve.png','story.mp4'}
    import hashlib
    assert result['assets']['story.mp4']['dark_sha256']==hashlib.sha256(b'dark frames').hexdigest()
    (source/'_static/media-dark/story.mp4').unlink()
    with pytest.raises(ValueError,match='story.mp4'):
        tool.inventory(html,source,require_pairs=True)
    assert 'dark_sha256' not in tool.inventory(html,source)['assets']['story.mp4']
