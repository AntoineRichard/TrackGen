"""Selective media refreshes must replace only the files' actual provenance."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from viz import render_generator_guide as guide
from viz.racing_theme import plotting_theme, rendered_geometry, save_figure


@pytest.mark.parametrize('mode', ['construction', 'repulsive-animation', 'repulsive-assets'])
def test_selective_refresh_updates_current_sources_hashes_and_extra_posters(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(guide, 'FAMILIES', ('polar', 'repulsive'))
    current_source = dict(git_commit='review-current', theme='dark', source_tree_dirty=False)
    monkeypatch.setattr(guide, '_source_record', lambda: dict(current_source))
    original_check_output = guide.subprocess.check_output
    monkeypatch.setattr(guide.subprocess, 'check_output',
        lambda args, **kwargs: '' if args[:2] == ['git', 'status'] else original_check_output(args, **kwargs))

    def samples(name, *, seeds, **kwargs):
        return [dict(seed=seed, env=i, config={'generator': name}, valid=True,
                     center=np.array([[0., 0.]])) for i, seed in enumerate(seeds)]
    monkeypatch.setattr(guide, 'sample_batch', samples)
    monkeypatch.setattr(guide, '_record', lambda sample: {key:sample[key] for key in ('seed','env','config','valid')})
    untouched = 'generator-hull-examples.png'
    stale = dict(source=dict(git_commit='previous', theme='light'), command='old command')
    manifest = dict(assets={untouched: 123}, asset_provenance={untouched: stale},
                    plotted_geometry_sha256={untouched: 'untouched-hash'}, specials={},
                    families={name: dict(seed_pool=[7], examples=[samples(name,seeds=[7])[0]])
                              for name in guide.FAMILIES},
                    animations={name: dict(sample=samples(name,seeds=[7])[0]) for name in guide.FAMILIES})
    # Serialize metadata, keeping synthetic geometry out of the input manifest.
    for family in manifest['families'].values():
        for sample in family['examples']:
            sample.pop('center')
    for animation in manifest['animations'].values():
        animation['sample'].pop('center')
    for name in ('generator-polar-construction.png','generator-polar-construction.mp4',
                 'generator-polar-construction-poster.png','generator-repulsive-growth.png',
                 'generator-repulsive-growth.mp4','generator-repulsive-examples.png',
                 'generator-repulsive-controls.png','generator-repulsive-numerics.png',
                 'generator-repulsive-controls-obstacles.png','generators-gallery.png'):
        manifest['assets'][name]=1
        manifest['asset_provenance'][name]=stale
        if name.endswith('.png'):
            manifest['plotted_geometry_sha256'][name]='old-plotted-hash'
    if mode == 'construction':
        # Animation selection may differ from the family's default gallery pool.
        manifest['animations']['polar']['sample']['seed'] = 112
    source_manifest=tmp_path/'previous.json';source_manifest.write_text(json.dumps(manifest))
    out=tmp_path/'new';out.mkdir()
    retired='generator-repulsive-retired.png'
    (out/retired).write_bytes(b'unchanged retired figure')
    manifest['assets'][retired]=(out/retired).stat().st_size
    manifest['asset_provenance'][retired]=stale
    manifest['plotted_geometry_sha256'][retired]='retired-hash'
    source_manifest.write_text(json.dumps(manifest))

    def png(name):
        fig,ax=plt.subplots();ax.plot([0.,1.],[1.,0.])
        save_figure(fig,out/name);plt.close(fig)
    def movie(name):
        (out/name).write_bytes(b'new encoded video')
    def construction(target,sample):
        assert sample['seed'] == 112
        png('generator-polar-construction.png');movie('generator-polar-construction.mp4')
        png('generator-polar-construction-poster.png')
        return dict(poster='generator-polar-construction-poster.png')
    def growth(target,sample,**kwargs):
        png('generator-repulsive-growth.png');movie('generator-repulsive-growth.mp4')
        return {}
    monkeypatch.setattr(guide,'render_construction',construction)
    monkeypatch.setattr(guide,'render_growth',growth)
    for method,filename in (
        ('render_examples','generator-repulsive-examples.png'),
        ('render_comparisons','generator-repulsive-controls.png'),
        ('render_repulsive_numerics','generator-repulsive-numerics.png'),
        ('render_repulsive_barriers','generator-repulsive-controls-obstacles.png'),
        ('render_gallery','generators-gallery.png'),
    ):
        def render(*args,_filename=filename,**kwargs):
            png(_filename);return {}
        monkeypatch.setattr(guide,method,render)
    functions={'construction':guide.render_construction_animations,
               'repulsive-animation':guide.render_repulsive_animation,
               'repulsive-assets':guide.render_repulsive_assets}
    with plotting_theme('dark'):
        result=functions[mode](out,source_manifest)
        coordinates=rendered_geometry()
    refreshed=[path for path in out.iterdir() if path.suffix in ('.png','.mp4') and path.name!=retired]
    assert refreshed
    for path in refreshed:
        assert result['assets'][path.name]==path.stat().st_size
        provenance=result['asset_provenance'][path.name]
        assert provenance['source']['git_commit']=='review-current'
        assert provenance['source']['theme']=='dark'
        assert '--theme dark' in provenance['command']
        if path.suffix=='.png':
            assert result['plotted_geometry_sha256'][path.name]==coordinates[path.name]
    animation=result['animations']['polar' if mode=='construction' else 'repulsive']
    assert animation['source']['git_commit']=='review-current'
    assert animation['source']['theme']=='dark'
    assert result['asset_provenance'][retired]==stale
    assert result['plotted_geometry_sha256'][retired]=='retired-hash'
    assert result['asset_provenance'][untouched]==stale
    assert result['assets'][untouched]==123
    assert result['plotted_geometry_sha256'][untouched]=='untouched-hash'
