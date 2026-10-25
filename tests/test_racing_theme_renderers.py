"""Real renderer save/frame boundaries honor the selected presentation palette."""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import pytest

from viz.racing_theme import get_palette, plotting_theme, geometry_digest


@pytest.mark.parametrize('producer', ['data', 'course', 'xpbd-frame'])
def test_renderer_dark_canvas_uses_real_theme_without_changing_input(producer,tmp_path):
    p=get_palette('dark');data=None;before=None
    with plotting_theme('dark'):
        if producer=='data':
            from viz.render_data_guide import render_layout
            path=tmp_path/'layout.png';render_layout(path)
        elif producer=='course':
            from viz.render_course_guide import _save
            fig,ax=plt.subplots();ax.plot([0,1],[1,0],color='#2563eb')
            _save(fig,tmp_path,'course','Actual course');path=tmp_path/'course.png'
        else:
            from viz.render_relaxation_video import _frame
            t=np.arange(16)*2*np.pi/16
            center=np.c_[np.cos(t),np.sin(t)][None,:,:]
            data=dict(center=center,outer=1.1*center,inner=.9*center,
                      count=np.array([16]),valid=np.array([1]),fail_mask=np.array([0]),
                      phase='sweep',iteration=3)
            before=geometry_digest(data);path=tmp_path/'frame.png'
            _frame(path,data,[0],[(np.array([-1.5,-1.5]),np.array([1.5,1.5]))],
                   {'seed_start':0},False)
    corner=np.asarray(Image.open(path).convert('RGB'))[0,0]
    np.testing.assert_array_equal(corner,np.rint(np.array(to_rgb(p.background))*255))
    if data is not None:assert geometry_digest(data)==before


def test_paired_data_cli_reuses_capture_and_records_identical_geometry(tmp_path):
    import json
    import os
    import subprocess
    import sys
    done=subprocess.run([sys.executable,'-m','viz.render_data_guide','--out-dir',str(tmp_path),
                         '--theme','both'],capture_output=True,text=True,env=os.environ.copy())
    assert done.returncode==0,done.stderr
    manifests=[json.loads((tmp_path/theme/'data-manifest.json').read_text())
               for theme in ('light','dark')]
    assert [m['theme'] for m in manifests]==['light','dark']
    assert manifests[0]['render_geometry']==manifests[1]['render_geometry']
    assert len(manifests[0]['render_geometry'])==4
    assert manifests[0]['import']==manifests[1]['import']
    assert manifests[0]['gate_course']==manifests[1]['gate_course']
