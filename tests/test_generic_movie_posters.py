"""Generic movie posters use a real final scene, while detailed diagrams stay intact."""
from contextlib import contextmanager
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from viz import render_generator_guide as guide


@pytest.mark.parametrize('name',['voronoi','checkpoint'])
@pytest.mark.parametrize('theme',['light','dark'])
def test_poster_only_render_keeps_static_diagram_and_native_movie_poster(tmp_path,name,theme,monkeypatch):
    def no_encoder():
        raise AssertionError('Posters must render without an encoder')
    monkeypatch.setattr(guide,'_writer',no_encoder)
    sample=guide.sample_batch(name,seeds=[7])[0]
    before={key:sample[key].tobytes() for key in ('raw','center','inner','outer')}
    record=guide.render_construction(tmp_path,sample,movies=False,theme=theme)
    assert record['poster']==f'generator-{name}-construction-poster.png'
    assert record['output']==f'generator-{name}-construction.png'
    poster=Image.open(tmp_path/record['poster'])
    diagram=Image.open(tmp_path/record['output'])
    assert poster.size==(1440,810)
    assert record['frame_size']==record['poster_frame_size']==list(poster.size)
    assert record['aspect_ratio']=='16:9'
    assert diagram.height*16>diagram.width*9
    assert before=={key:sample[key].tobytes() for key in before}


@pytest.mark.parametrize('name',['voronoi','checkpoint'])
@pytest.mark.parametrize('theme',['light','dark'])
def test_movie_poster_matches_the_actual_last_encoded_scene(tmp_path,name,theme,monkeypatch):
    class Writer:
        frames=0
        @contextmanager
        def saving(self,fig,path,dpi):
            self.figure,self.dpi=fig,dpi
            yield
            buf=BytesIO();fig.savefig(buf,dpi=dpi)
            self.last_frame=np.array(Image.open(buf).convert('RGBA'))
        def grab_frame(self,**kwargs):
            self.frames+=1
    writer=Writer();monkeypatch.setattr(guide,'_writer',lambda:writer)
    sample=guide.sample_batch(name,seeds=[7])[0]
    record=guide.render_construction(tmp_path,sample,movies=True,theme=theme)
    assert record['frames']==writer.frames==96
    assert record['poster']==f'generator-{name}-construction-poster.png'
    actual=np.asarray(Image.open(tmp_path/record['poster']).convert('RGBA'))
    np.testing.assert_array_equal(actual,writer.last_frame)
