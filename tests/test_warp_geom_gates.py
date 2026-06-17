import math, pytest, torch
pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from track_gen import geometry
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

def _circle(n=256,r=2.0,dev="cpu"):
    t=torch.linspace(0,2*math.pi,n+1,device=dev)[:-1]
    return torch.stack([r*torch.cos(t),r*torch.sin(t)],-1).unsqueeze(0)
def _fig8(n=256,s=1.0,dev="cpu"):
    t=torch.linspace(0,2*math.pi,n+1,device=dev)[:-1]
    return torch.stack([s*torch.sin(t),s*torch.sin(t)*torch.cos(t)],-1).unsqueeze(0)

@pytest.mark.parametrize("dev", DEVS)
def test_self_intersections_matches(dev):
    poly=torch.cat([_circle(64,1.0,dev), _fig8(64,1.0,dev)],0)
    got=wpl.self_intersections(poly); ref=geometry.self_intersections(poly)
    assert torch.equal(got.cpu(), ref.cpu())

@pytest.mark.parametrize("dev", DEVS)
def test_thickness_matches(dev):
    torch.manual_seed(0)
    c=(torch.randn(6,256,2,device=dev)*0.7)
    band=torch.randint(2,10,(6,),device=dev)
    assert torch.allclose(wpl.thickness(c,band), geometry.thickness(c,band), atol=1e-4)

@pytest.mark.parametrize("dev", DEVS)
def test_thickness_circle(dev):
    c=_circle(400,2.0,dev); band=torch.tensor([400//2-2],device=dev)
    assert torch.allclose(wpl.thickness(c,band), torch.tensor([2.0],device=dev), atol=2e-2)
