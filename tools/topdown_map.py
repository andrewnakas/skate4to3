"""Top-down render of the map, each triangle filled with its texture's mean colour.

One viewpoint in game shows a few hundred square metres of a 1.35 million m2 island, and
booting to see another spot costs a rebuild - the pack's spawn comes from the startline
marker, so neither `--index` nor d2s3's `--spawn` moves it. This covers the whole map at
once: a region wearing one obviously wrong flat colour, or the magenta of a material with no
texture at all, is visible immediately.

    python tools/topdown_map.py out.png [glb_dir]
"""
import glob, io, json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
_argv = sys.argv[:]          # verify_build reads sys.argv AT IMPORT to pick its GLB dir
sys.argv = [_argv[0]]
import verify_build as V
from texcache import WORK
sys.argv = _argv

GLB = sys.argv[2] if len(sys.argv) > 2 else V.GLB
W = H = 1400

# material name -> mean RGB of its embedded texture
def glb_material_colours(path):
    b=open(path,'rb').read(); off=12; js=bn=None
    while off<len(b):
        ln,ty=struct.unpack_from('<II',b,off); off+=8
        if ty==0x4E4F534A: js=json.loads(b[off:off+ln])
        elif ty==0x004E4942: bn=b[off:off+ln]
        off+=ln
    out={}
    for m in js['materials']:
        t=m.get('pbrMetallicRoughness',{}).get('baseColorTexture')
        if not t: continue
        src=js['textures'][t['index']]['source']; im=js['images'][src]
        bv=js['bufferViews'][im['bufferView']]
        blob=bn[bv.get('byteOffset',0):bv.get('byteOffset',0)+bv['byteLength']]
        try:
            with Image.open(io.BytesIO(blob)) as pim:
                a=np.asarray(pim.convert('RGB'),dtype=np.float32)
            out[m['name']]=a.reshape(-1,3).mean(0)
        except Exception: pass
    return out

cols={}
for f in sorted(glob.glob(os.path.join(GLB,'block_*.glb'))):
    cols.update(glb_material_colours(f))
print(f"material colours: {len(cols)}")

tris=[]
for _f, mn, P, _UV, T in V.prims():
    if 'Collision' in mn: continue
    c = cols.get(mn)
    if c is None: c = np.array([255,0,255],dtype=np.float32)
    tris.append((P, T, c, mn))
xs=np.concatenate([p[:,0] for p,_,_,_ in tris]); zs=np.concatenate([p[:,2] for p,_,_,_ in tris])
x0,x1,z0,z1=xs.min(),xs.max(),zs.min(),zs.max()
print(f"extent x {x0:.0f}..{x1:.0f}  z {z0:.0f}..{z1:.0f}")
img=np.zeros((H,W,3),np.float32); zbuf=np.full((H,W),-1e9,np.float32)
sx=(W-1)/(x1-x0); sz=(H-1)/(z1-z0)
rng=np.random.default_rng(0)
for P,T,c,mn in tris:
    px=((P[:,0]-x0)*sx); py=(H-1)-((P[:,2]-z0)*sz)
    a,b,cc=T[:,0],T[:,1],T[:,2]
    ax,ay=px[a],py[a]; bx,by=px[b],py[b]; cx_,cy_=px[cc],py[cc]
    # screen area -> how many samples this triangle needs to fill its footprint
    area=0.5*np.abs((bx-ax)*(cy_-ay)-(cx_-ax)*(by-ay))
    k=np.clip(np.ceil(area*1.6).astype(np.int64),1,4000)
    idx=np.repeat(np.arange(len(T)),k)
    u=rng.random(len(idx)); v=rng.random(len(idx))
    fl=u+v>1.0; u[fl]=1.0-u[fl]; v[fl]=1.0-v[fl]
    w=1.0-u-v
    X=(ax[idx]*w+bx[idx]*u+cx_[idx]*v).astype(np.int32)
    Y=(ay[idx]*w+by[idx]*u+cy_[idx]*v).astype(np.int32)
    Yv=(P[a,1][idx]*w+P[b,1][idx]*u+P[cc,1][idx]*v).astype(np.float32)
    ok=(X>=0)&(X<W)&(Y>=0)&(Y<H)
    X,Y,Yv=X[ok],Y[ok],Yv[ok]
    o=np.argsort(Yv)          # highest last, so it wins the z-test
    X,Y,Yv=X[o],Y[o],Yv[o]
    sel=Yv>zbuf[Y,X]
    zbuf[Y[sel],X[sel]]=Yv[sel]
    img[Y[sel],X[sel]]=c
out=np.clip(img,0,255).astype(np.uint8)
Image.fromarray(out).save(sys.argv[1] if len(sys.argv)>1 else 'topdown.png')
print("wrote", sys.argv[1] if len(sys.argv)>1 else 'topdown.png')
