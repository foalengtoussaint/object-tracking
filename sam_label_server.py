"""Browser-based SAM click-annotator for building YOLO-seg seed labels.

Run:
    python sam_label_server.py --dataset data/datasets/cube_p01_seed
Then open the forwarded port (default 5008) in your browser.

Left-click  = foreground point (add to object)
Right-click = background point (carve away)
Each click re-runs SAM with ALL current points. "Accept" writes a YOLO-seg
polygon label to <dataset>/labels/train/<name>.txt and advances.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, Response
from ultralytics import SAM

app = Flask(__name__)
STATE = {"model": None, "img_dir": None, "lbl_dir": None, "names": []}


def load_model(weights: str):
    print(f"loading SAM ({weights}) ...")
    STATE["model"] = SAM(weights)
    print("SAM ready.")


def list_names():
    imgs = sorted(p.name for p in STATE["img_dir"].glob("*.jpg"))
    return imgs


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/list")
def listing():
    out = []
    for n in STATE["names"]:
        lbl = STATE["lbl_dir"] / (Path(n).stem + ".txt")
        out.append({"name": n, "labeled": lbl.exists()})
    return jsonify(out)


@app.route("/image/<name>")
def image(name):
    return send_file(STATE["img_dir"] / name)


@app.route("/segment", methods=["POST"])
def segment():
    d = request.get_json()
    name = d["name"]
    pts = d["points"]          # [[x,y],...] image px
    labs = d["labels"]         # [1,0,...]
    img = cv2.imread(str(STATE["img_dir"] / name))
    h, w = img.shape[:2]
    res = STATE["model"](img, points=[pts], labels=[labs], verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return jsonify({"polys": [], "w": w, "h": h})
    # pick the mask with the largest area
    polys_all = res.masks.xy  # list of (N,2) pixel-coord arrays
    best = max(polys_all, key=lambda p: cv2.contourArea(p.astype(np.float32))
               if len(p) >= 3 else 0)
    # simplify
    peri = cv2.arcLength(best.astype(np.float32), True)
    approx = cv2.approxPolyDP(best.astype(np.float32), 0.0015 * peri, True)
    poly = approx.reshape(-1, 2).tolist()
    return jsonify({"polys": [poly], "w": w, "h": h})


@app.route("/save", methods=["POST"])
def save():
    d = request.get_json()
    name = d["name"]
    poly = d["poly"]          # normalized [[x,y],...]
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
    out = STATE["lbl_dir"] / (Path(name).stem + ".txt")
    out.write_text(f"0 {coords}\n")
    return jsonify({"ok": True})


@app.route("/unsave", methods=["POST"])
def unsave():
    name = request.get_json()["name"]
    out = STATE["lbl_dir"] / (Path(name).stem + ".txt")
    if out.exists():
        out.unlink()
    return jsonify({"ok": True})


PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>SAM labeler</title>
<style>
 body{font-family:sans-serif;margin:0;background:#1e1e1e;color:#ddd}
 #bar{padding:8px 12px;background:#252526;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 button{padding:6px 12px;font-size:14px;cursor:pointer;background:#0e639c;color:#fff;border:none;border-radius:4px}
 button.sec{background:#444}
 #wrap{position:relative;display:inline-block;margin:10px}
 canvas{cursor:crosshair;display:block}
 #prog{font-size:14px}
 .pill{padding:2px 8px;border-radius:10px;background:#333}
 #thumbs{display:flex;flex-wrap:wrap;gap:3px;padding:6px 12px;background:#252526}
 .th{width:46px;height:26px;object-fit:cover;border:2px solid #555;cursor:pointer}
 .th.done{border-color:#3fb950}
 .th.cur{border-color:#f0c040}
</style></head><body>
<div id=bar>
  <button class=sec onclick=prev()>&larr; Prev</button>
  <span id=prog class=pill>...</span>
  <button class=sec onclick=clearPts()>Clear points</button>
  <button onclick=accept()>Accept &amp; Next &rarr;</button>
  <button class=sec onclick=skip()>Skip</button>
  <span id=hint class=pill>L-click = object &nbsp; R-click = remove</span>
</div>
<div id=thumbs></div>
<div id=wrap><canvas id=cv></canvas></div>
<script>
let names=[], i=0, img=new Image(), scale=1, pts=[], labs=[], poly=null;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
async function boot(){ names=await (await fetch('/list')).json(); renderThumbs(); load(0);}
function renderThumbs(){
  const t=document.getElementById('thumbs'); t.innerHTML='';
  names.forEach((n,k)=>{const im=document.createElement('img');
    im.src='/image/'+n.name; im.className='th'+(n.labeled?' done':'')+(k===i?' cur':'');
    im.onclick=()=>load(k); t.appendChild(im);});
}
function load(k){ i=k; pts=[];labs=[];poly=null;
  img=new Image(); img.onload=()=>{
    const maxW=Math.min(1180, window.innerWidth-40);
    scale=Math.min(1, maxW/img.width);
    cv.width=img.width*scale; cv.height=img.height*scale; redraw();
  }; img.src='/image/'+names[i].name;
  document.getElementById('prog').textContent=
    `img ${i+1}/${names.length}  |  ${names.filter(x=>x.labeled).length} labeled  |  ${names[i].name}`;
  renderThumbs();
}
function redraw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  if(poly){ctx.beginPath();
    poly.forEach((p,j)=>{const x=p[0]*scale,y=p[1]*scale; j?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.closePath(); ctx.fillStyle='rgba(80,220,120,0.35)';
    ctx.strokeStyle='#3fb950'; ctx.lineWidth=2; ctx.fill(); ctx.stroke();}
  pts.forEach((p,j)=>{ctx.beginPath();
    ctx.arc(p[0]*scale,p[1]*scale,5,0,7);
    ctx.fillStyle=labs[j]?'#f0c040':'#e05050'; ctx.fill();});
}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',async e=>{
  const r=cv.getBoundingClientRect();
  const x=(e.clientX-r.left)/scale, y=(e.clientY-r.top)/scale;
  pts.push([x,y]); labs.push(e.button===2?0:1); redraw();
  const res=await (await fetch('/segment',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:names[i].name,points:pts,labels:labs})})).json();
  poly = res.polys.length? res.polys[0] : null; window._wh=[res.w,res.h]; redraw();
});
function clearPts(){pts=[];labs=[];poly=null;redraw();}
async function accept(){
  if(!poly){alert('click the object first');return;}
  const [w,h]=window._wh;
  const norm=poly.map(p=>[Math.min(1,Math.max(0,p[0]/w)),Math.min(1,Math.max(0,p[1]/h))]);
  await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:names[i].name,poly:norm})});
  names[i].labeled=true; next();
}
async function skip(){ next(); }
function next(){ if(i<names.length-1) load(i+1); else {renderThumbs(); alert('done — all images visited');}}
function prev(){ if(i>0) load(i-1); }
document.addEventListener('keydown',e=>{ if(e.key==='Enter')accept();
  if(e.key==='c')clearPts(); if(e.key==='ArrowRight')skip(); if(e.key==='ArrowLeft')prev();});
boot();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--weights", default="mobile_sam.pt")
    ap.add_argument("--port", type=int, default=5008)
    args = ap.parse_args()

    STATE["img_dir"] = args.dataset / "images" / "train"
    STATE["lbl_dir"] = args.dataset / "labels" / "train"
    STATE["lbl_dir"].mkdir(parents=True, exist_ok=True)
    STATE["names"] = list_names()
    if not STATE["names"]:
        raise SystemExit(f"no .jpg in {STATE['img_dir']}")

    # write data.yaml for later finetuning
    (args.dataset / "data.yaml").write_text(
        f"path: {args.dataset.resolve()}\ntrain: images/train\n"
        f"val: images/train\nnc: 1\nnames: [cube]\n")

    load_model(args.weights)
    print(f"\n  {len(STATE['names'])} images. open http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
