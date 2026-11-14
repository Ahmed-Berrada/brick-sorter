"""Control page: live cameras, loop state, last inspection + heatmap, latencies, bowl counts and controls.

GET  /                      the page
GET  /api/status            orchestrator snapshot (JSON)
GET  /stream/<camera>       MJPEG stream (wrist | top | inspect)
GET  /heatmap.jpg           last inspection crop with the anomaly map
POST /api/<cmd>             start | pause | resume | stop | estop | next | step_on | step_off
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Brick Sorter</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#9aa3b2;--line:#2a2f3a;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:18px;margin:0}.pill{padding:3px 10px;border-radius:99px;background:var(--line);font-weight:600}
main{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:12px 16px}@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}@media(max-width:700px){.cams{grid-template-columns:1fr}}
.cams figure{margin:0}.cams img{width:100%;border-radius:6px;background:#000;aspect-ratio:4/3;object-fit:cover}
figcaption{color:var(--mut);font-size:12px;margin-top:4px}
button{background:var(--line);color:var(--fg);border:1px solid #3a4150;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer}
button:hover{border-color:var(--acc)}#estop{background:var(--bad);border-color:var(--bad);color:#fff;font-size:15px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}table{width:100%;border-collapse:collapse}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--mut);font-weight:500}
.bowls{display:flex;gap:10px}.bowl{flex:1;text-align:center;padding:10px;border-radius:8px;border:2px solid}
.bowl b{display:block;font-size:26px}#events{max-height:280px;overflow:auto;font:12px ui-monospace,monospace;color:var(--mut)}
#heat{width:100%;max-width:320px;border-radius:6px;background:#000}.big{font-size:22px;font-weight:700}
.err{color:var(--bad)}.mut{color:var(--mut)}
</style></head><body>
<header><h1>Brick Sorter</h1><span class="pill" id="state">…</span><span class="mut" id="detail"></span>
<span style="flex:1"></span><button id="estop">E-STOP</button></header>
<main>
<section style="display:grid;gap:12px">
 <div class="card cams">
  <figure><img src="/stream/wrist" alt="wrist camera"><figcaption>wrist</figcaption></figure>
  <figure><img src="/stream/top" alt="top camera"><figcaption>top</figcaption></figure>
  <figure><img src="/stream/inspect" alt="inspect camera"><figcaption>inspect</figcaption></figure>
 </div>
 <div class="card"><div class="row">
  <button data-c="start">Start</button><button data-c="pause">Pause</button><button data-c="resume">Resume</button>
  <button data-c="stop">Stop</button><label><input type="checkbox" id="step"> step mode</label>
  <button data-c="next">Next brick</button><span class="err" id="error"></span></div>
  <p class="mut" id="poses"></p></div>
 <div class="card"><h3>Events</h3><div id="events"></div></div>
</section>
<section style="display:grid;gap:12px;align-content:start">
 <div class="card"><h3>Bowls</h3><div class="bowls" id="bowls"></div><p class="mut" id="tpm"></p></div>
 <div class="card"><h3>Last inspection</h3><div class="big" id="outcome">–</div><p id="reason" class="mut"></p>
  <img id="heat" alt="anomaly heatmap"><table id="probs"></table></div>
 <div class="card"><h3>Latency (median / p99)</h3><table id="lat"></table><p class="mut" id="dev"></p></div>
</section></main>
<script>
const $=id=>document.getElementById(id);
const COL={yellow:'#e3b341',orange:'#f0883e',purple:'#a371f7'};
async function cmd(c){const r=await fetch('/api/'+c,{method:'POST'});const j=await r.json();if(!r.ok)$('error').textContent=j.error||r.status;}
document.querySelectorAll('button[data-c]').forEach(b=>b.onclick=()=>cmd(b.dataset.c));
$('estop').onclick=()=>cmd('estop');$('step').onchange=e=>cmd(e.target.checked?'step_on':'step_off');
let heatT=0;
async function tick(){try{const s=await (await fetch('/api/status')).json();
 $('state').textContent=s.state;$('detail').textContent=s.detail||'';$('error').textContent=s.error||'';
 $('state').style.background=s.state==='error'?'var(--bad)':s.running?'var(--ok)':'var(--line)';
 $('step').checked=s.step_mode;
 $('poses').textContent=s.missing_poses.length?'Teach these poses before starting: '+s.missing_poses.join(', '):'';
 const bowls=Object.assign({yellow:0,orange:0,purple:0},s.counts);
 $('bowls').innerHTML=Object.entries(bowls).map(([k,v])=>`<div class="bowl" style="border-color:${COL[k]||'#888'}"><b>${v}</b>${k}</div>`).join('');
 $('tpm').textContent=s.tasks_per_min?`${s.tasks_per_min} bricks / min`:'';
 if(s.inspection){const i=s.inspection;$('outcome').textContent=i.outcome+(s.decision&&s.decision.bowl?' → '+s.decision.bowl:'');
  $('reason').textContent=`${i.reason} · score ${i.score} (threshold ${i.threshold.toFixed(6)})`;
  $('probs').innerHTML=Object.entries(i.probabilities).map(([k,v])=>`<tr><td>${k}</td><td>${(v*100).toFixed(0)}%</td></tr>`).join('');
  if(Date.now()-heatT>1500){$('heat').src='/heatmap.jpg?'+Date.now();heatT=Date.now();}}
 $('lat').innerHTML='<tr><th>stage</th><th>n</th><th>median</th><th>p99</th></tr>'+Object.entries(s.latency).map(([k,v])=>`<tr><td>${k}</td><td>${v.n}</td><td>${v.median??'–'}</td><td>${v.p99??'–'}</td></tr>`).join('');
 $('dev').textContent=`policy on ${s.devices.policy} · inspection on ${s.devices.inspector} · log ${s.run_dir}`;
 $('events').innerHTML=s.events.slice().reverse().map(e=>`<div>${new Date(e.t*1000).toLocaleTimeString()} #${e.brick} [${e.state}] ${e.msg}</div>`).join('');
}catch(e){$('state').textContent='offline';}setTimeout(tick,500);}
tick();
</script></body></html>"""


def serve(orch, host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    cmds = {"start": orch.start, "pause": orch.pause, "resume": orch.resume, "stop": orch.stop,
            "estop": orch.estop, "next": orch.next}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(200, orch.snapshot())
            if path == "/heatmap.jpg":
                ins = orch.last_inspection
                if ins is None or not ins.heatmap_jpeg:
                    return self._send(404, b"", "image/jpeg")
                return self._send(200, ins.heatmap_jpeg, "image/jpeg")
            if path.startswith("/stream/"):
                cam = orch.cameras.get(path.removeprefix("/stream/"))
                if cam is None:
                    return self._send(404, b"no such camera", "text/plain")
                return self._stream(cam)
            self._send(404, b"not found", "text/plain")

        def _stream(self, cam) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    seq = cam.wait_new(seq, 2.0)
                    jpg = cam.jpeg()
                    if jpg is None:
                        time.sleep(0.2)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 15)  # 15 fps preview is plenty and spares the CPU
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            c = self.path.removeprefix("/api/").split("?")[0]
            try:
                if c == "step_on":
                    orch.step_mode = True
                elif c == "step_off":
                    orch.step_mode = False
                    orch.next()
                elif c in cmds:
                    cmds[c]()
                else:
                    return self._json(404, {"error": f"unknown command {c}"})
                self._json(200, {"ok": True})
            except Exception as e:  # noqa: BLE001
                self._json(400, {"error": str(e)})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="web", daemon=True).start()
    log.info("control page: http://%s:%d", "localhost" if host in ("0.0.0.0", "") else host, srv.server_address[1])
    return srv
