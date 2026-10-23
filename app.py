"""Local correction and observable-propagation interfaces over saved NLP models.

Run python app.py --port 8765. Original verification routes and CHC remain;
CHC-Propagate adds recorded examples and optional local model inference. Public
weights may be downloaded on first setup, but no paid inference API is required.
"""
from typing import Any as _DocAny
import argparse
import os
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parent
if (ROOT/'.chc_deps').exists():sys.path.insert(0,str(ROOT/'.chc_deps'))
sys.path.insert(0,str(ROOT/'src'))
from correction_validity import CorrectionValidityModel

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Evidence-backed correction verifier</title>
<style>body{font:17px system-ui;max-width:950px;margin:40px auto;padding:0 20px;background:#101827;color:#edf2fa}h1{font-size:32px}textarea,input{box-sizing:border-box;width:100%;padding:12px;margin:8px 0 22px;background:#1b2940;color:white;border:1px solid #53627c;border-radius:8px}textarea{min-height:110px}button{background:#a9f0cb;border:0;padding:14px 24px;border-radius:8px;font-weight:bold;cursor:pointer}section{background:#1b2940;border-radius:12px;padding:22px;margin:24px 0}p{line-height:1.6}a{color:#a9d5ff}.muted{color:#b8c5d9}pre{white-space:pre-wrap;overflow-wrap:anywhere}label{font-weight:650}</style>
<h1>Evidence-backed correction verifier</h1><p class="muted">A trained local model checks evidence support and preservation separately. Supply source text yourself. This application does not search the web or prove factual truth.</p>
<label for="original">Original claim</label><textarea id="original">The Aster Museum opened in 2001 and is located in Riverford.</textarea>
<label for="evidence">Source passage</label><textarea id="evidence">The Aster Museum opened in 1998 and is located in Riverford. The Hazel Museum opened in 2005 and is located in Hillcrest.</textarea>
<label for="url">Source URL (optional; a user-supplied reference)</label><input id="url" placeholder="https://...">
<label for="candidate">Proposed correction (optional; leave blank for extractive repair)</label><textarea id="candidate" placeholder="Try a valid repair, a wrong-entity copy, or deletion of the location."></textarea>
<button id="run">Verify / repair</button><div id="result" aria-live="polite"></div>
<script>const el=id=>document.getElementById(id);el('run').onclick=async()=>{el('run').disabled=true;el('result').textContent='Checking…';try{const response=await fetch('/api/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({original:el('original').value,candidate:el('candidate').value,sources:[{text:el('evidence').value,url:el('url').value}]})});const data=await response.json();el('result').replaceChildren();const section=document.createElement('section');const h=document.createElement('h2');h.textContent=data.status||'Error';section.append(h);const p=document.createElement('p');p.textContent=data.final_claim||data.error;section.append(p);for(const s of data.sources||[]){if(s.url){const a=document.createElement('a');a.href=s.url;a.textContent='Open supplied source';a.target='_blank';a.rel='noopener noreferrer';section.append(a)}}const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Scores, evidence excerpts, and candidate decisions';const pre=document.createElement('pre');pre.textContent=JSON.stringify(data,null,2);details.append(summary,pre);section.append(details);el('result').append(section)}catch(e){el('result').textContent=e.message}finally{el('run').disabled=false}};</script></html>'''

def verify_payload(model: _DocAny,data: _DocAny) -> _DocAny:
    """Verify payload.

    Serves the local correction UI and JSON endpoint; sources remain user supplied.

    Learning notes — input/output contract:
        Parameters:
            model: Actual fitted model used for prediction or evaluation.
            data: Input structured request/record dictionary; validate fields before use.
        Returns:
            The value assembled by this function (score, table, records, or model as
            described above). The Any type preserves the legacy dynamic interface;
            inspect the documented call site for concrete tensor/table shapes.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    if not isinstance(data,dict):raise ValueError('Expected a JSON object')
    original=data.get('original',''); candidate=data.get('candidate',''); sources=data.get('sources',[])
    if not isinstance(original,str) or not original.strip():raise ValueError('Original claim is required')
    if not isinstance(candidate,str):raise ValueError('Candidate must be text')
    if not isinstance(sources,list) or not 1<=len(sources)<=20:raise ValueError('Supply 1–20 source passages')
    cleaned=[]
    for source in sources:
        if not isinstance(source,dict) or not isinstance(source.get('text'),str):raise ValueError('Source must contain text')
        url=source.get('url','')
        if not isinstance(url,str) or (url and (urlparse(url).scheme not in ('http','https') or not urlparse(url).netloc)):
            raise ValueError('Source URLs must be http(s) URLs')
        cleaned.append({'text':source['text'],'url':url,'provenance':'User supplied; URL contents were not fetched or authenticated'})
    evidence='\n'.join(s['text'] for s in cleaned)
    if max(len(original),len(candidate),len(evidence))>30000:raise ValueError('Text exceeds 30,000-character limit')
    result=model.assess(evidence,original,candidate) if candidate.strip() else model.propose(evidence,original)
    if candidate.strip():result['proposal']=candidate
    result['sources']=cleaned
    return result

def main() -> None:
    """Main.

    Serves the local correction UI and JSON endpoint; sources remain user supplied.

    Learning notes — input/output contract:
        Parameters: No explicit caller arguments beyond an instance/class if applicable.
        Returns:
            None; effects or assertions are performed directly.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    model=CorrectionValidityModel()
    # CHC loads large components only on its first API request. The original
    # correction interface therefore keeps its previous lightweight startup.
    chc_services = []
    propagation_services = []
    class Handler(BaseHTTPRequestHandler):
        """Group related Handler operations.

        Responsibility: Serves the local correction UI and JSON endpoint; sources remain user supplied.
        Constructor arguments and function contracts are documented on the methods. Instance attributes are assigned in __init__; tests construct fresh instances to avoid cross-test state.
        """
        def do_GET(self) -> _DocAny:
            """Do get.

            Serves the local correction UI and JSON endpoint; sources remain user supplied.

            Learning notes — input/output contract:
                Parameters: No explicit caller arguments beyond an instance/class if applicable.
                Returns:
                    The value assembled by this function (score, table, records, or model as
                    described above). The Any type preserves the legacy dynamic interface;
                    inspect the documented call site for concrete tensor/table shapes.
                Assumptions and failures:
                    Inputs must match the saved dataset/model schema. Missing files, invalid
                    fields, or incompatible checkpoints can raise errors; tests cover normal
                    and selected failure paths. Scores are not independent factual proof.

            """
            if self.path == '/chc/propagate':
                page = (ROOT/'chc'/'static'/'propagate.html').read_text(encoding='utf-8')
            elif self.path == '/chc':
                # Static page loading avoids importing PyTorch on a GET request.
                page = (ROOT/'chc'/'static'/'chc.html').read_text(encoding='utf-8')
            elif self.path == '/':
                page = PAGE.replace('<h1>', '<p><a href="/chc">Open CHC research interface</a></p><h1>', 1)
            else:
                self.send_error(404)
                return
            body=page.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self) -> _DocAny:
            """Do post.

            Serves the local correction UI and JSON endpoint; sources remain user supplied.

            Learning notes — input/output contract:
                Parameters: No explicit caller arguments beyond an instance/class if applicable.
                Returns:
                    The value assembled by this function (score, table, records, or model as
                    described above). The Any type preserves the legacy dynamic interface;
                    inspect the documented call site for concrete tensor/table shapes.
                Assumptions and failures:
                    Inputs must match the saved dataset/model schema. Missing files, invalid
                    fields, or incompatible checkpoints can raise errors; tests cover normal
                    and selected failure paths. Scores are not independent factual proof.

            """
            if self.path not in ('/api/verify','/api/chc','/api/chc-v2'):self.send_error(404);return
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{args.port}':self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=200000:raise ValueError('Request size must be between 1 and 200,000 bytes')
                data=json.loads(self.rfile.read(size))
                if self.path == '/api/chc-v2':
                    from chc.v2.web import PropagationService
                    if not propagation_services:
                        propagation_services.append(PropagationService(ROOT))
                    result=propagation_services[0].analyze(data)
                elif self.path == '/api/chc':
                    from chc.web import CHCService
                    if not isinstance(data, dict):
                        raise ValueError('Expected JSON object')
                    if not chc_services:
                        chc_services.append(CHCService(ROOT, Path(os.environ.get('CHC_WORK', str(ROOT/'outputs'/'chc')))))
                    result=chc_services[0].analyze(data)
                else:
                    result=verify_payload(model,data)
                status=200
            except (ValueError,TypeError,FileNotFoundError) as exc:result={'error':str(exc),'status':'request_or_regeneration_failed'};status=400
            except RuntimeError as exc:result={'error':str(exc),'status':'downstream_regeneration_failed'};status=503
            body=json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    print(f'Open http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
