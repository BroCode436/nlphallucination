"""Research UI/service extension using the existing application's HTTP framework.

Live recommendations receive no benchmark reference labels. Optional references
are scored only after both actual continuations are generated. Local retrieval
searches real cached passages; it is explicitly not an internet search service.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from .data import read_jsonl
from .detector import ExistingDetector
from .experiment import apply_actions
from .generation import SmallReasoner
from .graph import build_state, node_features
from .metrics import exact_match
from .models import GraphPolicy
from .policy import TabularPolicy, baseline_scores, select_claims
from .schema import Budget

PAGE = (Path(__file__).resolve().parent/'static'/'chc.html').read_text(encoding='utf-8')


class CHCService:
    """Lazy real-model service sharing the original project's saved components.

    Args:
        project: Existing project root with detector and correction checkpoints.
        work: Experiment cache containing smoke/pilot intervention checkpoints.
    Attributes:
        lock: Serializes generation/RNG changes across simultaneous browser calls.
        detector/reasoner: Real local components initialized at service creation.
    """
    def __init__(self, project: Path, work: Path) -> None:
        """Load actual inference models and locate the strongest available run.

        Args:
            project: Existing project root.
            work: Directory containing caches and completed experiment stages.
        Returns:
            None; missing learned checkpoints remain an explicit request error.
        """
        self.project, self.work = project, work
        self.detector = ExistingDetector(project)
        self.reasoner = SmallReasoner(work/'reasoner',max_length=1024,max_new_tokens=24,sample=True)
        self.stage = work/'pilot' if (work/'pilot'/'models'/'tabular_42.joblib').exists() else work/'smoke'
        self.lock = threading.Lock()

    def retrieve(self, question: str) -> list[dict[str, str]]:
        """Rank actual cached corpus documents using transparent lexical matching.

        Args:
            question: User query; no reference answers are searched or returned.
        Returns:
            Up to three real source documents with title, text, and provenance.
        Limitations:
            Small cached corpus and simple lexical retrieval can miss evidence.
        """
        import re
        stopwords={'the','and','what','which','who','where','when','does','was','are','did','have','with','from','that','this','contains'}
        terms = set(re.findall(r'\w{3,}',question.lower()))-stopwords
        documents: dict[str, dict[str, str]] = {}
        for path in (self.work/'hotpot').glob('*.jsonl'):
            for row in read_jsonl(path):
                for title,sentences in zip(row['context']['title'],row['context']['sentences']):
                    documents[title]={'title':title,'text':' '.join(sentences),'source':'Cached HotpotQA context; not live web'}
        def score(document: dict[str, str]) -> float:
            """Rank title matches above generic document-word overlap.

            Args:
                document: Cached title and passage, without a reference answer.
            Returns:
                Heuristic retrieval score, not a relevance probability.
            """
            return 3*len(terms & set(re.findall(r'\w{3,}',document['title'].lower())))+len(terms & set(re.findall(r'\w{3,}',document['text'].lower())))
        ranked = sorted(documents.values(),key=score,reverse=True)
        return [d for d in ranked[:3] if score(d)>0]

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute a live intervention without leaking optional evaluation labels.

        Args:
            data: Question, optional evidence/trace, budget, policy, and optional
                reference answer. Reference is used only after model outputs exist.
        Returns:
            Predictions, evidence, selected actions, actual continuations, and
            post-hoc exact-match evaluation only when a reference was supplied.
        Raises:
            ValueError: For invalid inputs, missing evidence, or unsupported policy.
        """
        question = data.get('question','')
        if not isinstance(question,str) or not question.strip() or len(question)>3000:
            raise ValueError('Supply a nonempty question of at most 3000 characters')
        evidence = data.get('evidence','')
        if not isinstance(evidence,str) or len(evidence)>30000:
            raise ValueError('Evidence must be text of at most 30,000 characters')
        documents = [{'title':'User supplied passage','text':evidence,'source':'User supplied; not authenticated'}] if evidence.strip() else self.retrieve(question)
        if not documents:
            return {'status':'evidence_insufficient','error':'No relevant local passages. Supply evidence explicitly.'}
        passages = {f'doc{i}':d for i,d in enumerate(documents)}
        trace = data.get('trace',[])
        if not isinstance(trace,list) or any(not isinstance(t,str) or len(t)>5000 for t in trace):
            raise ValueError('Trace must be a list of short claim strings')
        import re
        if not trace:
            terms=set(question.lower().split())
            candidates=[(len(terms & set(s.lower().split())),k,s) for k,d in passages.items()
                        for s in re.split(r'(?<=[.!?])\s+',d['text']) if len(s.split())>3]
            candidates.sort(reverse=True)
            seen:set[str]=set();facts=[]
            for _,key,sentence in candidates:
                if sentence not in seen:
                    seen.add(sentence);facts.append({'text':sentence,'evidence_id':key,'title':passages[key]['title']})
                if len(facts)==3:break
        else:
            facts=[{'text':t,'evidence_id':max(passages,key=lambda k:len(set(t.lower().split())&set(passages[k]['text'].lower().split()))),
                    'title':'User supplied trace'} for t in trace[:3]]
        if not facts:raise ValueError('Could not extract a claim; supply a trace')
        budget=data.get('budget',1)
        if not isinstance(budget,int) or not 0<=budget<=3:raise ValueError('Budget must be 0–3 calls')
        method=data.get('policy','chc_tabular')
        with self.lock:
            visible_id='live-'+hashlib.sha256(question.encode()).hexdigest()[:12]
            state=self.detector.annotate(build_state(visible_id,question,facts,passages))
            if method=='chc_tabular':scores=TabularPolicy(self.stage/'models'/'tabular_42.joblib').scores(state)
            elif method=='chc_graph':scores=GraphPolicy(self.stage/'models'/'graph_42.pt').scores(state)
            else:scores=baseline_scores(state,method)
            chosen=select_claims(state,scores,Budget(budget),overlap_discount=method.startswith('chc'))
            before=self.reasoner.generate(state,101)
            actions=[self.detector.verify(state,node) for node in chosen]
            changed=apply_actions(state,actions)
            after=self.reasoner.generate(changed,101)
        evaluation=None
        reference=data.get('reference','')
        if reference:
            if not isinstance(reference,str):raise ValueError('Reference must be text')
            b,a=exact_match(before['answer'],reference),exact_match(after['answer'],reference)
            evaluation={'basis':'User supplied reference; normalized exact match', 'before_correct':b,
                        'after_correct':a,'observed_final_error_reduction':int(a)-int(b),
                        'broader_downstream_error_count':None}
        x,_,ids=node_features(state)
        from .graph import FEATURE_NAMES
        return {'state':state.to_dict(),'selected':chosen,'scores':scores,'budget':budget,'actions':actions,
                'features':{node:dict(zip(FEATURE_NAMES,map(float,row))) for node,row in zip(ids,x)},
                'before':before,'after':after,'corrected_state':changed.to_dict(),'evaluation':evaluation,
                'retrieval_basis':documents[0]['source'],'graph_basis':'Uncertain textual dependencies, not verified causal edges',
                'policy_training_scope':'Observed final-answer effects in a small controlled working-note experiment'}


def serve(project: Path, work: Path, port: int = 8766) -> None:
    """Run an optional standalone CHC HTTP view using the existing server framework.

    Args:
        project: Original detector/correction project root.
        work: Experiment outputs and model-cache root.
        port: Localhost port; original correction app can also delegate /chc here.
    Returns:
        None; blocks serving requests until interrupted.
    """
    service=CHCService(project,work)
    class Handler(BaseHTTPRequestHandler):
        """Serve the research page and real-model JSON endpoint on localhost."""
        def do_GET(self) -> None:
            """Return the HTML interface for root or /chc; reject unknown routes.

            Returns:
                None; writes a UTF-8 HTTP response to the requesting client.
            """
            if self.path not in ('/','/chc'):self.send_error(404);return
            body=PAGE.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self) -> None:
            """Validate request size/origin and execute the real CHC service.

            Returns:
                None; writes JSON success or an explicit error without fake output.
            """
            if self.path!='/api/chc':self.send_error(404);return
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{port}':self.send_error(403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=100000:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('Expected JSON object')
                result=service.analyze(data);status=200
            except Exception as error:
                result={'error':str(error),'status':'request_or_regeneration_failed'};status=400
            body=json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    print(f'CHC interface: http://127.0.0.1:{port}/chc',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
