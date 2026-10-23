"""Small author-written challenge suite. Not a human-reviewed benchmark.
"""
from typing import Any as _DocAny
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from correction_validity import CorrectionValidityModel
CASES=[
 ('date_repair','Chandrayaan-3 landed on August 23, 2023. It was developed by ISRO.','Chandrayaan-3 landed on August 24, 2023.','Chandrayaan-3 landed on August 23, 2023.',True),
 ('retain_developer','Chandrayaan-3 landed on August 23, 2023 and was developed by ISRO.','Chandrayaan-3 landed on August 24, 2023 and was developed by ISRO.','Chandrayaan-3 landed on August 23, 2023 and was developed by ISRO.',True),
 ('drop_developer','Chandrayaan-3 landed on August 23, 2023 and was developed by ISRO.','Chandrayaan-3 landed on August 24, 2023 and was developed by ISRO.','Chandrayaan-3 landed on August 23, 2023.',False),
 ('wrong_entity','Alice scored 10 points. Bob scored 20 points.','Alice scored 30 points.','Bob scored 20 points.',False),
 ('wrong_binding','Alice scored 10 points. Bob scored 20 points.','Alice scored 30 points.','Alice scored 20 points.',False),
 ('correct_unchanged','Alice scored 10 points.','Alice scored 10 points.','Alice scored 10 points.',True),
 ('error_unchanged','Alice scored 10 points.','Alice scored 30 points.','Alice scored 30 points.',False),
 ('hindi_repair','आरव ने १० पदक जीते और दिल्ली का प्रतिनिधित्व किया।','आरव ने २० पदक जीते और दिल्ली का प्रतिनिधित्व किया।','आरव ने १० पदक जीते और दिल्ली का प्रतिनिधित्व किया।',True),
 ('hindi_drop','आरव ने १० पदक जीते और दिल्ली का प्रतिनिधित्व किया।','आरव ने २० पदक जीते और दिल्ली का प्रतिनिधित्व किया।','आरव ने १० पदक जीते।',False),
 ('hinglish_repair','Aarav ne 10 medals jeete aur Delhi ko represent kiya.','Aarav ne 20 medals jeete aur Delhi ko represent kiya.','Aarav ne 10 medals jeete aur Delhi ko represent kiya.',True),
 ('decimal_error','The mass is 1.5 kg.','The mass is 1.6 kg.','The mass is 1.6 kg.',False),
 ('signed_error','The temperature is -5 degrees.','The temperature is 5 degrees.','The temperature is 5 degrees.',False),
 ('negation','The medicine does not cure the disease.','The medicine cures the disease.','The medicine does cure the disease.',False),
 ('role_reversal','Alice hired Bob.','Alice hired Bob.','Bob hired Alice.',False),
 ('paraphrase','Alice purchased a car.','Alice purchased a bicycle.','Alice bought a car.',True),
 ('unit_error','The distance is 10 kilometers.','The distance is 20 kilometers.','The distance is 10 meters.',False),
 ('empty_evidence','','Alice scored 30 points.','Alice scored 10 points.',False),
 ('reordered_clauses','Aster Museum opened in 1998 and is located in Riverford.','Aster Museum opened in 2001 and is located in Riverford.','Aster Museum is located in Riverford and opened in 1998.',True),
 ('unsupported_added_detail','Alice scored 10 points.','Alice scored 30 points.','Alice scored 10 points in Paris.',False),
 ('correct_entity_repair','The mission was developed by ISRO.','The mission was developed by NASA.','The mission was developed by ISRO.',True),
]
def main() -> None:
    """Main.

    This operation participates in the saved local NLP experiment workflow.

    Learning notes — input/output contract:
        Parameters: No explicit caller arguments beyond an instance/class if applicable.
        Returns:
            None; effects or assertions are performed directly.
        Assumptions and failures:
            Inputs must match the saved dataset/model schema. Missing files, invalid
            fields, or incompatible checkpoints can raise errors; tests cover normal
            and selected failure paths. Scores are not independent factual proof.

    """
    model=CorrectionValidityModel();results=[]
    for name,ev,orig,cand,gold in CASES:
        assessed=model.assess(ev,orig,cand)
        results.append(dict(name=name,evidence=ev,original=orig,candidate=cand,gold=gold,correct=assessed['accepted']==gold,**assessed))
    report={'basis':'20 author-written diagnostic cases; labels not independently reviewed',
            'correct':sum(r['correct'] for r in results),'n':len(results),'results':results}
    (ROOT/'evaluation'/'challenge_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'correct':report['correct'],'n':report['n'],'failures':[r['name'] for r in results if not r['correct']]},indent=2))
if __name__=='__main__':main()
