from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_passage_pipeline import check_prepare_against_pilot


class PilotPreparationGate(unittest.TestCase):
    def test_exact_match_is_order_independent_and_detects_text_or_fanout_change(self):
        a=dict(query_id=1,doc_id=4,text_sha256='s',seed='s-0',seed_rank=0,
               start=10,end=13,chunk_text='abc',passage_tokens=2,doc_ids=[4,5])
        b={**a,'query_id':2}
        pilot=pd.DataFrame([a,b]);candidate=pilot.iloc[::-1].copy()
        check_prepare_against_pilot(candidate,pilot)
        broken=candidate.copy();broken.loc[0,'chunk_text']='xyz'
        with self.assertRaises(AssertionError):check_prepare_against_pilot(broken,pilot)
        broken=candidate.copy();broken.at[0,'doc_ids']=[4]
        with self.assertRaisesRegex(ValueError,'fanout'):check_prepare_against_pilot(broken,pilot)


if __name__=='__main__':unittest.main()
