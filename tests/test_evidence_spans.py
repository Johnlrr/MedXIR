import unittest
from scripts.evidence_spans import join_overlapping,centered_span,contiguous_around_anchor


class EvidenceSpans(unittest.TestCase):
    def test_original_overlap_and_contained_row(self):
        text='A patient.\nNext paragraph. 中文。'
        rows=[dict(sha='s',start=a,end=b,text=text[a:b]) for a,b in [(0,18),(10,len(text)),(12,16)]]
        self.assertEqual(join_overlapping(rows),(0,text))

    def test_reject_gap_or_wrong_identity(self):
        base=dict(sha='a',start=0,end=3,text='abc')
        for other in [dict(sha='a',start=4,end=5,text='d'),dict(sha='b',start=2,end=4,text='cd'),
                      dict(sha='a',start=2,end=4,text='Xd')]:
            with self.assertRaises(ValueError):join_overlapping([base,other])

    def test_window_is_source_slice_and_keeps_anchor_centre(self):
        text='First sentence.\nSecond sentence.\nThird sentence. 最后。'
        offsets=[(i,i+1) for i in range(len(text))]
        for start,end in [(0,10),(17,32),(len(text)-5,len(text))]:
            a,b,span=centered_span(text,offsets,start,end,24)
            self.assertEqual(span,text[a:b]);self.assertLessEqual(len(span),24)
            self.assertLessEqual(a,(start+end)//2);self.assertGreaterEqual(b,(start+end)//2)

    def test_short_document_is_unchanged(self):
        text='  A\nB  '
        self.assertEqual(centered_span(text,[(2,3),(4,5)],2,5,8),(0,len(text),text))

    def test_distant_gap_preserves_connected_neighbours(self):
        text='abc defghijk  nopqrst'
        rows=[dict(sha='s',start=a,end=b,text=text[a:b]) for a,b in [(0,3),(4,10),(8,12),(14,20)]]
        self.assertEqual(contiguous_around_anchor(rows,rows[1]),(4,text[4:12]))
        self.assertEqual(contiguous_around_anchor(rows,rows[2]),(4,text[4:12]))
        self.assertEqual(contiguous_around_anchor(rows,rows[0]),(0,'abc'))

    def test_malformed_component_is_not_silently_ignored(self):
        anchor=dict(sha='s',start=0,end=3,text='abc')
        rows=[anchor,dict(sha='s',start=6,end=9,text='xyz'),dict(sha='s',start=8,end=10,text='QQ')]
        with self.assertRaisesRegex(ValueError,'disagree'):
            contiguous_around_anchor(rows,anchor)


if __name__=='__main__':unittest.main()
