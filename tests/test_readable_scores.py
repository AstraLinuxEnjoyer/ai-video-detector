import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from demo import app as module
from demo.human_report import render_report
from demo.presentation import signal, signal_level, points


def report_fixture():
    return dict(filename='<unsafe>.mp4', metadata=dict(width=640,height=360,duration=4),
        frames=[dict(time=0,spatial=.13632)],clips=[],models={},warnings=[],
        spectral=dict(status='complete',score=.5085,threshold=.5808323213930072,samples=[]),
        summary=dict(verdict='uncertain',combined=.639,spatial=.713,temporal=None,spectral=.5085,
            threshold=.65,segments=[],review_segments=[],weights=dict(spatial=.63,temporal=.27,spectral=.1)))


class ReadableScoresTests(unittest.TestCase):
    def test_units_do_not_claim_probabilities(self):
        self.assertEqual(signal(.639),'63,9 из 100')
        self.assertEqual(signal(0),'0 из 100')
        self.assertEqual(signal(1),'100 из 100')
        for missing in (None,float('nan'),float('inf'),-.1,1.1,True):
            self.assertEqual(signal(missing),'Нет оценки')

    def test_display_rounding_does_not_change_boundary_decisions(self):
        self.assertEqual(signal_level(.649999),'Повышенный сигнал')
        self.assertEqual(signal_level(.65),'Сильный сигнал')
        self.assertEqual(signal_level(.349999),'Слабый сигнал')
        self.assertEqual(signal_level(.35),'Повышенный сигнал')
        self.assertEqual(signal_level(.6,threshold=.580832),'Сильный сигнал')

    def test_influence_is_a_difference_not_a_probability(self):
        self.assertEqual(points(.125),'12,5 п. шкалы')
        self.assertEqual(points(-.05),'-5 п. шкалы')

    def test_report_preserves_source_and_explains_each_model(self):
        job=report_fixture()
        original=copy.deepcopy(job)
        with tempfile.TemporaryDirectory() as root:
            html=render_report(job,Path(root))
        self.assertEqual(job,original)
        self.assertIn('63,9 из 100',html)
        self.assertIn('58,1 из 100',html)
        self.assertIn('не вероятность',html)
        for model in ('Изображение','Движение','Спектр'):
            self.assertIn(f'Как читать оценку «{model}»',html)
        self.assertNotIn('<unsafe>',html)
        self.assertIn('&lt;unsafe&gt;',html)

    def test_download_updates_presentation_without_modifying_saved_data(self):
        with tempfile.TemporaryDirectory() as root, patch.object(module,'RUNTIME',Path(root)):
            folder=Path(root)/('a'*32)
            folder.mkdir()
            source=json.dumps(report_fixture(),ensure_ascii=False).encode()
            (folder/'report.json').write_bytes(source)
            (folder/'report.html').write_text('historical presentation',encoding='utf-8')
            response=TestClient(module.app).get(f'/api/jobs/{folder.name}/files/report.html')
            self.assertEqual(response.status_code,200)
            self.assertIn('63,9 из 100',response.text)
            self.assertIn('attachment',response.headers['content-disposition'])
            self.assertEqual((folder/'report.json').read_bytes(),source)
            self.assertEqual((folder/'report.html').read_text(),'historical presentation')


if __name__=='__main__':
    unittest.main()
