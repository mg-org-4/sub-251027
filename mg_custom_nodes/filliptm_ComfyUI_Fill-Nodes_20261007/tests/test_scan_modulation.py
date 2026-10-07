import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np

root=Path(__file__).parents[1]/"nodes"
for name,path in (("modulation_test",root),("modulation_test.vfx",root/"vfx"),("modulation_test.audio",root/"audio")):
    package=ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
fx=importlib.import_module("modulation_test.vfx.FL_InteractiveScanFX")
mod=importlib.import_module("modulation_test.vfx.scan_modulation")
scan=importlib.import_module("modulation_test.vfx.FL_StreetScan")
edit=importlib.import_module("modulation_test.vfx.FL_ScanAudioEdit")


class MappingTests(unittest.TestCase):
    def curves(self,rows):
        settings=fx.DEFAULTS|{"relief":.65,"animation":.18,"speed":.7,"motion_strength":.6,"audio_mappings":rows}
        return mod.compile_mappings(settings,[{"values":[0,1,0,1,0,1,0,1]}]*3,24,[4,4])

    def row(self,**changes):
        return {"source":0,"target":"relief","minimum":.5,"maximum":1,"start_frame":2,"end_frame":6,"smoothing":0}|changes

    def test_mapping_is_absolute_and_only_active_in_range(self):
        values,_=self.curves([self.row()])
        self.assertEqual(values["relief"],[.65,.65,.5,1,.5,1,.65,.65])
        values,_=self.curves([self.row(invert=True)])
        self.assertEqual(values["relief"][2:6],[1,.5,1,.5])

    def test_smoothing_resets_at_shot_boundary(self):
        values,_=self.curves([self.row(start_frame=0,end_frame=None,smoothing=.2)])
        self.assertGreater(values["relief"][1],.5)
        self.assertLess(values["relief"][1],1)
        self.assertEqual(values["relief"][4],.5)

    def test_bad_ranges_and_overlaps_fail(self):
        for rows in ([self.row(end_frame=9)],[self.row(source=3)],[self.row(target="fps")],
                     [self.row(),self.row(start_frame=5,end_frame=8)], [self.row(minimum=float("nan"))]):
            with self.assertRaises(ValueError):self.curves(rows)
        self.curves([self.row(end_frame=4),self.row(start_frame=4)])

    def test_disabled_row_does_not_override(self):
        values,_=self.curves([self.row(enabled=False)])
        self.assertEqual(values["relief"],[.65]*8)

    def test_zero_weight_total_fails(self):
        with self.assertRaisesRegex(ValueError,"positive total"):
            self.curves([self.row(target="voxel_weight",minimum=0,maximum=0,start_frame=0,end_frame=None),
                         self.row(target="edge_weight",minimum=0,maximum=0,start_frame=0,end_frame=None)])

    def test_depth_selection_is_seeded_and_latched(self):
        values,_=self.curves([])
        values["depth_weight"]=[100]*8;values["voxel_weight"]=[0]*8;values["edge_weight"]=[0]*8
        def events():return edit.cursor_plan(([1,0,0,0,1,0,0,0],)*3,3,10,24)
        a=edit.assign_reveal_layers(events(),values,73);b=edit.assign_reveal_layers(events(),values,73)
        self.assertTrue(a);self.assertEqual(a,b);self.assertTrue(all(e["effect"]==2 for e in a))

    def test_native_camera_anchors_steady_depth(self):
        d=np.full((64,64),.6,np.float32)
        x,y,z=scan.project_parallax(d,0,0,1,1,1,.1,-.1,.2,.6)
        yy,xx=np.mgrid[:64,:64]
        np.testing.assert_allclose(x,xx,atol=1e-5);np.testing.assert_allclose(y,yy,atol=1e-5)
        moved=scan.project_parallax(np.full_like(d,.1),0,0,1,1,1,.1,-.1,.2,.6)
        self.assertGreater(float(np.abs(moved[0]-x).max()),.5)
        self.assertTrue(np.isfinite(moved[2]).all())

    def test_native_camera_matches_old_at_neutral_settings(self):
        d=np.random.default_rng(1).random((48,64)).astype(np.float32)
        old=scan.project_depth(d,.35,5.5,1,.74)
        new=scan.project_parallax(d,.35,5.5,1,.74,0,0,0,0,.5)
        for a,b in zip(old,new):np.testing.assert_allclose(a,b,atol=1e-5)


if __name__=="__main__":unittest.main()
