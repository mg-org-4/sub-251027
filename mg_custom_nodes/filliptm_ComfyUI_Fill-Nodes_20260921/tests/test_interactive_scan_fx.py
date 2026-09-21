import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import av
import torch

root = Path(__file__).parents[1] / "nodes"
for name, path in (("interactive_test",root),("interactive_test.vfx",root/"vfx"),("interactive_test.audio",root/"audio")):
    package=ModuleType(name)
    package.__path__=[str(path)]
    sys.modules[name]=package
fx=importlib.import_module("interactive_test.vfx.FL_InteractiveScanFX")


class InteractiveScanTests(unittest.TestCase):
    def test_layer_controls_and_mappings_render_deterministically(self):
        args = self.inputs()
        settings = fx.DEFAULTS | {"stack_count":6,"stack_palette":"cyan","stack_rotation":4,
            "window_order":"random_on_snare","window_blend":"screen","window_fade_in":.1,"window_fade_out":.1,
            "depth_weight":50,"depth_opacity":.6,"audio_mappings":[
                {"source":0,"target":"stack_spacing","minimum":.5,"maximum":3},
                {"source":1,"target":"stack_rotation","minimum":-5,"maximum":5},
                {"source":2,"target":"voxel_opacity","minimum":.2,"maximum":1}]}
        args["advanced_settings"] = json.dumps(settings)
        with patch.object(fx,"write_preview",return_value={}):
            first = fx.FL_InteractiveScanFX.execute(**args)
            second = fx.FL_InteractiveScanFX.execute(**args)
            args["advanced_settings"] = "{}"
            default = fx.FL_InteractiveScanFX.execute(**args)
        torch.testing.assert_close(first.args[0],second.args[0],rtol=0,atol=0)
        self.assertFalse(torch.equal(first.args[0],default.args[0]))
        self.assertTrue(torch.isfinite(first.args[0]).all())
        for invalid in ({"stack_count":2.5},{"stack_count":9},{"window_blend":"unknown"},{"window_fade_in":-1}):
            with self.assertRaises(ValueError):fx.settings_from_json(json.dumps(invalid))

    def test_dynamic_shots_accept_any_section_count(self):
        images = torch.rand(61, 8, 8, 3)
        for count in (1, 2, 4, 8, 13):
            schedule = {"sections": [{"start_frame": 61*i//count, "end_frame": 61*(i+1)//count} for i in range(count)]}
            chunks, = fx.FL_ScanVideoShots().split(images, schedule)
            self.assertEqual(len(chunks), count)
            torch.testing.assert_close(torch.cat(chunks), images)

    def test_dynamic_shot_list_preserves_groups_and_remainders(self):
        images = torch.rand(17, 8, 8, 3)
        schedule = {"sections": [
            {"start_frame": 0, "end_frame": 3},
            {"start_frame": 3, "end_frame": 6, "render_group": 1},
            {"start_frame": 6, "end_frame": 10, "render_group": 1},
            {"start_frame": 10, "end_frame": 17},
        ]}
        chunks, = fx.FL_ScanVideoShots().split(images, schedule)
        self.assertEqual([len(c) for c in chunks], [3, 7, 7])
        torch.testing.assert_close(torch.cat(chunks), images)
        self.assertNotEqual(chunks[0].data_ptr(), images.data_ptr())
        with self.assertRaisesRegex(ValueError, "length must match"):
            fx.FL_ScanVideoShots().split(torch.rand(18,8,8,3), schedule)

    def test_collected_analysis_matches_manual_connections(self):
        args = self.inputs()
        shot = args["analysis"]["shot0"]
        with patch.object(fx, "write_preview", return_value={}):
            expected = fx.FL_InteractiveScanFX.execute(**args)
            bundle, = fx.FL_ScanAnalysisCollect().collect([shot])
            args["analysis"] = {"shot0": bundle}
            actual = fx.FL_InteractiveScanFX.execute(**args)
        torch.testing.assert_close(actual.args[0], expected.args[0])

    def test_finishing_falls_back_to_cpu_with_low_gpu_headroom(self):
        args = self.inputs()
        with patch.object(fx, "write_preview", return_value={}), patch.object(fx.model_management, "get_torch_device", return_value=torch.device("cpu")):
            expected = fx.FL_InteractiveScanFX.execute(**args)
        with patch.object(fx, "write_preview", return_value={}), patch.object(fx.model_management, "get_torch_device", return_value=torch.device("cuda")), patch.object(fx.model_management, "get_free_memory", return_value=0):
            actual = fx.FL_InteractiveScanFX.execute(**args)
        torch.testing.assert_close(actual.args[0], expected.args[0], rtol=0, atol=0)

    def inputs(self):
        torch.manual_seed(5)
        images=torch.rand(12,32,32,3)
        depth=torch.rand_like(images)
        normals=torch.rand_like(images)
        shot,=fx.FL_ScanAnalysis().pack(depth,normals)
        envelope={"type":"fl_audio_envelope","version":1,"fps":24,"duration":.5,"total_frames":12,"values":[float(i%4==0) for i in range(12)]}
        return dict(images=images,analysis={"shot0":shot},kick_envelope=envelope,snare_envelope=envelope,hihat_envelope=envelope,
                    fps=24,cube_size=8,relief=.65,animation=.18,speed=.7,cursor_count=2,motion_strength=.6,seed=73,advanced_settings=json.dumps(fx.DEFAULTS))

    def test_matches_existing_effect_chain(self):
        args=self.inputs();s=fx.DEFAULTS;shot=args["analysis"]["shot0"];im=args["images"];original=im.clone();e=args["kick_envelope"]
        voxels,=fx.FL_VoxelNormalRelief().render(shot["normals"],shot["depth"],8,.65,.18,.7,24,41)
        tracks={"frames":[[] for _ in im],"height":32,"width":32}
        scan,_,surface=fx.FL_StreetScanComposite().render(im,shot["depth"],voxels,torch.zeros(12,32,32),tracks,24,41,5.5,1,.74,0,0,.7,0,None,.5,"digital_layers")
        expected,_,mask,report=fx.FL_ScanAudioEdit().render(im,scan,surface,e,e,e,"12",24,73,10,20,1.7,2,1.2,1,.6,"audio_locked")
        expected,=fx.FL_Audio_Reactive_Brightness().apply_brightness(expected,e,mask=mask[:,:,:,None].expand(-1,-1,-1,3),brightness_intensity=.16)
        expected,=fx.FL_Audio_Reactive_Saturation().apply_saturation(expected,e,base_saturation=.9,saturation_intensity=.3)
        expected,=fx.FL_Audio_Reactive_Edge_Glow().apply_edge_glow(expected,e,edge_threshold=.15,glow_intensity=0,envelope_intensity=.28,glow_color="white",blend_mode="screen")
        with patch.object(fx,"write_preview",return_value={}): result=fx.FL_InteractiveScanFX.execute(**args)
        torch.testing.assert_close(result.args[0],expected)
        torch.testing.assert_close(result.args[1],surface)
        torch.testing.assert_close(result.args[2],mask)
        torch.testing.assert_close(im,original)
        self.assertEqual(result.args[3],report)

    def test_preview_encodes_every_frame(self):
        args=self.inputs()
        with tempfile.TemporaryDirectory() as directory,patch.object(fx.folder_paths,"get_temp_directory",return_value=directory),patch.object(fx,"scan_progress") as progress:
            result=fx.FL_InteractiveScanFX.execute(**args)
            preview=result.ui["fl_interactive_scan"][0]
            with av.open(str(Path(directory)/preview["filename"])) as video:
                frames=list(video.decode(video=0))
            self.assertEqual(len(frames),12)
            self.assertEqual((frames[0].width,frames[0].height),(96,64))
            self.assertEqual(preview["envelopes"][0],args["kick_envelope"]["values"])
            stages=[call.args[0] for call in progress.call_args_list]
            self.assertEqual(list(dict.fromkeys(stages)),["Shot 1/1 · Voxel normals","Shot 1/1 · Depth projection",
                "Cursor reveals and audio edit","Color and glow","Encoding previews","Complete"])
            self.assertEqual(progress.call_args.args,("Complete",1,1,False))

    def test_invalid_timing_rejected_before_render(self):
        args=self.inputs();args["fps"]=30
        with self.assertRaisesRegex(ValueError,"FPS"):fx.FL_InteractiveScanFX.execute(**args)

    def test_analysis_lengths_rejected(self):
        args=self.inputs();args["images"]=args["images"][:8]
        with self.assertRaisesRegex(ValueError,"video has 8 frames, but analysis covers 12"):fx.FL_InteractiveScanFX.execute(**args)

    def test_dynamic_sections_cover_video_without_gaps_or_duplicates(self):
        for count in (4, 12, 192, 384, 385, 391):
            images=torch.arange(count).reshape(count,1,1,1)
            chunks=[fx.FL_ScanVideoSection().split(images,4,i)[0] for i in range(4)]
            torch.testing.assert_close(torch.cat(chunks),images)
            self.assertLessEqual(max(map(len,chunks))-min(map(len,chunks)),1)
            self.assertTrue(all(c.untyped_storage().data_ptr()!=images.untyped_storage().data_ptr() for c in chunks))

    def test_dynamic_sections_render_as_aligned_analysis(self):
        args=self.inputs();shot=args['analysis']['shot0'];args['analysis']={}
        for i in range(4):
            depth,=fx.FL_ScanVideoSection().split(shot['depth'],4,i)
            normals,=fx.FL_ScanVideoSection().split(shot['normals'],4,i)
            args['analysis'][f'shot{i}']=fx.FL_ScanAnalysis().pack(depth,normals)[0]
        with patch.object(fx,'write_preview',return_value={}):
            result=fx.FL_InteractiveScanFX.execute(**args)
        self.assertEqual(len(result.args[0]),len(args['images']))

    def test_settings_validation(self):
        for bad in ({"oops":1},{"scene_scale":-1},{"edge_threshold":1},{"surface_seed":.5},{"glow_color":"nope"}):
            with self.assertRaises(ValueError):fx.settings_from_json(json.dumps(bad))

    def test_reversed_cut_range_is_normalized_and_renders(self):
        args=self.inputs();args['advanced_settings']=json.dumps({'min_cut_frames':20,'max_cut_frames':3})
        settings=fx.settings_from_json(args['advanced_settings'])
        self.assertEqual((settings['min_cut_frames'],settings['max_cut_frames']),(3,20))
        with patch.object(fx,'write_preview',return_value={}):
            result=fx.FL_InteractiveScanFX.execute(**args)
        self.assertEqual(len(result.args[0]),12)

    def test_multiple_shots_preserve_authored_boundaries(self):
        args=self.inputs();shot=args["analysis"]["shot0"]
        a,=fx.FL_ScanAnalysis().pack(shot["depth"][:6],shot["normals"][:6])
        b,=fx.FL_ScanAnalysis().pack(shot["depth"][6:],shot["normals"][6:])
        args["analysis"]={"shot1":b,"shot0":a}
        with patch.object(fx,"write_preview",return_value={}):result=fx.FL_InteractiveScanFX.execute(**args)
        report=json.loads(result.args[3])
        self.assertEqual(report["source_indices"],list(range(12)))
        self.assertTrue(any(s["start_frame"]==6 and s["shot"]==2 for s in report["segments"]))

    def test_reveals_only_preserves_scene_without_cursors(self):
        args=self.inputs();args["cursor_count"]=0
        with patch.object(fx,"write_preview",return_value={}):
            original=fx.FL_InteractiveScanFX.execute(**args)
            args["advanced_settings"]=json.dumps(fx.DEFAULTS|{"motion_mode":"depth_parallax","parallax_scope":"reveals_only","offset_x":.15,"dolly":.2})
            reveal=fx.FL_InteractiveScanFX.execute(**args)
            args["advanced_settings"]=json.dumps(fx.DEFAULTS|{"motion_mode":"depth_parallax","offset_x":.15,"dolly":.2})
            whole=fx.FL_InteractiveScanFX.execute(**args)
        torch.testing.assert_close(original.args[0],reveal.args[0])
        self.assertFalse(torch.equal(original.args[1],reveal.args[1]))
        self.assertFalse(torch.equal(original.args[0],whole.args[0]))

    def test_depth_windows_and_mappings_execute(self):
        args=self.inputs()
        args["advanced_settings"]=json.dumps(fx.DEFAULTS|{"motion_mode":"depth_parallax","depth_style":"contours",
            "voxel_weight":0,"edge_weight":0,"depth_weight":100,
            "audio_mappings":[{"source":0,"target":"dolly","minimum":0,"maximum":.1,"start_frame":0,"end_frame":12}]})
        with patch.object(fx,"write_preview",return_value={}) as preview:result=fx.FL_InteractiveScanFX.execute(**args)
        report=json.loads(result.args[3])
        self.assertTrue(report["cursor_events"])
        self.assertTrue(all(e["effect"]==2 for e in report["cursor_events"]))
        self.assertTrue(result.args[2].any())
        self.assertEqual(preview.call_args.args[6].shape,(12,32,32))


if __name__=="__main__":unittest.main()
