from comfy_api.latest import io
from typing import Union
import os
import torch

import math
import folder_paths
import copy
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

from .ad_settings import AnimateDiffSettings
from .adapter_cameractrl import CameraEntry
from .logger import logger
from .utils_model import get_available_motion_models, calculate_file_hash, strip_path, BIGMAX
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import (MotionModelGroup, MotionModelPatcher, get_mm_attachment, load_motion_module_gen2, inject_camera_encoder_into_model)
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode


class CameraMotion:
    def __init__(self, rotate: tuple[float], translate: tuple[float]):
        assert len(rotate) == 3
        assert len(translate) == 3
        self.rotate = np.array(rotate)
        self.translate = np.array(translate)

    def multiply(self, mult: float):
        if math.isclose(mult, 1.0):
            return self.clone()
        new_rotate = self.rotate.copy()
        new_translate = self.translate.copy()
        new_rotate *= mult
        new_translate *= mult
        return CameraMotion(rotate=new_rotate, translate=new_translate)

    def clone(self):
        return CameraMotion(rotate=self.rotate.copy(), translate=self.translate.copy())

    @staticmethod
    def combine(deltas: list['CameraMotion']) -> 'CameraMotion':
        new_rotate = np.array([0., 0., 0.])
        new_translate = np.array([0., 0., 0.])
        for delta in deltas:
            new_rotate += delta.rotate
            new_translate += delta.translate
        return CameraMotion(rotate=new_rotate, translate=new_translate)


class CAM:
    BASE_T_NORM = 1.5
    BASE_ANGLE = np.pi/3

    DEFAULT_FX = 0.474812461
    DEFAULT_FY = 0.844111024
    DEFAULT_CX = 0.5
    DEFAULT_CY = 0.5

    DEFAULT_POSE_WIDTH = 1280
    DEFAULT_POSE_HEIGHT = 720

    STATIC = "Static"
    PAN_UP = "Pan Up"
    PAN_DOWN = "Pan Down"
    PAN_LEFT = "Pan Left"
    PAN_RIGHT = "Pan Right"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    ROLL_CLOCKWISE = "Roll Clockwise"
    ROLL_ANTICLOCKWISE = "Roll Anticlockwise"
    TILT_UP = "Tilt Up"
    TILT_DOWN = "Tilt Down"
    TILT_LEFT = "Tilt Left"
    TILT_RIGHT = "Tilt Right"
    
    _PAIRS = [
        (STATIC,        CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., 0.))),
        (PAN_UP,        CameraMotion(rotate=(0., 0., 0.), translate=(0., 1., 0.))),
        (PAN_DOWN,      CameraMotion(rotate=(0., 0., 0.), translate=(0., -1., 0.))),
        (PAN_LEFT,      CameraMotion(rotate=(0., 0., 0.), translate=(1., 0., 0.))),
        (PAN_RIGHT,     CameraMotion(rotate=(0., 0., 0.), translate=(-1., 0., 0.))),
        (ZOOM_IN,       CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., -2.))),
        (ZOOM_OUT,      CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., 2.))),
        (ROLL_CLOCKWISE,     CameraMotion(rotate=(0., 0., -1.), translate=(0., 0., 0.))),
        (ROLL_ANTICLOCKWISE, CameraMotion(rotate=(0., 0., 1.), translate=(0., 0., 0.))),
        (TILT_DOWN,     CameraMotion(rotate=(1., 0., 0.), translate=(0., 0., 0.))),
        (TILT_UP,    CameraMotion(rotate=(-1., 0., 0.), translate=(0., 0., 0.))),
        (TILT_LEFT,       CameraMotion(rotate=(0., 1., 0.), translate=(0., 0., 0.))),
        (TILT_RIGHT,     CameraMotion(rotate=(0., -1., 0.), translate=(0., 0., 0.))),
    ]
    _DICT: dict[str, CameraMotion] = OrderedDict(_PAIRS)
    _LIST = list(_DICT.keys())

    @staticmethod
    def get(motion: str):
        return CAM._DICT[motion]


def compute_R_from_rad_angle(angles: np.ndarray):
    theta_x, theta_y, theta_z = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def get_camera_motion(angle: np.ndarray, T: np.ndarray, speed: float, n=16, base=16):
    RT = []
    for i in range(n):
        _angle = (i/base)*speed*(CAM.BASE_ANGLE)*angle
        R = compute_R_from_rad_angle(_angle) 
        # _T = (i/n)*speed*(T.reshape(3,1))
        _T=(i/base)*speed*(CAM.BASE_T_NORM)*(T.reshape(3,1))
        _RT = np.concatenate([R,_T], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    return RT
    
def combine_RTs(RT_0: np.ndarray, RT_1: np.ndarray):
    RT = copy.deepcopy(RT_0[-1])
    R = RT[:,:3]
    R_inv = RT[:,:3].T
    T =  RT[:,-1]

    temp = []
    for _RT in RT_1:
        _RT[:,:3] = np.dot(_RT[:,:3], R)
        _RT[:,-1] =  _RT[:,-1] + np.dot(np.dot(_RT[:,:3], R_inv), T) 
        temp.append(_RT)

    RT_1 = np.stack(temp)

    return np.concatenate([RT_0, RT_1], axis=0)

def stack_RTs(RT_0: np.ndarray, RT_1: np.ndarray):
    RT_target = copy.deepcopy(RT_1)
    static_motion = CAM.get(CAM.STATIC)
    RT_static = get_camera_motion(static_motion.rotate, static_motion.translate, 1.0, 1)
    RT_offset = RT_0[-1] - RT_static[-1]

    temp = []
    for sub_RT in RT_target:
        temp.append(sub_RT + RT_offset)

    RT_1 = np.stack(temp)
    RT_0 = RT_0[:-1]

    return np.concatenate([RT_0, RT_1], axis=0)


def set_original_pose_dims(poses: list[list[float]], pose_width, pose_height):
    # indexes 5 and 6 are not used for anything in the poses, so can use 5 and 6 to set original pose width/height
    new_poses = copy.deepcopy(poses)
    for pose in new_poses:
        pose[5] = pose_width
        pose[6] = pose_height
    return new_poses

def combine_poses(poses0: list[list[float]], poses1: list[list[float]]):
    new_poses = copy.deepcopy(poses0) + copy.deepcopy(poses1)
    new_RT = combine_RTs(poses_to_ndarray(poses0), poses_to_ndarray(poses1))
    inter_poses = ndarray_to_poses(new_RT)
    # maintain fx, fy, cx, and cy values by pasting only the movement portion of poses
    for i in range(len(new_poses)):
        new_poses[i][7:] = inter_poses[i][7:]
    return new_poses


def combine_poses_redux(poses0: list[list[float]], poses1: list[list[float]]):
    new_poses = copy.deepcopy(poses0[:-1]) + copy.deepcopy(poses1)
    new_RT = stack_RTs(poses_to_ndarray(poses0), poses_to_ndarray(poses1))
    inter_poses = ndarray_to_poses(new_RT)
    # maintain fx, fy, cx, and cy values by pasting only the movement portion of poses
    for i in range(len(new_poses)):
        new_poses[i][7:] = inter_poses[i][7:]
    return new_poses


def combine_poses_with_ndarray(poses: list[list[float]], RT: np.ndarray):
    return combine_poses(poses0=poses, poses1=ndarray_to_poses(RT))


def ndarray_to_poses(RT: np.ndarray, fx=CAM.DEFAULT_FX, fy=CAM.DEFAULT_FY, cx=CAM.DEFAULT_CX, cy=CAM.DEFAULT_CY) -> list[list[float]]:
    '''
    Converts ndarray (motion) to cameractrl_poses.
    '''
    motion_list=RT.tolist()
    poses = []
    for motion in motion_list:
        traj = [0, fx, fy, cx, cy, CAM.DEFAULT_POSE_WIDTH, CAM.DEFAULT_POSE_HEIGHT]
        traj.extend(motion[0])
        traj.extend(motion[1])
        traj.extend(motion[2])
        poses.append(traj)
    return poses

def poses_to_ndarray(poses: list[list[float]]) -> np.ndarray:
    '''
    Converts cameractrl_poses (list) to ndarray (motion) to be used for math stuff.
    '''
    motion_list = []
    for pose in poses:
        # pose will have 19 components;
        # idx 7-10 have first column, idx 11-14 have second column, idx 15-18 have third column
        motion_list.append(np.array(pose[7:]).reshape(3, 4))
    RT = np.array(motion_list)
    return RT


class ApplyAnimateDiffWithCameraCtrl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ApplyAnimateDiffModelWithCameraCtrl',
            display_name='Apply AnimateDiff+CameraCtrl Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl',
            inputs=[
                io.Custom("MOTION_MODEL_ADE").Input('motion_model'),
                io.Custom("CAMERACTRL_POSES").Input('cameractrl_poses'),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Custom("MOTION_LORA").Input('motion_lora', optional=True),
                io.Custom("MULTIVAL").Input('scale_multival', optional=True),
                io.Custom("MULTIVAL").Input('effect_multival', optional=True),
                io.Custom("MULTIVAL").Input('cameractrl_multival', optional=True),
                io.Custom("AD_KEYFRAMES").Input('ad_keyframes', optional=True),
                io.Custom("M_MODELS").Input('prev_m_models', optional=True),
                io.Custom("PER_BLOCK").Input('per_block', optional=True),
            ],
            outputs=[
                io.Custom("M_MODELS").Output('M_MODELS'),
            ],
        )
    

    @classmethod
    def execute(cls, motion_model: MotionModelPatcher, cameractrl_poses: list[list[float]], start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, cameractrl_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.execute( motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes, per_block=per_block,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models).args
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model contains camera_encoder
        if curr_model.model.camera_encoder is None:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' does not contain a camera_encoder; cannot be used with Apply AnimateDiff-CameraCtrl Model node.")
        camera_entries = [CameraEntry(entry) for entry in cameractrl_poses]
        attachment = get_mm_attachment(curr_model)
        attachment.orig_camera_entries = camera_entries
        attachment.cameractrl_multival = cameractrl_multival
        return io.NodeOutput(*new_m_models)


class LoadAnimateDiffModelWithCameraCtrl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_LoadAnimateDiffModelWithCameraCtrl',
            display_name='Load AnimateDiff+CameraCtrl Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl',
            inputs=[
                io.Combo.Input('model_name', options=get_available_motion_models()),
                io.Combo.Input('camera_ctrl', options=get_available_motion_models()),
                io.Custom("AD_SETTINGS").Input('ad_settings', optional=True),
            ],
            outputs=[
                io.Custom("MOTION_MODEL_ADE").Output('MOTION_MODEL'),
            ],
        )


    @classmethod
    def execute(cls, model_name: str, camera_ctrl: str, ad_settings: AnimateDiffSettings=None):
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_camera_encoder_into_model(motion_model=loaded_motion_model, camera_ctrl_name=camera_ctrl)
        return io.NodeOutput(loaded_motion_model,)


class CameraCtrlADKeyframeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CameraCtrlAnimateDiffKeyframe',
            display_name='AnimateDiff+CameraCtrl Keyframe 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl',
            inputs=[
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Custom("AD_KEYFRAMES").Input('prev_ad_keyframes', optional=True),
                io.Custom("MULTIVAL").Input('scale_multival', optional=True),
                io.Custom("MULTIVAL").Input('effect_multival', optional=True),
                io.Custom("MULTIVAL").Input('cameractrl_multival', optional=True),
                io.Boolean.Input('inherit_missing', optional=True, default=True),
                io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0),
            ],
            outputs=[
                io.Custom("AD_KEYFRAMES").Output('AD_KEYFRAMES'),
            ],
        )
    


    @classmethod
    def execute(cls,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      cameractrl_multival: Union[float, torch.Tensor]=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        return io.NodeOutput(*ADKeyframeNode.execute(
                    start_percent=start_percent, prev_ad_keyframes=prev_ad_keyframes,
                    scale_multival=scale_multival, effect_multival=effect_multival, cameractrl_multival=cameractrl_multival,
                    inherit_missing=inherit_missing, guarantee_steps=guarantee_steps
                ).args)


class LoadCameraPosesFromFile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_LoadCameraPoses',
            display_name='Load CameraCtrl Poses (File) 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Combo.Input('pose_filename', options=sorted(f for f in os.listdir(folder_paths.get_input_directory()) if os.path.isfile(os.path.join(folder_paths.get_input_directory(), f)) and f.endswith(".txt"))),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )


    @classmethod
    def execute(cls, pose_filename: str):
        file_path = folder_paths.get_annotated_filepath(pose_filename)
        with open(file_path, 'r') as f:
            poses = f.readlines()
        # first line of file is the link to source, so can be skipped,
        # and the rest is a header-less CSV file separated by single spaces
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        poses = [[float(x) for x in pose] for pose in poses]
        poses = set_original_pose_dims(poses, pose_width=CAM.DEFAULT_POSE_WIDTH, pose_height=CAM.DEFAULT_POSE_HEIGHT)
        return io.NodeOutput(poses,)
    

class LoadCameraPosesFromPath(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_LoadCameraPosesFromPath',
            display_name='Load CameraCtrl Poses (Path) 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.String.Input('file_path', optional=True, default='X://path/to/pose_file.txt'),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )
    
    @classmethod
    def fingerprint_inputs(cls, file_path, **kwargs):
        if Path(file_path).is_file():
            return calculate_file_hash(strip_path(file_path))
        return False
    
    @classmethod
    def validate_inputs(cls, file_path, **kwargs):
        # This function never gets ran for some reason, I don't care enough to figure out why right now.
        if not Path(strip_path(file_path)).is_file():
            return f"Pose file not found: {file_path}"
        return True


    @classmethod
    def execute(cls, file_path: str):
        file_path = strip_path(file_path)
        if not Path(file_path).is_file():
            raise Exception(f"Pose file not found: {file_path}")
        with open(file_path, 'r') as f:
            poses = f.readlines()
        # first line of file is the link to source, so can be skipped,
        # and the rest is a header-less CSV file separated by single spaces
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        poses = [[float(x) for x in pose] for pose in poses]
        poses = set_original_pose_dims(poses, pose_width=CAM.DEFAULT_POSE_WIDTH, pose_height=CAM.DEFAULT_POSE_HEIGHT)
        return io.NodeOutput(poses,)


class CameraCtrlPoseBasic(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CameraPoseBasic',
            display_name='Create CameraCtrl Poses 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Combo.Input('motion_type', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('speed', default=1.0, max=100.0, min=-100.0, step=0.01),
                io.Int.Input('frame_length', default=16),
                io.Custom("CAMERACTRL_POSES").Input('prev_poses', optional=True),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )


    @classmethod
    def execute(cls, motion_type: str, speed: float, frame_length: int, prev_poses: list[list[float]]=None):
        motion = CAM.get(motion_type)
        RT = get_camera_motion(motion.rotate, motion.translate, speed, frame_length)
        new_motion = ndarray_to_poses(RT=RT)
        if prev_poses is not None:
            new_motion = combine_poses(prev_poses, new_motion)
        return io.NodeOutput(new_motion,)


class CameraCtrlPoseCombo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CameraPoseCombo',
            display_name='Create CameraCtrl Poses (Combo) 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Combo.Input('motion_type1', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Combo.Input('motion_type2', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Combo.Input('motion_type3', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Combo.Input('motion_type4', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Combo.Input('motion_type5', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Combo.Input('motion_type6', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('speed', default=1.0, max=100.0, min=-100.0, step=0.01),
                io.Int.Input('frame_length', default=16),
                io.Custom("CAMERACTRL_POSES").Input('prev_poses', optional=True),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )


    @classmethod
    def execute(cls,
                          motion_type1: str, motion_type2: str, motion_type3: str,
                          motion_type4: str, motion_type5: str, motion_type6: str,
                          speed: float, frame_length: int,
                          prev_poses: list[list[float]]=None,
                          strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0, strength5=1.0, strength6=1.0):
        combined_motion = CameraMotion.combine([
            CAM.get(motion_type1).multiply(strength1), CAM.get(motion_type2).multiply(strength2), CAM.get(motion_type3).multiply(strength3),
            CAM.get(motion_type4).multiply(strength4), CAM.get(motion_type5).multiply(strength5), CAM.get(motion_type6).multiply(strength6)
            ])
        RT = get_camera_motion(combined_motion.rotate, combined_motion.translate, speed, frame_length)
        new_motion = ndarray_to_poses(RT=RT)
        if prev_poses is not None:
            new_motion = combine_poses(prev_poses, new_motion)
        return io.NodeOutput(new_motion,)


class CameraCtrlPoseAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CameraPoseAdvanced',
            display_name='Create CameraCtrl Poses (Adv.) 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Combo.Input('motion_type1', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('strength1', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Combo.Input('motion_type2', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('strength2', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Combo.Input('motion_type3', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('strength3', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Combo.Input('motion_type4', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('strength4', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Combo.Input('motion_type5', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('strength5', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Combo.Input('motion_type6', options=['Static', 'Pan Up', 'Pan Down', 'Pan Left', 'Pan Right', 'Zoom In', 'Zoom Out', 'Roll Clockwise', 'Roll Anticlockwise', 'Tilt Down', 'Tilt Up', 'Tilt Left', 'Tilt Right']),
                io.Float.Input('strength6', default=1.0, max=10.0, min=0.0, step=0.01),
                io.Float.Input('speed', default=1.0, max=100.0, min=-100.0, step=0.01),
                io.Int.Input('frame_length', default=16),
                io.Custom("CAMERACTRL_POSES").Input('prev_poses', optional=True),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )


    @classmethod
    def execute(cls,
                          motion_type1: str, motion_type2: str, motion_type3: str,
                          motion_type4: str, motion_type5: str, motion_type6: str,
                          speed: float, frame_length: int,
                          prev_poses: list[list[float]]=None,
                          strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0, strength5=1.0, strength6=1.0):
        return io.NodeOutput(*CameraCtrlPoseCombo.execute(
                                                     motion_type1=motion_type1, motion_type2=motion_type2, motion_type3=motion_type3,
                                                     motion_type4=motion_type4, motion_type5=motion_type5, motion_type6=motion_type6,
                                                     speed=speed, frame_length=frame_length, prev_poses=prev_poses,
                                                     strength1=strength1, strength2=strength2, strength3=strength3,
                                                     strength4=strength4, strength5=strength5, strength6=strength6).args)


class CameraCtrlManualAppendPose(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CameraManualPoseAppend',
            display_name='Manual Append CameraCtrl Poses 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Custom("CAMERACTRL_POSES").Input('poses_first'),
                io.Custom("CAMERACTRL_POSES").Input('poses_last'),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )
    

    @classmethod
    def execute(cls, poses_first: list[list[float]], poses_last: list[list[float]]):
        return io.NodeOutput(combine_poses(poses0=poses_first, poses1=poses_last),)


class CameraCtrlReplaceCameraParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ReplaceCameraParameters',
            display_name='Replace Camera Parameters 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Custom("CAMERACTRL_POSES").Input('poses'),
                io.Float.Input('fx', default=0.474812461, max=1, min=0, step=1e-09),
                io.Float.Input('fy', default=0.844111024, max=1, min=0, step=1e-09),
                io.Float.Input('cx', default=0.5, max=1, min=0, step=0.01),
                io.Float.Input('cy', default=0.5, max=1, min=0, step=0.01),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )
    

    @classmethod
    def execute(cls, poses: list[list[float]], fx: float, fy: float, cx: float, cy: float):
        new_poses = copy.deepcopy(poses)
        for pose in new_poses:
            # fx,fy,cx,fy are in indexes 1-4 of the 19-long pose list
            pose[1] = fx
            pose[2] = fy
            pose[3] = cx
            pose[4] = cy
        return io.NodeOutput(new_poses,)


class CameraCtrlSetOriginalAspectRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ReplaceOriginalPoseAspectRatio',
            display_name='Replace Orig. Pose Aspect Ratio 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/CameraCtrl/poses',
            inputs=[
                io.Custom("CAMERACTRL_POSES").Input('poses'),
                io.Int.Input('orig_pose_width', default=1280, max=9007199254740991, min=1),
                io.Int.Input('orig_pose_height', default=720, max=9007199254740991, min=1),
            ],
            outputs=[
                io.Custom("CAMERACTRL_POSES").Output('CAMERACTRL_POSES'),
            ],
        )
    

    @classmethod
    def execute(cls, poses: list[list[float]], orig_pose_width: int, orig_pose_height: int):
        return io.NodeOutput(set_original_pose_dims(poses, pose_width=orig_pose_width, pose_height=orig_pose_height),)
