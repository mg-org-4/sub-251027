from .film_stock_color import NODE_CLASS_MAPPINGS as FSC, NODE_DISPLAY_NAME_MAPPINGS as FSC_N
from .film_stock_bw import NODE_CLASS_MAPPINGS as FSB, NODE_DISPLAY_NAME_MAPPINGS as FSB_N
from .film_grain import NODE_CLASS_MAPPINGS as FG, NODE_DISPLAY_NAME_MAPPINGS as FG_N
from .halation import NODE_CLASS_MAPPINGS as HAL, NODE_DISPLAY_NAME_MAPPINGS as HAL_N
from .print_stock import NODE_CLASS_MAPPINGS as PS, NODE_DISPLAY_NAME_MAPPINGS as PS_N
from .cross_process import NODE_CLASS_MAPPINGS as CP, NODE_DISPLAY_NAME_MAPPINGS as CP_N
from .chromatic_aberration import NODE_CLASS_MAPPINGS as CA, NODE_DISPLAY_NAME_MAPPINGS as CA_N
from .vignette import NODE_CLASS_MAPPINGS as VIG, NODE_DISPLAY_NAME_MAPPINGS as VIG_N
from .lens_distortion import NODE_CLASS_MAPPINGS as LD, NODE_DISPLAY_NAME_MAPPINGS as LD_N
from .perspective_correct import NODE_CLASS_MAPPINGS as PC, NODE_DISPLAY_NAME_MAPPINGS as PC_N
from .lens_profile import NODE_CLASS_MAPPINGS as LP, NODE_DISPLAY_NAME_MAPPINGS as LP_N
from .white_balance import NODE_CLASS_MAPPINGS as WB, NODE_DISPLAY_NAME_MAPPINGS as WB_N
from .exposure_tone import NODE_CLASS_MAPPINGS as ET, NODE_DISPLAY_NAME_MAPPINGS as ET_N
from .hsl_selective import NODE_CLASS_MAPPINGS as HSL, NODE_DISPLAY_NAME_MAPPINGS as HSL_N
from .clarity_texture_dehaze import NODE_CLASS_MAPPINGS as CTD, NODE_DISPLAY_NAME_MAPPINGS as CTD_N
from .vibrance import NODE_CLASS_MAPPINGS as VIB, NODE_DISPLAY_NAME_MAPPINGS as VIB_N
from .sharpening_pro import NODE_CLASS_MAPPINGS as SP, NODE_DISPLAY_NAME_MAPPINGS as SP_N
from .noise_reduction import NODE_CLASS_MAPPINGS as NR, NODE_DISPLAY_NAME_MAPPINGS as NR_N
from .skin_tone_uniformity import NODE_CLASS_MAPPINGS as STU, NODE_DISPLAY_NAME_MAPPINGS as STU_N
from .color_qualifier import NODE_CLASS_MAPPINGS as CQ, NODE_DISPLAY_NAME_MAPPINGS as CQ_N
# Wave 3: Color Grading
from .tone_curve import NODE_CLASS_MAPPINGS as TC, NODE_DISPLAY_NAME_MAPPINGS as TC_N
from .lift_gamma_gain import NODE_CLASS_MAPPINGS as LGG, NODE_DISPLAY_NAME_MAPPINGS as LGG_N
from .log_wheels import NODE_CLASS_MAPPINGS as LW, NODE_DISPLAY_NAME_MAPPINGS as LW_N
from .three_way_color_balance import NODE_CLASS_MAPPINGS as TWCB, NODE_DISPLAY_NAME_MAPPINGS as TWCB_N
from .hue_vs_hue import NODE_CLASS_MAPPINGS as HVH, NODE_DISPLAY_NAME_MAPPINGS as HVH_N
from .hue_vs_sat import NODE_CLASS_MAPPINGS as HVS, NODE_DISPLAY_NAME_MAPPINGS as HVS_N
from .lum_vs_sat import NODE_CLASS_MAPPINGS as LVS, NODE_DISPLAY_NAME_MAPPINGS as LVS_N
from .sat_vs_sat import NODE_CLASS_MAPPINGS as SVS, NODE_DISPLAY_NAME_MAPPINGS as SVS_N
from .color_warper import NODE_CLASS_MAPPINGS as CW, NODE_DISPLAY_NAME_MAPPINGS as CW_N
# Wave 5: Pipeline — LUT & Color Management
from .lut_identity import NODE_CLASS_MAPPINGS as LI, NODE_DISPLAY_NAME_MAPPINGS as LI_N
from .lut_export import NODE_CLASS_MAPPINGS as LE, NODE_DISPLAY_NAME_MAPPINGS as LE_N
from .lut_apply import NODE_CLASS_MAPPINGS as LA, NODE_DISPLAY_NAME_MAPPINGS as LA_N
from .lut_bake_inject import NODE_CLASS_MAPPINGS as LBI, NODE_DISPLAY_NAME_MAPPINGS as LBI_N
from .lut_bake_extract import NODE_CLASS_MAPPINGS as LBE, NODE_DISPLAY_NAME_MAPPINGS as LBE_N
from .color_space_transform import NODE_CLASS_MAPPINGS as CST, NODE_DISPLAY_NAME_MAPPINGS as CST_N
from .aces_tonemap import NODE_CLASS_MAPPINGS as AT, NODE_DISPLAY_NAME_MAPPINGS as AT_N
# Wave 6: RAW Pipeline
from .raw_load import NODE_CLASS_MAPPINGS as RL, NODE_DISPLAY_NAME_MAPPINGS as RL_N
from .raw_metadata_split import NODE_CLASS_MAPPINGS as RMS, NODE_DISPLAY_NAME_MAPPINGS as RMS_N

NODE_CLASS_MAPPINGS = {**FSC, **FSB, **FG, **HAL, **PS, **CP, **CA, **VIG, **LD, **PC, **LP, **WB, **ET, **HSL, **CTD, **VIB, **SP, **NR, **STU, **CQ, **TC, **LGG, **LW, **TWCB, **HVH, **HVS, **LVS, **SVS, **CW, **LI, **LE, **LA, **LBI, **LBE, **CST, **AT, **RL, **RMS}
NODE_DISPLAY_NAME_MAPPINGS = {**FSC_N, **FSB_N, **FG_N, **HAL_N, **PS_N, **CP_N, **CA_N, **VIG_N, **LD_N, **PC_N, **LP_N, **WB_N, **ET_N, **HSL_N, **CTD_N, **VIB_N, **SP_N, **NR_N, **STU_N, **CQ_N, **TC_N, **LGG_N, **LW_N, **TWCB_N, **HVH_N, **HVS_N, **LVS_N, **SVS_N, **CW_N, **LI_N, **LE_N, **LA_N, **LBI_N, **LBE_N, **CST_N, **AT_N, **RL_N, **RMS_N}
