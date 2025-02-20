# from ..models.rome import ROMEHyperParams, apply_rome_to_model
# from ..models.memit import MEMITHyperParams, apply_memit_to_model
# from ..models.kn import KNHyperParams, apply_kn_to_model
# from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor, MendPerRewriteExecutor
# from ..models.ft import FTHyperParams, apply_ft_to_model
# from ..models.serac import SERACHparams, SeracRewriteExecutor, SeracMultimodalRewriteExecutor
from ..dataset import ZsreDataset, CounterFactDataset, CaptionDataset, VQADataset, PersonalityDataset, SafetyDataset
# from ..models.lora import LoRAHyperParams, apply_lora_to_model
# from ..models.grace import GraceHyperParams, apply_grace_to_model
# from ..models.melo import MELOHyperParams, apply_melo_to_model
# from ..models.wise import WISEHyperParams, apply_wise_to_model

# from ..models.baft import BaFTHyperParams, apply_baft_to_model
from ..models.roselora import RoseLoRAHyperParams, apply_roselora_to_model

ALG_DICT = {
    "RoseLoRA": apply_roselora_to_model
}


ALG_MULTIMODAL_DICT = {
    # "MEND": MendMultimodalRewriteExecutor().apply_to_model,
    # "SERAC": SeracMultimodalRewriteExecutor().apply_to_model,
    # "SERAC_MULTI": SeracMultimodalRewriteExecutor().apply_to_model,
}


PER_ALG_DICT = {
    # "MEND": MendPerRewriteExecutor().apply_to_model,
}


DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}


MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}


PER_DS_DICT = {
    "personalityEdit": PersonalityDataset
}


Safety_DS_DICT ={
    "safeEdit": SafetyDataset
}