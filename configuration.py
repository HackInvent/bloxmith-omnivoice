"""Pure configuration validation; model SDK imports belong only to the worker."""
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from .process import OmniVoiceError

DEFAULTS = {
    "asset_directory": "",
    "environment_python": "",
    "voice_mode": "auto",
    "language": "auto",
    "voice_instruction": "",
    "reference_audio": "",
    "reference_text": "",
    "device": "cuda:0",
    "precision": "float16",
    "format": "ogg_opus",
    "output_directory": "",
    "target_duration_sec": 0,
    "speed": 1,
    "num_step": 32,
    "guidance_scale": 2,
    "t_shift": 0.1,
    "position_temperature": 5,
    "class_temperature": 0,
    "layer_penalty_factor": 5,
    "denoise": True,
    "preprocess_prompt": True,
    "postprocess_output": True,
    "pad_duration": 0.1,
    "fade_duration": 0.1,
    "audio_chunk_duration": 15,
    "audio_chunk_threshold": 30,
    "threads": 2,
    "max_audio_sec": 60,
    "max_segments": 16,
    "max_context_tokens": 4096,
    "max_text_chars": 1000,
    "timeout_sec": 240,
    "memory_limit_mib": 0,
    "min_available_ram_mib": 8192,
    "max_pending": 4
}
LANGUAGES = json.loads(Path(__file__).with_name("languages.json").read_text(encoding="utf-8"))["languages"]
DESIGN_CATEGORIES = (
    {"male", "female", "男", "女"},
    {"child", "teenager", "young adult", "middle-aged", "elderly", "儿童", "少年", "青年", "中年", "老年"},
    {"very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch", "极低音调", "低音调", "中音调", "高音调", "极高音调"},
    {"whisper", "耳语"},
    {"american accent", "british accent", "australian accent", "canadian accent", "indian accent", "chinese accent", "korean accent", "japanese accent", "portuguese accent", "russian accent"},
    {"河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"},
)

def configuration(raw):
    if not isinstance(raw, Mapping) or set(raw)-set(DEFAULTS)-{"execution","position","runtime_path","runtime_path_label"}:
        raise OmniVoiceError("config")
    cfg={**DEFAULTS,**{k:v for k,v in raw.items() if k in DEFAULTS}}
    for key,limit,multiline in (("asset_directory",4096,False),("environment_python",4096,False),
            ("reference_audio",4096,False),("output_directory",4096,False),
            ("reference_text",4096,True),("voice_instruction",512,False)):
        value=cfg[key]
        try:
            if not isinstance(value,str) or len(value.encode("utf-8"))>limit or any(
                    ord(c)<32 and not (multiline and c in "\n\t\r") for c in value):
                raise OmniVoiceError("config")
        except UnicodeError:
            raise OmniVoiceError("config") from None
    for key,choices in (("device",("cpu","cuda:0")),("precision",("float32","float16","bfloat16")),
            ("format",("wav","ogg_opus")),("voice_mode",("auto","design","clone")),("language",("auto",*LANGUAGES))):
        if not isinstance(cfg[key],str) or cfg[key] not in choices: raise OmniVoiceError("config")
    if cfg["device"]=="cpu" and cfg["precision"]!="float32": raise OmniVoiceError("config")
    if cfg["voice_mode"]=="clone" and (not cfg["reference_audio"].strip() or not cfg["reference_text"].strip()):
        raise OmniVoiceError("config")
    if cfg["voice_mode"]=="design":
        items=[s.strip().lower() for s in re.split("[,，]",cfg["voice_instruction"]) if s.strip()]
        valid=set().union(*DESIGN_CATEGORIES)
        if not items or any(s not in valid for s in items) or any(
                sum(s in category for s in items)>1 for category in DESIGN_CATEGORIES):
            raise OmniVoiceError("config")
        if any(s in DESIGN_CATEGORIES[-1] for s in items) and any(s in DESIGN_CATEGORIES[-2] for s in items):
            raise OmniVoiceError("config")
    for key,low,high in (("num_step",1,64),("threads",1,16),("max_audio_sec",1,90),
            ("max_text_chars",1,2000),("timeout_sec",1,240),("memory_limit_mib",0,262144),
            ("min_available_ram_mib",4096,262144),("max_pending",0,16),("max_segments",1,16),("max_context_tokens",512,8192)):
        if type(cfg[key]) is not int or not low<=cfg[key]<=high: raise OmniVoiceError("config")
    if 0<cfg["memory_limit_mib"]<4096: raise OmniVoiceError("config")
    for key,low,high in (("target_duration_sec",0,90),("speed",.5,2),("guidance_scale",0,8),("t_shift",.01,1),
            ("position_temperature",0,10),("class_temperature",0,2),("layer_penalty_factor",0,10),
            ("pad_duration",0,.5),("fade_duration",0,.5),("audio_chunk_duration",3,30),("audio_chunk_threshold",3,30)):
        if type(cfg[key]) not in (int,float) or not math.isfinite(cfg[key]) or not low<=cfg[key]<=high:
            raise OmniVoiceError("config")
    for key in ("denoise","preprocess_prompt","postprocess_output"):
        if type(cfg[key]) is not bool: raise OmniVoiceError("config")
    if cfg["audio_chunk_duration"]>cfg["audio_chunk_threshold"] or cfg["max_audio_sec"]<=2*cfg["pad_duration"]:
        raise OmniVoiceError("config")
    if cfg["target_duration_sec"] and not .04<=cfg["target_duration_sec"]<=cfg["max_audio_sec"]-2*cfg["pad_duration"]:
        raise OmniVoiceError("config")
    return cfg
