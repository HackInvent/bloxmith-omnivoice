"""Owned offline OmniVoice adapter, pinned to one SDK/model interface."""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import resource
import sys
import types
import wave

if __package__:
    from .assets_store import AssetError, prepare
    from .configuration import configuration
    from .process import OmniVoiceError
else:
    package=types.ModuleType("omni_worker_owned")
    package.__path__=[str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__]=package
    from omni_worker_owned.assets_store import AssetError, prepare
    from omni_worker_owned.configuration import configuration
    from omni_worker_owned.process import OmniVoiceError

SDK_REVISION="08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
RATE, FRAME_RATE, CODEBOOKS, MASK = 24000, 25, 8, 1024


def available_ram():
    values=dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
    available=int(values["MemAvailable"].split()[0])*1024
    try:
        cap=Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cap!="max":
            used=int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available=min(available,max(0,int(cap)-used))
    except (OSError,ValueError):
        pass
    return available


def require_dependencies():
    from importlib.metadata import distribution, version
    if sys.version_info[:2]!=(3,12): raise OmniVoiceError("dependency")
    for name,expected in (("torch","2.8.0"),("torchaudio","2.8.0"),("transformers","5.3.0"),
            ("accelerate","1.10.1"),("omnivoice","0.2.1"),("numpy","2.2.6"),
            ("soundfile","0.13.1"),("librosa","0.11.0")):
        if version(name).split("+")[0]!=expected: raise OmniVoiceError("dependency")
    try:
        origin=json.loads(distribution("omnivoice").read_text("direct_url.json") or "{}")
        if origin.get("url","").lower().removesuffix(".git")!="https://github.com/k2-fsa/omnivoice" or origin.get("vcs_info",{}).get("commit_id")!=SDK_REVISION:
            raise ValueError()
    except (ValueError,TypeError,AttributeError):
        raise OmniVoiceError("dependency") from None


def resource_preflight(torch,cfg):
    # Weight inventory plus an explicit working margin, not a peak-memory promise.
    model_bytes=612577280*(4 if cfg["precision"]=="float32" else 2)
    codec_bytes=805665628
    required_ram=max(cfg["min_available_ram_mib"]*1048576,
        model_bytes+codec_bytes+2147483648 if cfg["device"]=="cpu" else 4294967296)
    if available_ram()<required_ram: raise OmniVoiceError("resources")
    if cfg["device"]=="cuda:0":
        if not torch.cuda.is_available() or not torch.cuda.device_count(): raise OmniVoiceError("device")
        if cfg["precision"]=="bfloat16" and not torch.cuda.is_bf16_supported(): raise OmniVoiceError("device")
        if torch.cuda.mem_get_info(0)[0]<model_bytes+codec_bytes+2147483648: raise OmniVoiceError("resources")


def load_model(assets,cfg):
    import torch
    from omnivoice import OmniVoice
    model=OmniVoice.from_pretrained(str(assets/"model"),device_map=cfg["device"],
        dtype=getattr(torch,cfg["precision"]),local_files_only=True,
        attn_implementation="sdpa",load_asr=False)
    if (model.sampling_rate!=RATE or model.audio_tokenizer.config.frame_rate!=FRAME_RATE
            or model.config.num_audio_codebook!=CODEBOOKS or model.config.audio_mask_id!=MASK
            or model.config.audio_vocab_size!=MASK+1):
        raise OmniVoiceError("assets_corrupt")
    return model.eval()


def guard_generation(model,cfg,chunk_text):
    """Observe the pinned native algorithm; reject incomplete/missing chunks before decoding.

    These wrappers are local to a one-request child. No SDK files, global model
    methods or framework contracts are patched. All native sampling stays native.
    """
    import torch
    originals={key:getattr(model,key) for key in (
        "_preprocess_all","_generate_chunked","_generate_iterative","_prepare_inference_inputs","_decode_and_post_process")}
    state={"expected":0,"segments":[],"audio_tokens":0,"decoded":False,"preprocessed":False}
    budget=int((cfg["max_audio_sec"]-2*cfg["pad_duration"])*FRAME_RATE)

    def targets(task):
        if task.batch_size!=1 or len(task.target_lens)!=1 or type(task.target_lens[0]) is not int or not 1<=task.target_lens[0]<=budget:
            raise OmniVoiceError("audio_limit")

    def preprocess(*args,**kwargs):
        if state["preprocessed"]: raise OmniVoiceError("model_failed")
        task=originals["_preprocess_all"](*args,**kwargs)
        targets(task);state["preprocessed"]=True;state["expected"]=1
        return task

    def chunked(task,generation):
        targets(task)
        avg=task.target_lens[0]/len(task.texts[0])
        size=int(generation.audio_chunk_duration*FRAME_RATE/avg)
        chunks=chunk_text(text=task.texts[0],chunk_len=size,min_chunk_len=3)
        if not isinstance(chunks,list) or not 1<=len(chunks)<=cfg["max_segments"] or any(not isinstance(s,str) or not s for s in chunks):
            raise OmniVoiceError("audio_limit")
        state["expected"]=len(chunks)
        result=originals["_generate_chunked"](task,generation)
        if not isinstance(result,list) or len(result)!=1 or not isinstance(result[0],list) or len(result[0])!=len(chunks):
            raise OmniVoiceError("audio_invalid")
        return result

    def inputs(*args,**kwargs):
        result=originals["_prepare_inference_inputs"](*args,**kwargs)
        ids=result["input_ids"]
        if ids.ndim!=3 or ids.shape[:2]!=(1,CODEBOOKS) or not 1<=ids.shape[-1]<=cfg["max_context_tokens"]:
            raise OmniVoiceError("audio_limit")
        return result

    def iterative(task,generation):
        targets(task)
        if (not state["preprocessed"] or len(state["segments"])>=min(state["expected"],cfg["max_segments"])
                or state["audio_tokens"]+task.target_lens[0]>budget or generation.num_step!=cfg["num_step"]):
            raise OmniVoiceError("audio_limit")
        forwards=[0]
        def observe(_module,_args,_result):
            forwards[0]+=1
            if forwards[0]>cfg["num_step"]: raise OmniVoiceError("audio_limit")
        handle=model.register_forward_hook(observe)
        try:
            result=originals["_generate_iterative"](task,generation)
        finally:
            handle.remove()
        if forwards[0]!=cfg["num_step"] or not isinstance(result,list) or len(result)!=1:
            raise OmniVoiceError("audio_invalid")
        codes=result[0]
        if (not isinstance(codes,torch.Tensor) or codes.dtype!=torch.long
                or tuple(codes.shape)!=(CODEBOOKS,task.target_lens[0])
                or bool(((codes<0)|(codes>=MASK)).any().item())):
            raise OmniVoiceError("audio_invalid")
        state["segments"].append(codes);state["audio_tokens"]+=task.target_lens[0]
        return result

    def decode(tokens,rms,generation):
        segments=tokens if isinstance(tokens,list) else [tokens]
        if (state["decoded"] or not state["preprocessed"] or len(segments)!=state["expected"]
                or len(segments)!=len(state["segments"]) or any(a is not b for a,b in zip(segments,state["segments"]))):
            raise OmniVoiceError("audio_invalid")
        result=originals["_decode_and_post_process"](tokens,rms,generation)
        state["decoded"]=True
        return result

    for name,method in (("_preprocess_all",preprocess),("_generate_chunked",chunked),
            ("_generate_iterative",iterative),("_prepare_inference_inputs",inputs),("_decode_and_post_process",decode)):
        setattr(model,name,method)
    return state


def generate_audio(model,text,cfg,job):
    import numpy as np
    from omnivoice import OmniVoiceGenerationConfig
    from omnivoice.utils.text import chunk_text_punctuation
    state=guard_generation(model,cfg,chunk_text_punctuation)
    generation=OmniVoiceGenerationConfig(**{key:cfg[key] for key in ("num_step","guidance_scale","t_shift",
        "position_temperature","class_temperature","layer_penalty_factor","denoise","preprocess_prompt",
        "postprocess_output","pad_duration","fade_duration","audio_chunk_duration","audio_chunk_threshold")})
    parameters=dict(text=text,language=None if cfg["language"]=="auto" else cfg["language"],
        speed=cfg["speed"],duration=cfg["target_duration_sec"] or None,
        generation_config=generation,normalize_text=False)
    if cfg["voice_mode"]=="clone":
        # Mandatory transcript prevents automatic Whisper loading or downloads.
        parameters.update(ref_audio=str(job/"reference.wav"),ref_text=cfg["reference_text"])
    elif cfg["voice_mode"]=="design":
        parameters["instruct"]=cfg["voice_instruction"]
    outputs=model.generate(**parameters)
    if not state["decoded"] or not isinstance(outputs,list) or len(outputs)!=1:
        raise OmniVoiceError("audio_invalid")
    samples=outputs[0]
    if (not isinstance(samples,np.ndarray) or samples.ndim!=1 or samples.dtype.kind!="f"
            or not 0<len(samples)<=cfg["max_audio_sec"]*RATE or not np.all(np.isfinite(samples))
            or np.max(np.abs(samples))>4):
        raise OmniVoiceError("audio_invalid")
    clipped=int(np.count_nonzero(np.abs(samples)>1))
    pcm=np.rint(np.clip(samples,-1,1)*32767).astype("<i2")
    if not np.any(pcm): raise OmniVoiceError("audio_invalid")
    path=job/"generated.wav"
    with path.open("xb") as target:
        os.chmod(path,0o600)
        with wave.open(target,"wb") as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(RATE);wav.writeframes(pcm.tobytes())
    return {"complete":True,"steps":cfg["num_step"],"audio_tokens":state["audio_tokens"],
        "segments":len(state["segments"]),"sample_rate":RATE,"frames":len(pcm),"clipped_samples":clipped}


def infer(item):
    cfg=configuration(item["config"])
    if cfg["memory_limit_mib"]:
        limit=cfg["memory_limit_mib"]*1048576
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cpu=cfg["timeout_sec"]*cfg["threads"]
    resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))
    require_dependencies()
    import torch
    torch.set_num_threads(cfg["threads"]);torch.set_num_interop_threads(1)
    resource_preflight(torch,cfg)
    assets=prepare(item["assets"])
    model=load_model(assets,cfg)
    with torch.inference_mode():
        return generate_audio(model,item["request"]["text"],cfg,Path(item["job"]))


def main():
    data=sys.stdin.buffer.read(32769)
    if len(data)>32768: raise OmniVoiceError("input")
    with redirect_stdout(sys.stderr):
        result=infer(json.loads(data))
    sys.stdout.write(json.dumps(result,ensure_ascii=True,allow_nan=False))


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        from importlib.metadata import PackageNotFoundError
        if isinstance(exc,OmniVoiceError): code=exc.code
        elif isinstance(exc,AssetError): code=str(exc)
        elif isinstance(exc,(ImportError,PackageNotFoundError)): code="dependency"
        elif isinstance(exc,MemoryError) or isinstance(exc,RuntimeError) and any(s in str(exc) for s in ("out of memory","DefaultCPUAllocator","Cannot allocate memory")):
            code="resources"
        else: code="model_failed"
        print(json.dumps({"error_code":code}));raise SystemExit(1) from None
