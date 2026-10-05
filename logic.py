"""Bounded complete-file synthesis; publication is separate from cancellable native work."""

from collections.abc import Mapping
from dataclasses import dataclass
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from uuid import uuid4
import wave
from .process import OmniVoiceError, Cancelled, check, run_process

from .configuration import DEFAULTS, LANGUAGES, configuration


def json_value(raw, limit):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise OmniVoiceError("input")
            out[k]=v
        return out
    if not isinstance(raw,str): return raw
    try:
        if len(raw.encode("utf-8"))>limit: raise OmniVoiceError("input")
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(OmniVoiceError("input")))
    except (ValueError,UnicodeError,RecursionError):
        raise OmniVoiceError("input") from None


def interrupt(raw):
    value=json_value(raw,1024)
    if not isinstance(value,Mapping) or dict(value)!={"action":"interrupt"}:
        raise OmniVoiceError("input")
    return {"action":"interrupt"}


def request(raw, cfg):
    correlation=uuid4().hex
    # Plain strings remain speech, even if they look like commands. Correlated
    # JSON must arrive as a mapping, or be decoded according to its input MIME.
    if isinstance(raw,Mapping):
        if set(raw)!={"text","request_id"}: raise OmniVoiceError("input")
        correlation=raw["request_id"];raw=raw["text"]
        if not isinstance(correlation,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}",correlation):
            raise OmniVoiceError("input")
    try:
        if not isinstance(raw,str) or not raw.strip() or len(raw)>cfg["max_text_chars"] or len(raw.encode("utf-8"))>8192 or "\0" in raw:
            raise OmniVoiceError("input")
    except UnicodeError:
        raise OmniVoiceError("input") from None
    return {"text":raw.strip(),"request_id":correlation}


def resolve_path(root, value):
    path=Path(value).expanduser()
    return (Path(root)/path).absolute() if not path.is_absolute() else path


@dataclass
class PreparedAudio:
    owner: object
    path: Path
    output_directory: Path
    metadata: dict
    deadline: float

    def close(self):
        self.owner.cleanup()


def audio_info(path, cfg, raw):
    try:
        info=json_value(raw.decode("utf-8") if isinstance(raw,bytes) else raw,8192)
        if set(info)!={"complete","steps","audio_tokens","segments","sample_rate","frames","clipped_samples"} or info["complete"] is not True:
            raise ValueError()
        if type(info["steps"]) is not int or info["steps"] != cfg["num_step"]:
            raise ValueError()
        if type(info["audio_tokens"]) is not int or not 1<=info["audio_tokens"]<=25*cfg["max_audio_sec"]:
            raise ValueError()
        if type(info["frames"]) is not int or not 0<info["frames"]<=24000*cfg["max_audio_sec"] or type(info["sample_rate"]) is not int or info["sample_rate"]!=24000:
            raise ValueError()
        if type(info["segments"]) is not int or not 1<=info["segments"]<=cfg["max_segments"]:
            raise ValueError()
        if type(info["clipped_samples"]) is not int or not 0<=info["clipped_samples"]<=info["frames"]:
            raise ValueError()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            file=os.fstat(source.fileno())
            if not stat.S_ISREG(file.st_mode) or not 44<=file.st_size<=24000*cfg["max_audio_sec"]*2+128:
                raise ValueError()
            with wave.open(source,"rb") as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())!=(1,2,24000,info["frames"]):
                    raise ValueError()
                if len(wav.readframes(info["frames"]+1))!=info["frames"]*2:
                    raise ValueError()
        return info
    except (ValueError,TypeError,KeyError,OSError,EOFError,wave.Error):
        raise OmniVoiceError("audio_invalid") from None


def prepare_reference(cfg, root, job, deadline, cancel):
    """Optional owned reference, bounded after resampling; no implicit truncation."""
    if cfg["voice_mode"]!="clone":
        return None
    path=resolve_path(root,cfg["reference_audio"])
    if not cfg["reference_audio"] or path.suffix.lower() not in {".wav",".ogg",".webm",".flac",".mp3",".m4a",".mp4"}:
        raise OmniVoiceError("reference")
    snapshot=job/("reference-source"+path.suffix.lower())
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        with os.fdopen(fd,"rb") as source, snapshot.open("xb") as copy:
            os.chmod(snapshot,0o600)
            before=os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=33554432:
                raise OmniVoiceError("reference")
            digest=hashlib.sha256();total=0
            while chunk:=source.read(min(65536,33554433-total)):
                check(deadline,cancel);total+=len(chunk)
                if total>33554432: raise OmniVoiceError("reference")
                copy.write(chunk);digest.update(chunk)
            after=os.fstat(source.fileno())
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or total!=before.st_size:
                raise OmniVoiceError("reference")
        maximum=24000*20
        raw=run_process(["ffmpeg","-v","error","-nostdin","-threads","1","-protocol_whitelist","file,pipe",
            "-format_whitelist","wav,ogg,matroska,webm,flac,mp3,mov","-i",str(snapshot),"-map","0:a:0","-vn",
            "-ac","1","-af",f"aresample=24000,atrim=end_sample={maximum+1}","-ar","24000","-f","f32le","pipe:1"],
            directory=job,deadline=deadline,cancel=cancel,limit=(maximum+1)*4)
        samples=array.array("f");samples.frombytes(raw)
        if sys.byteorder!="little": samples.byteswap()
        if not 12000<=len(samples)<=maximum or any(not math.isfinite(x) or abs(x)>1.001 for x in samples):
            raise OmniVoiceError("reference")
        if max(abs(x) for x in samples)<1e-5: raise OmniVoiceError("reference")
        pcm=array.array("h",(round(max(-1,min(1,x))*32767) for x in samples))
        if sys.byteorder!="little": pcm.byteswap()
        with (job/"reference.wav").open("xb") as target:
            os.chmod(job/"reference.wav",0o600)
            with wave.open(target,"wb") as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm.tobytes())
        return {"sha256":digest.hexdigest(),"bytes":total,"duration_sec":len(samples)/24000}
    except OmniVoiceError as exc:
        if exc.code in {"timeout","cancelled"}: raise
        raise OmniVoiceError("reference") from None
    except (OSError,ValueError,wave.Error):
        raise OmniVoiceError("reference") from None


def generate(item, cfg, root, storage, cancel=lambda:False):
    cfg=configuration(cfg);item=request(item,cfg)
    storage=Path(storage)
    if not storage.is_absolute() or storage.is_symlink() or not storage.is_dir(): raise OmniVoiceError("storage")
    assets=resolve_path(root,cfg["asset_directory"])
    if not cfg["asset_directory"] or not assets.is_dir() or assets.is_symlink(): raise OmniVoiceError("assets_missing")
    python=resolve_path(root,cfg["environment_python"]) if cfg["environment_python"] else Path(sys.executable)
    if not python.is_file() or not os.access(python,os.X_OK): raise OmniVoiceError("dependency")
    target=resolve_path(root,cfg["output_directory"]) if cfg["output_directory"] else storage/"audio"
    if target.is_symlink() or target.exists() and not target.is_dir(): raise OmniVoiceError("storage")
    started=time.monotonic();deadline=started+cfg["timeout_sec"]
    owner=tempfile.TemporaryDirectory(prefix="omni-job-",dir=storage);job=Path(owner.name)
    try:
        reference=prepare_reference(cfg,root,job,deadline,cancel)
        envelope={"config":cfg,"request":item,"assets":str(assets),"job":str(job)}
        raw=run_process([str(python),"-I","-B",str(Path(__file__).with_name("worker.py"))],directory=job,
            deadline=deadline,cancel=cancel,limit=8192,stdin=json.dumps(envelope,ensure_ascii=True).encode())
        del envelope
        wav=job/"generated.wav";info=audio_info(wav,cfg,raw)
        output=wav
        if cfg["format"]=="ogg_opus":
            output=job/"generated.ogg"
            run_process(["ffmpeg","-v","error","-nostdin","-n","-threads","1","-protocol_whitelist","file,pipe",
                "-format_whitelist","wav","-i",str(wav),"-map_metadata","-1","-ac","1","-ar","48000",
                "-c:a","libopus","-b:a","96000","-f","ogg",str(output)],directory=job,deadline=deadline,cancel=cancel,limit=1024)
        check(deadline,cancel)
        info.update(request_id=item["request_id"],duration_sec=info["frames"]/24000,
            source_frames=info["frames"],
            format=cfg["format"],sample_rate=48000 if cfg["format"]=="ogg_opus" else 24000,channels=1,
            source_sample_rate=24000,elapsed_sec=round(time.monotonic()-started,4),
            model="k2-fsa/OmniVoice",model_revision="c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
            sdk_revision="08be0b4ccbac3e13e374e86fbfead4b4cac343e2",reference=reference,
            settings={k:cfg[k] for k in ("device","precision","language","voice_mode","target_duration_sec","speed","num_step",
                "guidance_scale","t_shift","position_temperature","class_temperature","layer_penalty_factor","denoise",
                "preprocess_prompt","postprocess_output","pad_duration","fade_duration","audio_chunk_duration",
                "audio_chunk_threshold","max_audio_sec","max_segments","max_context_tokens")})
        del info["frames"]
        return PreparedAudio(owner,output,target,info,deadline)
    except BaseException:
        owner.cleanup()
        raise


def publish(prepared, cancel=lambda:False):
    """Publish after the listener has drained pending interrupts; never overwrite a file."""
    check(prepared.deadline,cancel)
    target=prepared.output_directory
    if target.is_symlink(): raise OmniVoiceError("storage")
    target.mkdir(mode=0o700,parents=True,exist_ok=True)
    suffix=".ogg" if prepared.metadata["format"]=="ogg_opus" else ".wav"
    final=target/("omni-"+uuid4().hex+suffix)
    temporary=None
    try:
        fd=os.open(prepared.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            size=os.fstat(source.fileno())
            if not stat.S_ISREG(size.st_mode) or not 0<size.st_size<=33554432: raise OmniVoiceError("audio_invalid")
            descriptor,temporary=tempfile.mkstemp(prefix=".omni-publish-",dir=target)
            digest=hashlib.sha256();total=0
            with os.fdopen(descriptor,"wb") as output:
                while chunk:=source.read(65536):
                    check(prepared.deadline,cancel);total+=len(chunk)
                    if total>33554432: raise OmniVoiceError("audio_invalid")
                    output.write(chunk);digest.update(chunk)
                output.flush();os.fsync(output.fileno())
            if total!=size.st_size: raise OmniVoiceError("audio_invalid")
        check(prepared.deadline,cancel)
        os.link(temporary,final,follow_symlinks=False)
        metadata={**prepared.metadata,"path":str(final),"bytes":total,"sha256":digest.hexdigest(),
            "content_type":"audio/ogg" if suffix==".ogg" else "audio/wav"}
        return metadata
    except OSError:
        raise OmniVoiceError("storage") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
