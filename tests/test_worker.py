"""Explicit lightweight SDK/model doubles, never a substitute for real-model qualification."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import wave
import numpy as np
from blocs.omnivoice import logic, worker
from omni_fixture import settings

class Tensor:
    def __init__(self,data): self.data=np.asarray(data)
    @property
    def ndim(self): return self.data.ndim
    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    def __lt__(self,v): return Tensor(self.data<v)
    def __ge__(self,v): return Tensor(self.data>=v)
    def __or__(self,v): return Tensor(self.data|v.data)
    def any(self): return Tensor(self.data.any())
    def item(self): return self.data.item()

TORCH=NS(Tensor=Tensor,long=np.dtype("int64"))
def chunks(**kwargs): return ["first", "second"]

class Model:
    def __init__(self,long=False,change=None):
        self.long=long;self.change=change;self.hooks=[];self.decoded=0;self.index=0
        self.config=NS(num_audio_codebook=8,audio_mask_id=1024,audio_vocab_size=1025)
        self.audio_tokenizer=NS(config=NS(frame_rate=25));self.sampling_rate=24000
    def _preprocess_all(self,**kw):
        return NS(batch_size=1,target_lens=[800 if self.long else 13],texts=[kw["text"]])
    def _generate_chunked(self,task,gen):
        result=[]
        for text in chunks():
            result.extend(self._generate_iterative(NS(batch_size=1,target_lens=[13],texts=[text]),gen))
        if self.change=="missing": result.pop()
        if self.change=="reorder": result.reverse()
        return [result]
    def _prepare_inference_inputs(self,*args,**kwargs):
        return {"input_ids":Tensor(np.ones((1,8,9000 if self.change=="context" else 40),dtype=np.int64))}
    def register_forward_hook(self,callback):
        self.hooks.append(callback)
        return NS(remove=lambda:self.hooks.remove(callback))
    def _generate_iterative(self,task,gen):
        self._prepare_inference_inputs()
        for _ in range(gen.num_step-(1 if self.change=="steps" else 0)):
            for hook in list(self.hooks): hook(self,(),None)
        self.index+=1
        result=Tensor(np.full((8,task.target_lens[0]),self.index,dtype=np.int64))
        if self.change=="mask": result.data[0,0]=1024
        if self.change=="negative": result.data[0,0]=-1
        if self.change=="float": result=Tensor(result.data.astype(float))
        if self.change=="shape": result=Tensor(np.zeros((7,13),dtype=np.int64))
        return [result]
    def _decode_and_post_process(self,tokens,rms,gen):
        self.decoded+=1
        items=tokens if isinstance(tokens,list) else [tokens]
        return np.concatenate([np.full(12000,.2 if t.data[0,0]==1 else -.3,dtype=np.float32) for t in items])
    def generate(self,**kwargs):
        self.arguments=kwargs;gen=kwargs["generation_config"];task=self._preprocess_all(text=kwargs["text"])
        outputs=self._generate_chunked(task,gen) if self.long else self._generate_iterative(task,gen)
        return [self._decode_and_post_process(outputs[0],None,gen)]

def modules():
    return {"torch":TORCH,"omnivoice":NS(OmniVoiceGenerationConfig=lambda **kw:NS(**kw)),
        "omnivoice.utils.text":NS(chunk_text_punctuation=chunks)}

class WorkerTests(unittest.TestCase):
    def test_configuration_and_all_numeric_bounds(self):
        package=Path(__file__).parents[1]
        fields=json.loads((package/"fields.json").read_text())
        self.assertEqual({f["key"] for f in fields},set(logic.DEFAULTS))
        self.assertEqual(json.loads((package/"model.json").read_text())["config"],logic.DEFAULTS)
        self.assertEqual(len(logic.LANGUAGES),652)
        for field in fields:
            if field["type"] in {"integer","number"}:
                for value in (True,None,"1",float("nan"),field["min"]-1,field["max"]+1):
                    with self.subTest(key=field["key"],value=value),self.assertRaises(logic.OmniVoiceError):
                        logic.configuration({field["key"]:value})
        for cfg in ({"voice_mode":"clone"},{"language":"fake"},{"device":"cpu"},{"denoise":"true"},
                {"reference_text":"\ud800"},{"memory_limit_mib":4095},{"target_duration_sec":.01},
                {"target_duration_sec":2,"max_audio_sec":2},{"audio_chunk_duration":20,"audio_chunk_threshold":10}):
            with self.subTest(cfg=cfg),self.assertRaises(logic.OmniVoiceError): logic.configuration(cfg)

    def test_design_attributes_conflicts_and_modes(self):
        for value in ("male, female","unknown","female，女","british accent, 四川话",""):
            with self.subTest(value=value),self.assertRaises(logic.OmniVoiceError):
                logic.configuration({"voice_mode":"design","voice_instruction":value})
        self.assertEqual(logic.configuration({"voice_mode":"design","voice_instruction":"Female, low pitch, british accent"})["voice_mode"],"design")
        self.assertEqual(logic.configuration({"voice_mode":"design","voice_instruction":"男，四川话"})["voice_mode"],"design")
        self.assertEqual(logic.configuration({"voice_instruction":"an inactive draft"})["voice_mode"],"auto")

    def test_all_modes_native_arguments_and_complete_audio(self):
        for mode in ("auto","design","clone"):
            cfg=logic.configuration({"voice_mode":mode,"language":"fr","reference_audio":"selected.wav",
                "reference_text":"Reference words.","voice_instruction":"female","target_duration_sec":2})
            with TemporaryDirectory() as directory,patch.dict(sys.modules,modules()):
                job=Path(directory);model=Model();info=worker.generate_audio(model,"Bonjour.",cfg,job)
                self.assertEqual(logic.audio_info(job/"generated.wav",cfg,json.dumps(info)),info)
                self.assertEqual(info["steps"],32);self.assertEqual(info["audio_tokens"],13)
                self.assertEqual(model.arguments["language"],"fr")
                self.assertFalse(model.arguments["normalize_text"])
                self.assertEqual(model.arguments["duration"],2)
                self.assertEqual("ref_audio" in model.arguments,mode=="clone")
                self.assertEqual("instruct" in model.arguments,mode=="design")
                if mode=="clone": self.assertEqual(model.arguments["ref_text"],"Reference words.")
                if mode=="design": self.assertEqual(model.arguments["instruct"],"female")
                self.assertEqual(model.arguments["generation_config"].num_step,32)
                self.assertFalse(model.hooks)

    def test_long_form_preserves_all_chunks_in_order(self):
        with TemporaryDirectory() as directory,patch.dict(sys.modules,modules()):
            job=Path(directory);model=Model(long=True)
            info=worker.generate_audio(model,"First. Second.",logic.DEFAULTS,job)
            self.assertEqual(info["segments"],2);self.assertEqual(info["audio_tokens"],26)
            with wave.open(str(job/"generated.wav")) as wav:
                data=np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2")
            self.assertTrue(np.all(data[:12000]>0));self.assertTrue(np.all(data[12000:]<0))

    def test_incomplete_code_or_chunk_never_decodes(self):
        for change in ("mask","negative","float","shape","steps","context","missing","reorder"):
            with TemporaryDirectory() as directory,patch.dict(sys.modules,modules()),self.subTest(change=change):
                model=Model(long=True,change=change)
                with self.assertRaises(logic.OmniVoiceError):
                    worker.generate_audio(model,"First. Second.",logic.DEFAULTS,Path(directory))
                self.assertEqual(model.decoded,0)
                self.assertFalse((Path(directory)/"generated.wav").exists());self.assertFalse(model.hooks)

    def test_budgets_before_generation(self):
        for cfg in ({"max_audio_sec":1},{"max_segments":1}):
            with TemporaryDirectory() as directory,patch.dict(sys.modules,modules()):
                model=Model(long=True)
                with self.assertRaises(logic.OmniVoiceError):
                    worker.generate_audio(model,"First. Second.",logic.configuration(cfg),Path(directory))
                self.assertEqual(model.index,0);self.assertEqual(model.decoded,0)

    def test_native_loader_uses_local_verified_model_and_no_asr(self):
        captured={}
        def load(path,**kwargs): captured.update(path=path,**kwargs);return model
        model=Model();model.eval=lambda:model
        with patch.dict(sys.modules,{"torch":NS(float16="float16"),"omnivoice":NS(OmniVoice=NS(from_pretrained=load))}):
            self.assertIs(worker.load_model(Path("/prepared"),logic.DEFAULTS),model)
        self.assertEqual(captured["path"],"/prepared/model")
        self.assertFalse(captured["load_asr"]);self.assertTrue(captured["local_files_only"])
        self.assertEqual(captured["attn_implementation"],"sdpa")

    def test_sdk_commit_and_dependency_pins(self):
        versions={"torch":"2.8.0","torchaudio":"2.8.0","transformers":"5.3.0","accelerate":"1.10.1",
            "omnivoice":"0.2.1","numpy":"2.2.6","soundfile":"0.13.1","librosa":"0.11.0"}
        def dist(commit):
            return NS(read_text=lambda _:json.dumps({"url":"https://github.com/k2-fsa/OmniVoice.git","vcs_info":{"commit_id":commit}}))
        with patch("importlib.metadata.version",side_effect=versions.get),patch("importlib.metadata.distribution",return_value=dist(worker.SDK_REVISION)):
            worker.require_dependencies()
        with patch("importlib.metadata.version",side_effect=versions.get),patch("importlib.metadata.distribution",return_value=dist("wrong")),self.assertRaises(logic.OmniVoiceError):
            worker.require_dependencies()

    def test_reference_resampling_bounds_and_preservation(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg=logic.configuration(settings(root));original=Path(cfg["reference_audio"]).read_bytes()
            job=root/"job";job.mkdir()
            meta=logic.prepare_reference(cfg,root,job,time.monotonic()+10,lambda:False)
            self.assertEqual(meta["duration_sec"],1)
            self.assertEqual(Path(cfg["reference_audio"]).read_bytes(),original)
            self.assertIsNone(logic.prepare_reference({**cfg,"voice_mode":"auto"},root,job,time.monotonic()+5,lambda:False))
        for rate,frames in ((8000,20*8000+1),(48000,21*48000),(24000,11999)):
            with TemporaryDirectory() as directory:
                root=Path(directory);reference=root/"reference.wav";job=root/"job";job.mkdir()
                with wave.open(str(reference),"wb") as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(b"\x01\x10"*frames)
                cfg=logic.configuration({"voice_mode":"clone","reference_audio":str(reference),"reference_text":"Words."})
                with self.assertRaises(logic.OmniVoiceError):
                    logic.prepare_reference(cfg,root,job,time.monotonic()+10,lambda:False)

    def test_memory_and_device_preflight(self):
        cfg=logic.configuration({"device":"cpu","precision":"float32"})
        with patch.object(worker,"available_ram",return_value=0),self.assertRaises(logic.OmniVoiceError):
            worker.resource_preflight(TORCH,cfg)
        with patch.object(worker,"available_ram",return_value=32*1024**3):
            worker.resource_preflight(TORCH,cfg)
            with self.assertRaises(logic.OmniVoiceError):
                worker.resource_preflight(NS(cuda=NS(is_available=lambda:False)),logic.DEFAULTS)

    def test_bad_decoded_audio_is_never_published(self):
        cases=(np.array([],dtype=np.float32),np.zeros(20,dtype=np.float32),
            np.array([float("nan")],dtype=np.float32),np.array([float("inf")],dtype=np.float32),
            np.array([5.0],dtype=np.float32),np.ones((1,20),dtype=np.float32),
            np.ones(20,dtype=np.int64),np.ones(24000*60+1,dtype=np.float32))
        for samples in cases:
            with TemporaryDirectory() as directory,patch.dict(sys.modules,modules()):
                model=Model();model._decode_and_post_process=lambda *a:samples
                with self.assertRaises(logic.OmniVoiceError):
                    worker.generate_audio(model,"Hello.",logic.DEFAULTS,Path(directory))
                self.assertFalse((Path(directory)/"generated.wav").exists())

    def test_native_pcm_clipping_is_counted(self):
        with TemporaryDirectory() as directory,patch.dict(sys.modules,modules()):
            model=Model();model._decode_and_post_process=lambda *a:np.array([1.2,-1.1,.5],dtype=np.float32)
            job=Path(directory);info=worker.generate_audio(model,"Hello.",logic.DEFAULTS,job)
            self.assertEqual(info["clipped_samples"],2)
            self.assertEqual(info["frames"],3)
            self.assertEqual(logic.audio_info(job/"generated.wav",logic.DEFAULTS,json.dumps(info)),info)

if __name__=="__main__": unittest.main()
