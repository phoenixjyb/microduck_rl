from pathlib import Path
from hashlib import sha256
import io, importlib.metadata as md, json, os, platform, socket, subprocess, time
import torch
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_packed_evaluation_probe as probe
started=time.monotonic()
source='7bf19b4b53a559dae87b138c3fc628abec582c67'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source
assert subprocess.check_output(['git','status','--porcelain'],text=True).strip()==''
assert Path(probe.__file__).resolve()==Path.cwd()/'src/mjlab_microduck/stance_packed_evaluation_probe.py'
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
settings={k:os.environ.get(k) for k in ('ATEN_CPU_CAPABILITY','MKL_CBWR','OMP_NUM_THREADS')}
assert settings=={'ATEN_CPU_CAPABILITY':'default','MKL_CBWR':'COMPATIBLE','OMP_NUM_THREADS':'1'}
assert torch.backends.cpu.get_cpu_capability()=='DEFAULT' and torch.get_num_threads()==1
versions={name:md.version(name) for name in ('torch','warp-lang','mujoco','mujoco-warp','mjlab','better-actuator-models')}
assert versions=={'torch':'2.9.1','warp-lang':'1.12.0','mujoco':'3.10.0','mujoco-warp':'3.8.1','mjlab':'1.3.0','better-actuator-models':'1.0.1'}
lib=Path(torch.__file__).parent/'lib/libtorch_cpu.so'
library_sha256=sha256(lib.read_bytes()).hexdigest()
assert library_sha256=='7918fc09ff644c9b667921100b924e33ea432c85737c2982b56283691b32a4d7'
service=os.environ['MICRODUCK_R1_PARITY_SERVICE']
assert service in ('microduck-cpu-r1-portable-parity-100100.service','microduck-cpu-r1-portable-parity-10098.service')
props={key:subprocess.check_output(['systemctl','--user','show',service,'-p',key,'--value'],text=True).strip() for key in ('MainPID','RuntimeMaxUSec','KillMode','ActiveState','MemoryMax','CPUQuotaPerSecUSec','Nice')}
assert props=={'MainPID':str(os.getpid()),'RuntimeMaxUSec':'3min','KillMode':'control-group','ActiveState':'active','MemoryMax':str(6*1024**3),'CPUQuotaPerSecUSec':'2s','Nice':'10'}
root=Path(os.environ['MICRODUCK_R1_ARCHIVE'])
expected={"child.log":"40891801f409d846997ad7cc4a9176ed49c98b7b120f4f2a1b79a18365577a4e","launch.json":"fe3d40e4844a85e9089b6f9c37c8c74934dbfefca027de68f6689492806eceb0","measurements.json":"b38a8b3cfabd7c12ee8e0a04800962f5383656cdb59bb36140c016a588cd89db","packed-probe-255-seed-541.json":"fc585695848fbb921c8644020b63d8ddad7b1cf80a0cdfa13fb6fe1b5780296e","packed-probe-255-seed-541/checkpoint.pt":"2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5","packed-probe-255-seed-541/control.pt":"0a088942bdc03ccebce76ae816b619d6a218e7b333cf3a71fb30bef89fb597d6","packed-probe-255-seed-541/launch.json":"26b48db63256daffa17dcf577f6c3750f1b52a369be5679cdef2c3ad13df0988","packed-probe-255-seed-541/manifest.json":"b0863ff3d5b8bc08ccca3b580c584af5363cea94dee151bf1dc38878d2d7ad6c","packed-probe-255-seed-541/restore.json":"decacb77865d13e2f9c48502a2dc976fe7afb86ccb05a0d27d732f2d9c4c034f","packed-probe-255-seed-541/runtime.json":"48dc1fa2d594fe457da650594713a6f60756003647bf0e256021e2c8cadc1aec","packed-probe-255-seed-541/score.json":"f34fe7b4b1b9aec71720e89e794d78b3626b557a846a7457524f28ff5e5d514e","packed-probe-255-seed-541/trace.pt":"68c1f1bba3b7f30814471bd560d9956cf0e199e0513a314cd3848e124c821205","report.json":"e2f417eedecc6095ae5130f83f76a6abc17af26db449764da31e40247935ffdd","runtime.json":"48dc1fa2d594fe457da650594713a6f60756003647bf0e256021e2c8cadc1aec"}
assert all(not p.is_symlink() for p in root.rglob('*'))
actual={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
assert actual==expected
inventory_sha256=sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert inventory_sha256=='ad30b8c0ba6bda50a586d9798f4ac97ed886a5355b9599ddaa4afcba16e40b1f'
directory=root/'packed-probe-255-seed-541'
launch=json.loads((directory/'launch.json').read_text())
score=json.loads((directory/'score.json').read_text())
binding=score['binding']
actor,restored,compiled=bundle.checked_inputs(binding,(directory/'checkpoint.pt').read_bytes(),launch['checkpoint_identity'],(directory/'runtime.json').read_bytes(),(directory/'launch.json').read_bytes())
assert json.loads((directory/'restore.json').read_text())==restored
raw=(directory/'trace.pt').read_bytes()
trace_score=trace.verify(raw,expected['packed-probe-255-seed-541/trace.pt'],binding)
assert trace_score['complete_attempts']==128 and trace_score['numerical_passes']==128
payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
assert len(payload['ticks'])==250
outputs_hash=sha256(); inputs_hash=sha256(); max_error=0.0
for i,tick in enumerate(payload['ticks']):
    obs=tick['actor_input']; predicted=cp.infer(actor,obs)
    assert predicted.dtype==torch.float32 and tuple(predicted.shape)==(128,10)
    assert torch.allclose(predicted,tick['actions'],atol=bundle.ATOL,rtol=bundle.RTOL)
    header=(json.dumps([i,list(predicted.shape),'torch.float32'],separators=(',',':'))+'\n').encode()
    outputs_hash.update(header); outputs_hash.update(predicted.contiguous().numpy().tobytes())
    inputs_hash.update((json.dumps([i,list(obs.shape),'torch.float32'],separators=(',',':'))+'\n').encode()); inputs_hash.update(obs.contiguous().numpy().tobytes())
    max_error=max(max_error,float((predicted-tick['actions']).abs().max()))
assert not torch.cuda.is_initialized()
receipt={'protocol':'portable-cpu-actor-parity-diagnostic-v1','diagnosis_only':True,'source':source,'module_file':str(Path(probe.__file__).resolve()),'hostname':socket.gethostname(),'platform':platform.platform(),'python':platform.python_version(),'versions':versions,'cpu_capability':torch.backends.cpu.get_cpu_capability(),'intraop_threads':torch.get_num_threads(),'environment':settings,'torch_cpu_library_sha256':library_sha256,'inventory_sha256':inventory_sha256,'checkpoint_sha256':expected['packed-probe-255-seed-541/checkpoint.pt'],'trace_sha256':expected['packed-probe-255-seed-541/trace.pt'],'actor_inputs_sha256':inputs_hash.hexdigest(),'actor_outputs_sha256':outputs_hash.hexdigest(),'batches':250,'worlds_per_batch':128,'output_shape_per_batch':[128,10],'original_actor_tolerances':{'atol':bundle.ATOL,'rtol':bundle.RTOL},'all_original_actor_tolerance_checks_passed':True,'maximum_error_against_original_r1_actions':max_error,'entire_compiled_plant_checked':True,'service':service,'service_properties':props,'invocation_id':os.environ['INVOCATION_ID'],'elapsed_seconds':time.monotonic()-started,'cuda_initialized':False,'original_r1_strict_replay_passed_by_this_diagnostic':False,'independent_gpu_reexecution':False,'binary_runtime_equivalence':False,'full_evaluation_launched':False,'checkpoint_admitted':False,'learned_stance_accepted':False,'football_balance_accepted':False,'physical_motion_authorized':False}
print('PORTABLE_CPU_PARITY_JSON='+json.dumps(receipt,sort_keys=True,separators=(',',':')),flush=True)
