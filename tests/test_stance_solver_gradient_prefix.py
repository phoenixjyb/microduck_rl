"""Conditional CPU prefix math and prospective packet guards; never CUDA."""
import ast
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mjlab_microduck import stance_solver_gradient_prefix as g
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_efc_transition as efc


@pytest.fixture(scope="module")
def raw_source():
    return (Path(sys.prefix)/"lib/python3.12/site-packages/mujoco_warp/_src/solver.py").read_bytes()


def test_frozen_source_and_distinct_initialization_iteration_routes(raw_source):
    r = g.verify_source(raw_source)
    assert r["profile"]["caller_lines"] == {"_update_gradient": [2925,2927], "_update_gradient_incremental": [3061,3063]}
    assert r["profile"]["newton_pyramidal_iteration_track_changes"]
    assert r["profile"]["iteration_uses_incremental_gradient"]
    assert not r["profile"]["initialization_track_changes"]
    assert not r["actual_dispatch_observed"] and not any(r["qualification"].values())


@pytest.mark.parametrize("old,new", [
    (b"ctx_grad_dot_out[worldid] = 0.0", b"ctx_grad_dot_out[worldid] = -0.0"),
    (b"grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]", b"grad = efc_Ma_in[worldid, dofid] - (qfrc_smooth_in[worldid, dofid] + qfrc_constraint_in[worldid, dofid])"),
    (b"wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)", b"wp.atomic_add(ctx_grad_dot_out, worldid, grad)"),
    (b"ctx_grad_out[worldid, dofid] = grad", b"ctx_grad_out[worldid, dofid] = -grad"),
    (b"wp.launch(update_gradient_zero_grad_dot, dim=(d.nworld)", b"wp.launch(update_gradient_zero_grad_dot, dim=(m.nv)"),
    (b"inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done]", b"inputs=[d.qfrc_constraint, d.qfrc_smooth, d.efc.Ma, ctx.done]"),
    (b"outputs=[ctx.grad, ctx.grad_dot]", b"outputs=[ctx.Mgrad, ctx.grad_dot]"),
    (b"_update_constraint(m, d, ctx, track_changes=incremental)", b"_update_constraint(m, d, ctx, track_changes=False)"),
    (b"    _update_gradient_incremental(m, d, ctx)", b"    _update_gradient(m, d, ctx)"),
    (b"incremental = m.opt.solver == types.SolverType.NEWTON and m.opt.cone != types.ConeType.ELLIPTIC", b"incremental = False"),
    (b"  _update_constraint(m, d, ctx)\n\n  if grad:", b"  _update_constraint(m, d, ctx, track_changes=True)\n\n  if grad:"),
])
def test_static_contract_mutations_refused(raw_source,old,new):
    assert old in raw_source
    with pytest.raises(ValueError): g.source_profile(raw_source.replace(old,new,1))


@pytest.mark.parametrize("extra", [g.KERNELS.encode(), b"\nother = update_gradient_grad\n", b"\ndef _update_gradient(m,d,ctx): pass\n"])
def test_extra_definitions_references_refused(raw_source,extra):
    with pytest.raises(ValueError): g.source_profile(raw_source+extra)


@pytest.mark.parametrize("raw", [None,"source",b"",b"x"*(g.static.MAX_SOURCE_BYTES+1)])
def test_bad_source_type_length(raw):
    with pytest.raises(ValueError): g.verify_source(raw)


def test_structural_success_is_not_whole_source_pin(raw_source):
    assert g.source_profile(raw_source+b"\n# changed bytes\n")
    with pytest.raises(ValueError,match="whole frozen"): g.verify_source(raw_source+b"\n# changed bytes\n")


@pytest.mark.parametrize("ma,smooth,constraint,bits", [
    (3.0,1.0,1.0,0x3f800000), (0.0,0.0,0.0,0),
    (-0.0,0.0,0.0,0x80000000), (0.0,-0.0,-0.0,0),
    (float(2**24),-1.0,float(2**24),0),  # left association rounds away the one
    (2.0**-149,0.0,0.0,1), (0.0,2.0**-149,0.0,0x80000001),
])
def test_exact_gradient_rounding_and_zero_sign(ma,smooth,constraint,bits):
    assert struct.unpack("<I",struct.pack("<f",g.gradient(ma,smooth,constraint)))[0] == bits


@pytest.mark.parametrize("values", [(1,0.0,0.0),(0.1,0.0,0.0),(float('nan'),0.0,0.0),
    (float('inf'),0.0,0.0),(float(2**127),-float(2**127),0.0)])
def test_gradient_unsupported_or_overflow_refuses(values):
    with pytest.raises(ValueError): g.gradient(*values)


@pytest.fixture(scope="module")
def packet():
    return efc.historical(Path(g.__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def parents():
    root=Path(g.__file__).resolve().parents[2]/"artifacts/evaluations/efc-transition-7717b4cbef3a"
    expected={'reference':'c7dd694ed1dfc132d42ef9cdd252f7ca3f78a9b2b04b3c8722ace2e84861ab69',
              'control':'7f9f0fb349a3b9713cf8747e535b404f9516f161982f8fabd1e80a3156b3ce47'}
    out={a:(root/(a+'-gauss.after.bin')).read_bytes() for a in expected}
    assert all(sha256(raw).hexdigest()==expected[a] for a,raw in out.items())
    return out


@pytest.fixture(scope="module")
def prospective(parents,packet):
    initials=g.initial_banks(parents,packet)
    arms={}
    # Independent NumPy binary32 arithmetic, not calls to the predictor/oracle.
    for arm,raw in initials.items():
        f=g.unpack_bank(raw);banks={g.PHASES[0]:raw}
        f['context.grad_dot']=bytes(256)
        banks[g.PHASES[1]]=banks[g.PHASES[2]]=g.pack_bank(f)
        ma,smooth,force=(np.frombuffer(f[n],dtype='<f4').reshape(64,20) for n in ('efc.Ma','data.qfrc_smooth','data.qfrc_constraint'))
        grad=np.float32(np.float32(ma-smooth)-force)
        dot=np.zeros(64,dtype='<f4')
        for i in range(20): dot=np.float32(dot+np.float32(grad[:,i]*grad[:,i]))
        f['context.grad'],f['context.grad_dot']=grad.tobytes(),dot.tobytes()
        banks[g.PHASES[3]]=g.pack_bank(f);arms[arm]=banks
    anchors={a:{p:dict(bytes=len(raw),sha256=sha256(raw).hexdigest()) for p,raw in banks.items()} for a,banks in arms.items()}
    return NS(arms=arms,anchors=anchors)


def test_prospective_complete_two_arm_contract(prospective,parents,packet):
    r=g.receive_banks(prospective.arms,prospective.anchors,parents,packet)
    assert r['packet_bytes']==3735368 and r['total_bytes']==29882944
    assert r['arms']['control']['gradient_replacements']==1280
    assert r['arms']['control']['nonzero_norm_worlds']==64
    assert all(a['gradient_bit_mismatches']==a['norm_outside_interval']==0 for a in r['arms'].values())
    assert not any(r['qualification'].values()) and not r['native_execution_authenticated']
    assert not r['parent_capture_authenticated']


@pytest.mark.parametrize('arm,phase',[(a,p) for a in ('reference','control') for p in g.PHASES])
def test_all_eight_hashes_before_any_decode(prospective,parents,packet,monkeypatch,arm,phase):
    arms={a:dict(b) for a,b in prospective.arms.items()}
    arms[arm][phase]=b'x'+arms[arm][phase][1:]
    monkeypatch.setattr(g,'unpack_bank',lambda _:pytest.fail('premature decode'))
    monkeypatch.setattr(stages,'unpack_bank',lambda _:pytest.fail('premature parent decode'))
    with pytest.raises(ValueError,match='whole external'): g.receive_banks(arms,prospective.anchors,parents,packet)


@pytest.mark.parametrize('fault',['grad','norm','nan-grad','nan-norm','negative-norm','negative-zero-norm','unrelated','done','reset','continuity','initial'])
def test_reanchored_wrong_bank_rejected(prospective,parents,packet,fault):
    arms={a:dict(b) for a,b in prospective.arms.items()};a='control';p=g.PHASES[-1]
    f=g.unpack_bank(arms[a][p])
    if fault in ('grad','nan-grad'): f['context.grad']=struct.pack('<f',float('nan') if fault=='nan-grad' else 999.0)+f['context.grad'][4:]
    elif fault in ('norm','nan-norm','negative-norm','negative-zero-norm'):
        v={'norm':0.0,'nan-norm':float('nan'),'negative-norm':-1.0,'negative-zero-norm':-0.0}[fault]
        f['context.grad_dot']=struct.pack('<f',v)+f['context.grad_dot'][4:]
    elif fault=='unrelated': f['efc.force']=bytes(len(f['efc.force']))
    elif fault=='done': f['context.done']=b'\1'+bytes(63)
    else:
        p={'reset':g.PHASES[1],'continuity':g.PHASES[2],'initial':g.PHASES[0]}[fault]
        f=g.unpack_bank(arms[a][p]);f['context.grad_dot']=struct.pack('<64f',*([1.0]*64))
    arms[a][p]=g.pack_bank(f)
    anchors={a:{p:dict(bytes=len(raw),sha256=sha256(raw).hexdigest()) for p,raw in banks.items()} for a,banks in arms.items()}
    with pytest.raises(ValueError): g.receive_banks(arms,anchors,parents,packet)


@pytest.mark.parametrize('fault',['missing','extra','mutable','short','bad-done','bad-count','elliptic','overflow','square-overflow'])
def test_closed_prediction_domain(parents,fault):
    f=stages.unpack_bank(parents['control'])
    if fault=='missing': del f['efc.Ma']
    elif fault=='extra': f['extra']=b''
    elif fault=='mutable': f['efc.Ma']=bytearray(f['efc.Ma'])
    elif fault=='short': f['efc.Ma']=f['efc.Ma'][:-1]
    elif fault=='bad-done': f['context.done']=b'\1'+bytes(63)
    elif fault=='bad-count': f['data.nefc']=struct.pack('<64i',*([513]*64))
    elif fault=='elliptic': f['efc.type']=struct.pack('<i',7)+f['efc.type'][4:]
    elif fault=='overflow':
        f['efc.Ma']=struct.pack('<1280f',*([float(2**127)]*1280))
        f['data.qfrc_smooth']=struct.pack('<1280f',*([-float(2**127)]*1280))
    else: f['efc.Ma']=struct.pack('<1280f',*([float(2**100)]*1280))
    with pytest.raises(ValueError): g.prediction(f)


def test_conditional_norm_accepts_distinct_serial_orders():
    squares=(float(2**24),1.0,1.0)+((0.0,)*17)
    lo,hi,_,_=efc.cost_interval(tuple((v,v) for v in squares))
    results=[]
    for seq in (squares,tuple(reversed(squares))):
        value=0.0
        for x in seq: value=g.arithmetic.round32(Fraction(value)+Fraction(x))
        results.append(value);assert lo<=Fraction(value)<=hi
    assert results[0]!=results[1]


def test_original_kernel_bodies_on_cpu_only_fake_carriers(raw_source,prospective):
    g.verify_source(raw_source)
    tree=ast.parse(raw_source)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('update_gradient_zero_grad_dot','update_gradient_grad')]
    for fn in funcs:
        fn.decorator_list=[]
        for arg in fn.args.args: arg.annotation=None
    ids=[0,0]
    def tid(): return ids[0] if ids[1] is None else tuple(ids)
    def add(a,w,v): a[w]=np.float32(a[w]+v)
    wp=NS(tid=tid,atomic_add=add);ns={'wp':wp}
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),'<frozen-gradient-cpu-carrier>','exec'),ns)
    for arm in ('reference','control'):
        f=g.unpack_bank(prospective.arms[arm][g.PHASES[0]])
        def arr(n,shape): return np.frombuffer(f[n],dtype='<f4').reshape(shape).copy()
        ma,smooth,force,grad=(arr(n,(64,20)) for n in ('efc.Ma','data.qfrc_smooth','data.qfrc_constraint','context.grad'))
        dot=arr('context.grad_dot',(64,));done=np.zeros(64,dtype=bool)
        for w in range(64):
            ids[:]=[w,None];ns['update_gradient_zero_grad_dot'](done,dot)
            for i in range(20):
                ids[:]=[w,i];ns['update_gradient_grad'](smooth,force,ma,done,grad,dot)
        expected=g.unpack_bank(prospective.arms[arm][g.PHASES[-1]])
        assert grad.tobytes()==expected['context.grad'] and dot.tobytes()==expected['context.grad_dot']


def test_retained_reauthentication_predicts_but_does_not_capture(raw_source):
    root=Path(g.__file__).resolve().parents[2]
    path=(Path(sys.prefix)/'lib/python3.12/site-packages/mujoco_warp/_src/solver.py').resolve()
    r=g.audit_retained(root,path)
    assert r['paired_predicted_gradient_differences']==[20]*64
    assert r['arms']['reference']['predicted_gradient_sha256']=='224608728e232d39efa86c599f0ed07da7cdf918376092589ba3ebdcbc56c935'
    assert r['arms']['control']['predicted_gradient_sha256']=='a55349035f3ab775c9c36b91b384eb1d38436877e8d2467a4f8fd6ce67832aac'
    assert not r['gradient_prefix_gpu_run'] and not any(r['qualification'].values())
    assert all(not r['arms'][a]['captured_gradient_available'] for a in r['arms'])


@pytest.mark.parametrize('name',['inventory.json','retirement.json','wsl-receiver.json'])
def test_predecessor_whole_closeout_hashes_before_receive(monkeypatch,name):
    root=Path(g.__file__).resolve().parents[2]
    path=(Path(sys.prefix)/'lib/python3.12/site-packages/mujoco_warp/_src/solver.py').resolve()
    original=g.prior.read_plain
    def changed(p,*args):
        raw=original(p,*args)
        return raw+b'\n' if p.name==name and 'efc-transition-closeout-7717b4cbef3a' in p.parts else raw
    monkeypatch.setattr(g.prior,'read_plain',changed)
    monkeypatch.setattr(g.receiver,'receive',lambda *a:pytest.fail('premature predecessor receive'))
    with pytest.raises(ValueError,match='whole literal predecessor'): g.audit_retained(root,path)


def test_reread_parent_bytes_reauthenticated_before_prediction(monkeypatch):
    root=Path(g.__file__).resolve().parents[2]
    path=(Path(sys.prefix)/'lib/python3.12/site-packages/mujoco_warp/_src/solver.py').resolve()
    original=g.prior.read_plain
    def changed(p,*args):
        raw=original(p,*args)
        return b'x'+raw[1:] if p.name=='control-gauss.after.bin' else raw
    monkeypatch.setattr(g.prior,'read_plain',changed)
    monkeypatch.setattr(g,'prediction',lambda *a:pytest.fail('premature prediction'))
    with pytest.raises(ValueError,match='whole re-read parent'): g.audit_retained(root,path)


def test_new_source_authenticates_before_historical_decode(monkeypatch):
    root=Path(g.__file__).resolve().parents[2]
    path=(Path(sys.prefix)/'lib/python3.12/site-packages/mujoco_warp/_src/solver.py').resolve()
    original=g.prior.read_plain
    monkeypatch.setattr(g.prior,'read_plain',lambda p,*args: original(p,*args)+b'\n' if p==path else original(p,*args))
    monkeypatch.setattr(g.efc,'historical',lambda *a:pytest.fail('premature historical decode'))
    with pytest.raises(ValueError,match='whole frozen'): g.audit_retained(root,path)


def test_inert_import_and_hidden_cuda_cli_refusal():
    code="from mjlab_microduck import stance_solver_gradient_prefix; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True,timeout=10)
    import os
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    r=subprocess.run([sys.executable,'-m','mjlab_microduck.stance_solver_gradient_prefix','--output','/not-used'],env=env,capture_output=True,text=True,timeout=10)
    assert r.returncode!=0 and 'explicit CPU-only' in r.stderr
