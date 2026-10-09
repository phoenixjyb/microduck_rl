"""Frozen Python API boundary preparation; inert until explicitly constructed.

This authenticates selected Warp function bodies, not all transitive Python
dependencies, native library internals, device state or a hostile interpreter.
The native owner/child and independent receiver remain separate prerequisites.
"""
import ast
import builtins
import dis
from hashlib import sha256
from pathlib import Path
import sys
from types import BuiltinFunctionType, CodeType, FunctionType, MappingProxyType, ModuleType
import typing

from mjlab_microduck import stance_solver_gradient_executable as executable

PROTOCOL = "microduck-gradient-runtime-boundary-oct9-v1"
BASE = "e7b12f6d07f74a388b926b4252b9ba9d4f045e77"
OWN = frozenset({
    "src/mjlab_microduck/stance_solver_gradient_runtime.py",
    "tests/test_stance_solver_gradient_runtime.py",
    "docs/experiments/2026-10-09-gradient-runtime-boundary.md",
})
TESTS = executable.TESTS + ("tests/test_stance_solver_gradient_runtime.py",)
need = executable.need
PINS = MappingProxyType({
    "warp": (20260, "ad341b5dea2937b8885a0b3002b3bfc5bef9db83eb2ab8c9f9bfcdb81551fe56"),
    "warp._src.context": (408818, "eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee"),
    "warp._src.types": (256907, "9224fb234ff61f4a1c3338b7171d545171ab051fb625679568ea80a6af8321bd"),
    "warp._src.build": (22591, "5993e58b44a916fc12f35afde1cededc59ab1db7df7a7eecd75551191d9197df"),
})
CONTEXT_FUNCTIONS = ("launch", "copy", "empty", "get_device", "get_stream", "synchronize_stream")
CONTEXT_METHODS = MappingProxyType({
    "Module": ("__init__", "get_module_hash", "_compile", "load", "_get_compile_arch",
               "_get_compile_output_name", "_get_meta_name", "get_module_identifier"),
    "ModuleExec": ("__init__", "get_kernel_hooks"),
    "ModuleBuilder": ("__init__", "codegen"),
    "Kernel": ("__init__", "get_mangled_name"),
})


def source_binding(root):
    """New three-path fence; no changes to any retained source contract."""
    cmd = executable.adapter.prefix.prior.retained.command
    need(type(root) is type(Path.cwd()) and root == root.resolve(strict=True)
         and Path.cwd().resolve() == root
         and cmd("git", "branch", "--show-current").decode().strip() == executable.adapter.prefix.prior.BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact runtime-boundary branch")
    source = cmd("git", "rev-parse", "HEAD").decode().strip()
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN,
         "separate exact three-path runtime-boundary fence")
    # Reuse the mechanical leaf inventory, not an older phase's source fence.
    from hashlib import sha1
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed runtime-boundary leaf")
        raw = executable.adapter.prefix.prior.read_plain(root / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
             "whole committed runtime-boundary blob")
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(commit=source, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(),
                base=BASE, leaves=leaves)


def _child(code, name):
    rows = [c for c in code.co_consts if type(c) is CodeType and c.co_name == name]
    need(len(rows) == 1, "unique frozen qualified code")
    return rows[0]


def _default(node):
    # Only the two nonliteral defaults in these pinned entries are allowed.
    if type(node) is ast.Name:
        need(node.id in ("float", "Any"), "closed nonliteral default name")
        return builtins.float if node.id == "float" else typing.Any
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise ValueError("literal frozen API default") from error


def _value(actual, expected):
    """Typed literal comparison, never invoking arbitrary equality methods."""
    if expected is typing.Any or expected is builtins.float:
        return actual is expected
    if type(actual) is not type(expected):
        return False
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(_value(a, b) for a, b in zip(actual, expected))
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_value(actual[k], v) for k, v in expected.items())
    need(type(expected) in (type(None), bool, str, int, float), "closed default value type")
    return actual == expected


def _globals(code):
    names = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"}
    for child in code.co_consts:
        if type(child) is CodeType:
            names |= _globals(child)
    return names


def _builtin(name, value):
    """Reject Python shims and different named builtins, not C implementation proof."""
    if type(value) is BuiltinFunctionType:
        module = "_io" if name == "open" else "builtins"
        return value.__module__ == module and value.__name__ == name and value.__self__ is sys.modules.get(module)
    return (type(value) is type and value.__module__ == "builtins" and value.__name__ == name
            and not value.__flags__ & (1 << 9))  # CPython Py_TPFLAGS_HEAPTYPE


def _definition(code, node, qualified):
    for name in qualified.split("."):
        code = _child(code, name)
        nodes = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
        need(len(nodes) == 1, "unique frozen qualified definition")
        node = nodes[0]
    need(type(node) is ast.FunctionDef and not node.decorator_list, "plain frozen API definition")
    need(all(n is None for n in node.args.kw_defaults), "no frozen keyword-only defaults")
    defaults = tuple(_default(n) for n in node.args.defaults) or None
    return code, defaults


class WarpApiGuard:
    """Selected original API bodies and held dependencies, before allocation.

    No runtime function is invoked by construction, check or record. In
    particular this does not call init, get_device, compile, load or launch.
    Construct after the owner's independently authenticated initialization;
    a subsequent runtime-object replacement is refused, not silently adopted.
    """
    __slots__ = ("modules", "paths", "classes", "class_shapes", "rows", "aliases", "runtime", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("runtime API bindings are sealed")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise AttributeError("runtime API bindings are sealed")
        object.__delattr__(self, name)

    def __init__(self, *, wp, context, types, build):
        modules = dict(zip(PINS, (wp, context, types, build)))
        root = (Path(sys.prefix) / "lib/python3.12/site-packages/warp").resolve(strict=True)
        paths, definitions = {}, {}
        for name, module in modules.items():
            relative = "__init__.py" if name == "warp" else name.removeprefix("warp.").replace(".", "/") + ".py"
            path = root / relative
            need(type(module) is ModuleType and sys.modules.get(name) is module
                 and module.__name__ == name and Path(module.__file__).resolve(strict=True) == path
                 and path.resolve(strict=True) == path, "canonical installed Warp module")
            data, _ = executable.retained._read_plain(path, 1024 * 1024, "frozen runtime API source")
            need((len(data), sha256(data).hexdigest()) == PINS[name], "whole frozen runtime API source")
            paths[name] = path
            definitions[name] = (compile(data, str(path), "exec", dont_inherit=True, optimize=0),
                                 ast.parse(data, filename=str(path)))
        need(wp._src.context is context and wp._src.types is types and wp._src.build is build,
             "canonical exported Warp source modules")
        classes, selections = {}, []
        for name in CONTEXT_FUNCTIONS:
            selections.append((context, name, "warp._src.context", name))
        for name, methods in CONTEXT_METHODS.items():
            cls = context.__dict__.get(name)
            need(type(cls) is type and cls.__module__ == context.__name__ and cls.__qualname__ == name
                 and cls.__bases__ == (object,), "plain original runtime class boundary")
            classes[(context, name)] = cls
            selections.extend((cls, method, context.__name__, name + "." + method) for method in methods)
        cls = types.__dict__.get("array")
        need(type(cls) is type and cls.__module__ == types.__name__ and cls.__qualname__ == "array"
             and cls.__bases__ == (types.Array,), "plain original array class boundary")
        classes[(types, "array")] = cls
        classes[(types, "Array")] = types.Array
        selections.extend((cls, method, types.__name__, "array." + method) for method in ("__init__", "numpy"))
        selections.extend((build, name, build.__name__, name) for name in ("build_cuda", "load_cuda"))
        rows = []
        for owner, name, module_name, qualified in selections:
            module = modules[module_name]
            fn = owner.__dict__.get(name)
            expected, defaults = _definition(*definitions[module_name], qualified)
            need(type(fn) is FunctionType and fn.__code__ == expected
                 and fn.__globals__ is module.__dict__ and fn.__builtins__ is builtins.__dict__
                 and (fn.__name__, fn.__qualname__, fn.__module__) == (name, qualified, module_name)
                 and fn.__closure__ is None and fn.__kwdefaults__ is None
                 and not fn.__dict__ and _value(fn.__defaults__, defaults), "original frozen API body/defaults/globals")
            dependencies = {}
            for global_name in _globals(expected):
                namespace = module.__dict__ if global_name in module.__dict__ else builtins.__dict__
                need(global_name in namespace, "resolved frozen API global")
                value = namespace[global_name]
                # Disallow a module-global override of a standard built-in.
                if global_name in builtins.__dict__ and not (module is types and global_name == "bool"):
                    need(namespace is builtins.__dict__ and _builtin(global_name, value),
                         "unshadowed named CPython builtin API dependency")
                dependencies[global_name] = (namespace, value,
                    value.__code__ if type(value) is FunctionType else None)
            rows.append((owner, name, fn, fn.__code__, fn.__defaults__, defaults, module,
                         MappingProxyType(dependencies), qualified))
        aliases = tuple((wp, name, context.__dict__[name]) for name in CONTEXT_FUNCTIONS)
        aliases += ((wp, "array", cls), (wp, "Module", context.Module), (wp, "Kernel", context.Kernel))
        self.modules, self.paths, self.classes = MappingProxyType(modules), MappingProxyType(paths), MappingProxyType(classes)
        self.class_shapes = tuple((c, c.__module__, c.__qualname__, c.__bases__) for c in classes.values())
        self.rows, self.aliases, self.runtime = tuple(rows), aliases, context.runtime
        self._sealed = True
        self.check()

    def check(self):
        for name, module in self.modules.items():
            path = self.paths[name]
            need(sys.modules.get(name) is module and module.__name__ == name
                 and Path(module.__file__).resolve(strict=True) == path
                 and path.resolve(strict=True) == path, "held canonical Warp module")
            raw, _ = executable.retained._read_plain(path, 1024 * 1024, "held runtime API source")
            need((len(raw), sha256(raw).hexdigest()) == PINS[name], "unchanged frozen runtime API source")
        wp, context = self.modules["warp"], self.modules["warp._src.context"]
        need(wp._src.context is context and wp._src.types is self.modules["warp._src.types"]
             and wp._src.build is self.modules["warp._src.build"] and context.runtime is self.runtime,
             "held Warp module exports and runtime object")
        need(all(owner.__dict__.get(name) is cls for (owner, name), cls in self.classes.items()),
             "held runtime and array classes")
        need(all(type(c) is type and (c.__module__, c.__qualname__, c.__bases__) == (m, q, bases)
                 for c, m, q, bases in self.class_shapes), "held runtime class shapes")
        need(all(owner.__dict__.get(name) is fn for owner, name, fn in self.aliases), "original Warp API export aliases")
        for owner, name, fn, code, held_defaults, expected_defaults, module, dependencies, qualified in self.rows:
            need(owner.__dict__.get(name) is fn and type(fn) is FunctionType and fn.__code__ is code
                 and fn.__globals__ is module.__dict__ and fn.__builtins__ is builtins.__dict__
                 and (fn.__name__, fn.__qualname__, fn.__module__) == (name, qualified, module.__name__)
                 and fn.__closure__ is None and fn.__kwdefaults__ is None and not fn.__dict__
                 and fn.__defaults__ is held_defaults and _value(fn.__defaults__, expected_defaults),
                 "held frozen API callable/defaults/descriptor")
            for global_name, (namespace, value, dependency_code) in dependencies.items():
                current_namespace = module.__dict__ if global_name in module.__dict__ else builtins.__dict__
                need(current_namespace is namespace and namespace.get(global_name) is value
                     and (dependency_code is None or value.__code__ is dependency_code), "held direct API global dependency")

    def record(self):
        self.check()
        return dict(protocol=PROTOCOL, sources={n: dict(path=str(self.paths[n]), bytes=p[0], sha256=p[1]) for n, p in PINS.items()},
                    entries=[dict(module=m.__name__, qualified=q, line=c.co_firstlineno) for _, _, _, c, _, _, m, _, q in self.rows],
                    selected_python_bodies_authenticated=True, held_direct_globals_checked=True,
                    transitive_dependencies_authenticated=False, initialized_runtime_origin_authenticated=False,
                    native_execution_observed=False, flags=dict(executable.adapter.prefix.FLAGS))
