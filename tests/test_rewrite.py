import ast
from pathlib import Path

import pytest

from tensi.mapping import PositionMap, clean_display
from tensi.rewrite import IMPORT_TEXT, rewrite
from tensi.seeds import parse_shadow

HERE = Path(__file__).parent
FIXTURES = ["plain", "attn", "bad"]


@pytest.mark.parametrize("name", FIXTURES)
def test_golden(name):
    src = (HERE / "fixtures" / f"{name}.py").read_text()
    res = rewrite(src)
    assert res.shadow == (HERE / "golden" / f"{name}.shadow.py").read_text()
    assert res.notes == []


@pytest.mark.parametrize("name", FIXTURES)
def test_line_count_preserved_and_valid(name):
    src = (HERE / "fixtures" / f"{name}.py").read_text()
    res = rewrite(src)
    assert len(res.shadow.split("\n")) == len(src.split("\n"))
    parse_shadow(res.shadow)  # valid Python on every supported version


def sh(src: str) -> str:
    res = rewrite(src)
    assert len(res.shadow.split("\n")) == len(src.split("\n"))
    parse_shadow(res.shadow)  # valid Python on every supported version
    return res.shadow


def test_untouched_when_nothing_to_do():
    src = "import torch\n\ndef f(x):\n    return x\n"
    assert sh(src) == src


def test_import_before_trailing_comment():
    s = sh("import torch  # main lib\n\ndef f(n: int):\n    pass\n")
    assert s.splitlines()[0] == f"import torch; {IMPORT_TEXT}  # main lib"


def test_future_import_is_skipped():
    s = sh("from __future__ import annotations\nimport torch\ndef f(n: int): pass\n")
    lines = s.splitlines()
    assert lines[0] == "from __future__ import annotations"
    assert lines[1] == f"import torch; {IMPORT_TEXT}"


def test_only_future_import():
    s = sh("from __future__ import annotations\ndef f(n: int): pass\n")
    assert s.splitlines()[0] == f"from __future__ import annotations; {IMPORT_TEXT}"


def test_docstring_first_without_imports():
    s = sh('"""Doc."""\ndef f(n: int): pass\n')
    assert s.splitlines()[0] == f'"""Doc."""; {IMPORT_TEXT}'


def test_merge_into_existing_type_params():
    s = sh("import torch\ndef g[T](n: int, other: T) -> T:\n    return other\n")
    assert "def g[T, _ts_n: _ts_IntVar](n: _ts_Int[_ts_n], other: T) -> T:" in s


def test_int_default_and_optional_untouched():
    s = sh(
        "import torch\nfrom typing import Optional\ndef f(a: int = 4, b: Optional[int] = None, c: int | None = None): pass\n"
    )
    assert "def f[_ts_a: _ts_IntVar](a: _ts_Int[_ts_a] = 4, b: Optional[int] = None, c: int | None = None)" in s


def test_star_args_untouched():
    s = sh("import torch\ndef f(*sizes: int, n: int, **kw: int): pass\n")
    assert "def f[_ts_n: _ts_IntVar](*sizes: int, n: _ts_Int[_ts_n], **kw: int)" in s


def test_jaxtyping_module_alias_and_return():
    s = sh(
        "import torch\nimport jaxtyping as jt\nfrom torch import Tensor\n"
        "def f(x: jt.Float[Tensor, 'b 3']) -> jt.Float[Tensor, 'b']:\n    return x[:, 0]\n"
    )
    assert "def f[_ts_b: _ts_IntVar](x: Tensor[[_ts_b, 3]]) -> Tensor[[_ts_b]]:" in s


def test_jaxtyping_qualified_tensor():
    s = sh('import torch\nfrom jaxtyping import Float\ndef f(x: Float[torch.Tensor, "n d"]): pass\n')
    assert "x: torch.Tensor[[_ts_n, _ts_d]]" in s


def test_unsupported_dims_left_unshaped_with_note():
    src = "import torch\nfrom jaxtyping import Float\nfrom torch import Tensor\ndef f(x: Float[Tensor, '*batch d']): pass\n"
    res = rewrite(src)
    assert "def f(x: Tensor):" in res.shadow
    assert any("not supported" in n for n in res.notes)


def test_non_jaxtyping_subscript_untouched():
    src = "import torch\nfrom typing import Annotated\ndef f(x: Annotated[int, 'meta']): pass\n"
    assert sh(src) == src


def test_function_dims_unify_with_jaxtyping_names():
    s = sh(
        "import torch\nfrom jaxtyping import Float\nfrom torch import Tensor\ndef f(n: int, x: Float[Tensor, 'n k']): pass\n"
    )
    assert "def f[_ts_n: _ts_IntVar, _ts_k: _ts_IntVar](n: _ts_Int[_ts_n], x: Tensor[[_ts_n, _ts_k]])" in s


def test_method_param_colliding_with_class_dim_gets_suffix():
    s = sh(
        "import torch\nclass M:\n    def __init__(self, d: int):\n        self.d = d\n"
        "    def make(self, d: int):\n        return torch.zeros(d)\n"
    )
    assert "class M[_ts_d: _ts_IntVar]:" in s
    assert "def make[_ts_d__2: _ts_IntVar](self, d: _ts_Int[_ts_d__2]):" in s
    assert clean_display("Tensor[[_ts_d__2]]") == "Tensor[[d]]"


def test_decorated_init_and_async_and_nested():
    s = sh(
        "import torch\nimport functools\nclass M:\n    @functools.wraps(object)\n    def __init__(self, d: int):\n        pass\n"
        "async def outer(n: int):\n    def inner(m: int):\n        return torch.zeros(n, m)\n    return inner\n"
    )
    assert "class M[_ts_d: _ts_IntVar]:" in s
    assert "def __init__(self, d: _ts_Int[_ts_d]):" in s
    assert "async def outer[_ts_n: _ts_IntVar](n: _ts_Int[_ts_n]):" in s
    assert "def inner[_ts_m: _ts_IntVar](m: _ts_Int[_ts_m]):" in s


def test_nested_jaxtyping_resolves_enclosing_dims():
    s = sh(
        "import torch\nfrom jaxtyping import Float\nfrom torch import Tensor\n"
        "def outer(n: int):\n    def inner(x: Float[Tensor, 'n 2']):\n        return x\n    return inner\n"
    )
    assert "def inner(x: Tensor[[_ts_n, 2]]):" in s


def test_multiline_signature_keeps_lines():
    src = (
        "import torch\nfrom jaxtyping import Float\nfrom torch import Tensor\n"
        "def f(\n    batch: int,\n    x: Float[Tensor, 'batch d'],\n) -> None:\n    pass\n"
    )
    s = sh(src)
    assert s.splitlines()[3] == "def f[_ts_batch: _ts_IntVar, _ts_d: _ts_IntVar]("
    assert s.splitlines()[4] == "    batch: _ts_Int[_ts_batch],"


@pytest.mark.parametrize(
    "src",
    [
        "import torch\ndef f(n: int\n    x = torch.zeros(n)\n",
        "import torch\nclass A(\n",
        "def (",
        "",
        "\n\n",
        "x = 1\n",
    ],
)
def test_broken_code_never_raises(src):
    res = rewrite(src)
    assert len(res.shadow.split("\n")) == len(src.split("\n"))


def test_broken_signature_does_not_block_others():
    src = "import torch\ndef bad(n: int, :\n    pass\ndef good(m: int):\n    pass\n"
    res = rewrite(src)
    assert "def good[_ts_m: _ts_IntVar](m: _ts_Int[_ts_m]):" in res.shadow


# ---------------- mapping ----------------

MAP_SOURCES = [(HERE / "fixtures" / f"{n}.py").read_text() for n in FIXTURES] + [
    "import torch  # r\u00e9sum\u00e9 \U0001f680\nfrom jaxtyping import Float\nfrom torch import Tensor\n"
    "def f(n: int, x: Float[Tensor, 'n d']):  # \u00e9\U0001f680\n    return x\n",
]


@pytest.mark.parametrize("src", MAP_SOURCES)
def test_every_unedited_char_round_trips(src):
    res = rewrite(src)
    pm = PositionMap(src, res.shadow, res.edits)
    U, S = src.split("\n"), res.shadow.split("\n")
    for ln, (u, s) in enumerate(zip(U, S)):
        spans = [(e.start, e.end) for e in res.edits if e.line == ln]
        for c, ch in enumerate(u):
            if any(a <= c < b for a, b in spans):
                continue
            sc = pm.user_to_shadow_cp(ln, c)
            assert s[sc] == ch
            assert pm.shadow_to_user_cp(ln, sc) == c


def test_utf16_positions():
    src = MAP_SOURCES[-1]
    res = rewrite(src)
    pm = PositionMap(src, res.shadow, res.edits)
    line = 3
    user_line = src.split("\n")[line]
    shadow_line = res.shadow.split("\n")[line]
    # the comment after the signature: position of "#" in UTF-16 units on each side
    u16 = user_line.index("#")  # no astral chars before it
    s16 = shadow_line.index("#")
    assert pm.user_to_shadow(line, u16) == s16
    assert pm.shadow_to_user(line, s16) == u16
    # last char (after an astral emoji) maps correctly in UTF-16 units
    end_u16 = len(user_line.encode("utf-16-le")) // 2
    end_s16 = len(shadow_line.encode("utf-16-le")) // 2
    assert pm.user_to_shadow(line, end_u16) == end_s16
    assert pm.shadow_to_user(line, end_s16) == end_u16


def test_generated_text_maps_to_its_user_span():
    src = "import torch\ndef f(n: int):\n    pass\n"
    res = rewrite(src)
    pm = PositionMap(src, res.shadow, res.edits)
    s = res.shadow.split("\n")[1]
    # inside `_ts_Int[_ts_n]` -> start of `int`
    assert pm.shadow_to_user_cp(1, s.index("_ts_Int[") + 3) == src.split("\n")[1].index("int")
    # inside inserted type params -> the insertion point (end of the name `f`)
    assert pm.shadow_to_user_cp(1, s.index("[_ts_n") + 2) == 5


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("(parameter) n_heads: Int[_ts_n_heads]", "(parameter) n_heads: int"),
        ("n: _ts_Int[_ts_n]", "n: int"),
        ("Attn[_ts_d_model, _ts_n_heads, 1000]", "Attn[d_model, n_heads, 1000]"),
        (
            "def forward[_ts_batch: IntVar, _ts_seq: IntVar](self, x: Tensor[[_ts_batch, _ts_seq, _ts_d_model]])",
            "def forward(self, x: Tensor[[batch, seq, d_model]])",
        ),
        ("def g[T, _ts_n: IntVar](n: Int[_ts_n])", "def g[T](n: int)"),
        ("Tensor[[_ts_seq_len * _ts_d_model]]", "Tensor[[seq_len * d_model]]"),
        ("x: Tensor[[]]", "x: Tensor[[]]"),
        ("handle_ts_x", "handle_ts_x"),
    ],
)
def test_clean_display(raw, clean):
    assert clean_display(raw) == clean


@pytest.mark.parametrize(
    "pkg,src,expected",
    [
        ("pkg.sub", "from .layers import Proj\n", "from pkg.sub.layers import Proj\n"),
        ("pkg.sub", "from ..util import x, y\n", "from pkg.util import x, y\n"),
        ("pkg.sub", "from . import layers\n", "from pkg.sub import layers\n"),
        ("pkg.sub", "from ...too_far import z\n", "from ...too_far import z\n"),
        ("pkg", "from .layers import (\n    Proj,\n)\n", "from pkg.layers import (\n    Proj,\n)\n"),
        (None, "from .layers import Proj\n", "from .layers import Proj\n"),
    ],
)
def test_relative_imports_made_absolute(pkg, src, expected):
    assert rewrite(src, pkg).shadow == expected


def test_relative_import_mapping_round_trips():
    src = "from .layers import Proj  # é\nimport torch\ndef f(n: int): pass\n"
    res = rewrite(src, "pkg")
    pm = PositionMap(src, res.shadow, res.edits)
    line0 = src.split("\n")[0]
    s0 = res.shadow.split("\n")[0]
    assert s0.startswith("from pkg.layers import Proj")
    assert pm.user_to_shadow_cp(0, line0.index("Proj")) == s0.index("Proj")
    assert pm.shadow_to_user_cp(0, s0.index("#")) == line0.index("#")


# ---------------- int-parameter inference ----------------
from tensi.rewrite import infer_int_params  # noqa: E402


def test_infer_from_size_calls_and_int_defaults():
    src = (
        "import torch.nn as nn\n"
        "class A(nn.Module):\n"
        "    def __init__(self, dim, heads=8, dropout=0.1, kernel_size=3, name='a', opt=None, depth=4):\n"
        "        self.l = nn.Linear(dim, dim * heads)\n"
        "        self.c = nn.Conv2d(dim, dim, kernel_size)\n"
        "        self.blocks = [nn.Identity() for _ in range(depth)]\n"
    )
    got = infer_int_params(src)[(3, "__init__")]
    # depth: int default (harmless, it is an int); kernel_size excluded (often a tuple)
    assert got == {"dim", "heads", "depth"}


def test_infer_follows_same_file_calls():
    src = (
        "import torch.nn as nn\n"
        "class Inner(nn.Module):\n"
        "    def __init__(self, d, h):\n"
        "        self.l = nn.Linear(d, h)\n"
        "class Outer(nn.Module):\n"
        "    def __init__(self, width, hidden, depth):\n"
        "        self.inner = Inner(width, h=hidden)\n"
        "        self.n = depth\n"
    )
    got = infer_int_params(src)
    assert got[(3, "__init__")] == {"d", "h"}
    assert got[(6, "__init__")] == {"width", "hidden"}


def test_infer_skips_none_float_str_defaults_and_syntax_errors():
    src = "import torch\ndef f(n=None, m=1.0, k='x', j=2):\n    return torch.zeros(n, m, k, j)\n"
    assert infer_int_params(src) == {(2, "f"): {"j"}}
    assert infer_int_params("def f(:\n") == {}


def test_inferred_params_rewritten_and_switchable():
    src = "import torch\ndef f(batch, d=4):\n    return torch.zeros(batch, d)\n"
    s = rewrite(src).shadow
    assert "def f[_ts_batch: _ts_IntVar, _ts_d: _ts_IntVar](batch: _ts_Int[_ts_batch], d: _ts_Int[_ts_d]=4):" in s
    assert rewrite(src, infer_ints=False).shadow == src


# ---------------- reassigned shaped parameters ----------------
REB = "import torch\nfrom torch import Tensor\nfrom jaxtyping import Float\n"


def test_rebind_on_first_simple_statement():
    s = sh(REB + "def f(x: Float[Tensor, 'n d']):\n    x = x.sum(0)\n    return x\n")
    assert "def f[_ts_n: _ts_IntVar, _ts_d: _ts_IntVar](_ts_x__p: Tensor[[_ts_n, _ts_d]]):" in s
    assert "    x = _ts_x__p; x = x.sum(0)" in s


def test_rebind_in_if_and_for_headers():
    s = sh(REB + "def f(x: Float[Tensor, 'n'], flag):\n    if flag:\n        x = x * 2\n    return x\n")
    assert "    if (x := _ts_x__p, flag)[1]:" in s
    s = sh(
        REB
        + "def f(x: Float[Tensor, 'n'], y: Float[Tensor, 'n']):\n    for i in range(2):\n        x, y = y, x\n    return x\n"
    )
    assert "    for i in (x := _ts_x__p, y := _ts_y__p, range(2))[2]:" in s


def test_unrebindable_reassigned_param_left_unshaped():
    res = rewrite(REB + "def f(x: Float[Tensor, 'n']):\n    while x.sum() > 0:\n        x = x - 1\n    return x\n")
    assert "def f(x: Tensor):" in res.shadow
    assert any("left unshaped" in n for n in res.notes)


def test_not_reassigned_param_untouched_by_rebind():
    s = sh(REB + "def f(x: Float[Tensor, 'n']):\n    y = x * 2\n    return y\n")
    assert "(x: Tensor[[_ts_n]])" in s and "__p" not in s


def test_nested_function_assignment_does_not_count():
    s = sh(REB + "def f(x: Float[Tensor, 'n']):\n    def g():\n        x = 1\n        return x\n    return x\n")
    assert "__p" not in s


# ---------------- config objects, ModuleDict, list indices, literal defaults ----------------
from tensi.rewrite import find_config_classes  # noqa: E402
from tensi.seeds import parse_generic, parse_tensor, translate, unpack_seeds  # noqa: E402

CFG = (
    "import torch\nimport torch.nn as nn\nfrom dataclasses import dataclass\n"
    "@dataclass\nclass Cfg:\n    width: int = 8\n    vocab: int = 10\n    p: float = 0.1\n"
    "class M(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n"
    "        self.l = nn.Linear(config.width, config.vocab)\n"
)


def test_config_class_detection():
    got = find_config_classes(CFG)
    assert got["Cfg"].int_fields == ("width", "vocab")
    assert find_config_classes("class Point:\n    a: int = 1\n") == {}
    assert "TrainArgs" in find_config_classes("class TrainArgs:\n    steps: int = 1\n")


def test_config_class_made_generic_and_param_typed():
    s = sh(CFG)
    assert "class Cfg[_ts_width: _ts_IntVar, _ts_vocab: _ts_IntVar]:" in s
    assert "    width: _ts_Int[_ts_width] = 8" in s and "    p: float = 0.1" in s
    assert "class M[_ts_width: _ts_IntVar, _ts_vocab: _ts_IntVar](nn.Module):" in s
    assert "def __init__(self, config: Cfg[_ts_width, _ts_vocab]):" in s


def test_config_from_another_module_is_imported_under_an_alias():
    from tensi.rewrite import ConfigClass

    src = "import torch\nimport torch.nn as nn\nclass M(nn.Module):\n    def __init__(self, cfg):\n        self.l = nn.Linear(cfg.width, 2)\n"
    s = sh_idx(src, {"Cfg": ConfigClass("Cfg", ("width",), "pkg.config")})
    assert "from pkg.config import Cfg as _ts_cfg_Cfg" in s.splitlines()[0]
    assert "def __init__(self, cfg: _ts_cfg_Cfg[_ts_width]):" in s


def sh_idx(src, idx):
    res = rewrite(src, config_index=idx)
    assert len(res.shadow.split("\n")) == len(src.split("\n"))
    parse_shadow(res.shadow)  # valid Python on every supported version
    return res.shadow


def test_module_dict_values_captured_and_accesses_redirected():
    src = (
        "import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self):\n"
        "        self.d = nn.ModuleDict(dict(\n            a=nn.Linear(2, 3),\n            b=nn.ReLU(),\n        ))\n"
        "        self.e = nn.ModuleDict({'c': nn.Linear(3, 4)})\n"
        "    def forward(self, x):\n        return self.e['c'](self.d.b(self.d.a(x)))\n"
    )
    s = sh(src)
    assert "a=(_ts_md_d__a := nn.Linear(2, 3))," in s
    assert "        )); self._ts_md_d__a = _ts_md_d__a; self._ts_md_d__b = _ts_md_d__b" in s
    assert "return self._ts_md_e__c(self._ts_md_d__b(self._ts_md_d__a(x)))" in s, s
    assert clean_display("(attribute) _ts_md_d__a: Linear[2, 3]") == "(attribute) a: Linear[2, 3]"


@pytest.mark.parametrize("idx,expected", [("[-1]", "-1:"), ("[0]", "0:1"), ("[2]", "2:3"), ("[-2]", "-2:-1")])
def test_single_int_list_index_becomes_slice(idx, expected):
    s = sh(f"import torch\nx = torch.zeros(2, 3)\ny = x[:, {idx}]\n")
    assert f"y = x[:, {expected}]" in s


def test_multi_element_list_index_untouched():
    src = "import torch\nx = torch.zeros(2, 3)\ny = x[:, [0, 2]]\n"
    assert sh(src) == src


def test_literal_float_and_bool_defaults_get_types():
    s = sh("import torch\ndef f(x, t=1.0, flag=False, n=None, k=-0.5):\n    return x\n")
    assert "def f(x, t: float=1.0, flag: bool=False, n=None, k: float=-0.5):" in s


def test_seeds_applied_with_fresh_and_class_dims():
    src = CFG + "    def forward(self, x):\n        return self.l(x)\n"
    s = rewrite(src, seeds={("M", "forward", "x"): ["~b", "~t", "width"]}).shadow
    assert "def forward[_ts_b: _ts_IntVar, _ts_t: _ts_IntVar](self, x: _ts_Tensor[[_ts_b, _ts_t, _ts_width]]):" in s


# ---------------- seed-solver helpers ----------------
@pytest.mark.parametrize(
    "t,dims",
    [
        ("torch.Tensor[[_ts_b, _ts_t, 3 * _ts_n]]", ["_ts_b", "_ts_t", "3 * _ts_n"]),
        ("Tensor[[_ts_b, _ts_n]] | Unknown", ["_ts_b", "_ts_n"]),
        ("Tensor[[_ts_b, 1]] | Tensor[[_ts_b, _ts_t]]", ["_ts_b", "int"]),
        ("Tensor[[*tuple[int, ...], int]]", None),
        ("Tensor", None),
        ("Tensor[[_ts_a]] | Tensor[[_ts_a, _ts_b]]", None),
    ],
)
def test_parse_tensor(t, dims):
    assert parse_tensor(t) == dims


def test_parse_generic():
    assert parse_generic("Block[_ts_a, 3 * _ts_b]") == ("Block", ["_ts_a", "3 * _ts_b"])
    assert parse_generic("Self@GPT") == ("GPT", None)
    assert parse_generic("Tensor | None") is None


def test_translate_into_callee_vocabulary():
    mapping = {"_ts_n_embd": "ndim"}
    assert translate(["_ts_b", "_ts_t", "_ts_n_embd"], mapping) == ["~b", "~t", "ndim"]
    assert translate(["7", "int", "3 * _ts_n_embd"], mapping) == ["7", "int", "3 * ndim"]
    assert translate(["_ts_t + _ts_s"], mapping) == ["int"]


def test_unpack_seeds():
    src = "class G:\n    def forward(self, idx, t2=None):\n        b, t = idx.size()\n        return idx\n"
    info = {"methods": {"G": {"forward": ["idx", "t2"]}}}
    assert unpack_seeds(src, info) == {("G", "forward", "idx"): ["~b", "~t"]}


def test_none_default_used_as_size_becomes_optional_int():
    s = sh(
        "import torch\ndef f(x, top_k=None):\n    if top_k is not None:\n        v, _ = torch.topk(x, min(top_k, 5))\n    return x\n"
    )
    assert "def f(x, top_k: int | None=None):" in s


def test_min_max_at_size_positions_wrapped_in_int():
    s = sh(
        "import torch\ndef f(x, k: int):\n    return torch.topk(x, min(k, x.size(-1))), x.view(max(k, 1), -1), min(k, 3)\n"
    )
    assert "torch.topk(x, int(min(k, x.size(-1))))" in s
    assert "x.view(int(max(k, 1)), -1)" in s
    assert s.rstrip().endswith("min(k, 3)")  # not a size position: untouched


# ---- zero-annotation idioms from vit-pytorch -------------------------------------------------


def test_parse_tensor_ignores_unshaped_union_members_when_seeding():
    # a loop's first pass through a module that is not seeded yet
    assert parse_tensor("Tensor[[_ts_b, int, _ts_dim]] | Tensor") == ["_ts_b", "int", "_ts_dim"]
    assert parse_tensor("Tensor[[_ts_b, int, _ts_dim]] | Tensor[[*tuple[int, ...], _ts_dim]]") == [
        "_ts_b",
        "int",
        "_ts_dim",
    ]
    assert parse_tensor("Tensor") is None
    assert parse_tensor("Tensor[[_ts_b, int]] | Tensor", strict=True) is None


def test_identity_fallback_picks_the_shaping_branch():
    src = (
        "from torch import nn\n"
        "class A(nn.Module):\n"
        "    def __init__(self, d: int, p: bool):\n"
        "        super().__init__()\n"
        "        self.o = nn.Linear(d, d) if p else nn.Identity()\n"
        "        self.q = nn.Identity() if p else nn.Linear(d, d)\n"
    )
    sh = rewrite(src).shadow
    assert "nn.Linear(d, d) if True else nn.Identity()" in sh
    assert "nn.Identity() if False else nn.Linear(d, d)" in sh


def test_literal_choice_becomes_int():
    sh = rewrite("def f(pool):\n    n = 1 if pool == 'cls' else 0\n    return n\n").shadow
    assert "n = int(1 if pool == 'cls' else 0)" in sh


def test_module_list_append_keeps_module_types():
    src = (
        "from torch import nn\n"
        "class T(nn.Module):\n"
        "    def __init__(self, d: int, depth: int):\n"
        "        super().__init__()\n"
        "        self.layers = nn.ModuleList([])\n"
        "        for _ in range(depth):\n"
        "            self.layers.append(nn.ModuleList([nn.Linear(d, d), nn.ReLU()]))\n"
        "    def forward(self, x):\n"
        "        for lin, act in self.layers:\n"
        "            x = act(lin(x))\n"
        "        return x\n"
    )
    sh = rewrite(src).shadow
    assert "self._ts_ml_layers = [(((nn.Linear(d, d), nn.ReLU())))]" in sh
    assert "for lin, act in self._ts_ml_layers:" in sh
    assert len(sh.split("\n")) == len(src.split("\n"))


def test_param_rebound_to_a_tuple_is_not_an_int_dim():
    src = (
        "def cast_tuple(v, n): return v if isinstance(v, tuple) else (v,) * n\n"
        "def f(window=7, depth=2):\n"
        "    window = cast_tuple(window, 4)\n"
        "    depth = depth * 2\n"
        "    return window, depth\n"
    )
    found = infer_int_params(src)
    assert found == {(2, "f"): {"depth"}}


def test_map_lambda_over_a_tuple_is_spelled_out():
    src = "def f(a, b):\n    (a1, a2), (b1, b2) = map(lambda t: (t[:, :1], t[:, 1:]), (a, b))\n    return a1\n"
    sh = rewrite(src).shadow
    assert "(a1, a2), (b1, b2) = (((a[:, :1], a[:, 1:])), ((b[:, :1], b[:, 1:])))" in sh


def test_exists_helper_becomes_a_none_check():
    src = (
        "def exists(val):\n"
        "    return val is not None\n"
        "def f(mask, act):\n"
        "    if exists(mask):\n"
        "        mask = mask[:, None]\n"
        "    g = act() if not exists(act) else None\n"
        "    return mask\n"
    )
    sh = rewrite(src).shadow
    assert "    if (mask is not None):" in sh
    assert "if not (act is not None) else None" in sh


def test_pair_helper_calls_are_inlined_only_for_pair_shaped_helpers():
    from tensi.rewrite import pair_helpers

    src = (
        "def pair(t):\n    return t if isinstance(t, tuple) else (t, t)\n\n"
        "def other(t):\n    return (t, 1)\n\n"
        "a, b = pair(p)\nc, d = other(p)\n"
    )
    assert pair_helpers(ast.parse(src)) == {"pair": 2}


def test_clean_display_hides_variadic_tuple():
    assert clean_display("Tensor[[*tuple[int, ...], _ts_dim]]") == "Tensor[[..., dim]]"
