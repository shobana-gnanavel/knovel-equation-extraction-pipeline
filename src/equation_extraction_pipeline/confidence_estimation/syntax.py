"""Syntax confidence estimator — deterministic LaTeX structural validation."""

from __future__ import annotations

import re

from .config import ConfidenceConfig
from .models import Issue, SyntaxDetails

# Partial whitelist — covers amsmath, amssymb, mathtools, physics, siunitx essentials.
_KNOWN_COMMANDS: frozenset[str] = frozenset({
    "frac", "dfrac", "tfrac", "cfrac", "binom", "dbinom", "tbinom",
    "sqrt", "sum", "prod", "int", "oint", "iint", "iiint",
    "lim", "inf", "sup", "max", "min", "sin", "cos", "tan", "cot",
    "sec", "csc", "log", "ln", "exp", "det", "dim", "ker", "gcd",
    "hom", "arg", "deg", "Pr", "sinh", "cosh", "tanh",
    "left", "right", "big", "Big", "bigg", "Bigg",
    "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
    "zeta", "eta", "theta", "vartheta", "iota", "kappa", "lambda",
    "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma",
    "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
    "Upsilon", "Phi", "Psi", "Omega",
    "partial", "nabla", "infty", "pm", "mp", "times", "div",
    "cdot", "cdots", "ldots", "vdots", "ddots",
    "leq", "geq", "neq", "approx", "sim", "simeq", "cong", "equiv",
    "propto", "ll", "gg", "subset", "supset", "subseteq", "supseteq",
    "in", "notin", "cup", "cap", "wedge", "vee", "neg", "forall", "exists",
    "to", "leftarrow", "rightarrow", "Rightarrow", "Leftarrow",
    "leftrightarrow", "Leftrightarrow", "mapsto", "longrightarrow",
    "longleftarrow", "Longrightarrow", "Longleftarrow",
    "hat", "bar", "dot", "ddot", "tilde", "vec", "overline", "underline",
    "overbrace", "underbrace", "widehat", "widetilde", "overrightarrow",
    "text", "mathrm", "mathit", "mathbf", "mathsf", "mathbb", "mathcal",
    "mathfrak", "boldsymbol", "operatorname", "mbox",
    "begin", "end", "label", "ref", "tag", "notag",
    "quad", "qquad", "hspace", "vspace", "space", "enspace",
    "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
    "limits", "nolimits", "mathop",
    "pm", "mp", "ast", "star", "circ", "bullet", "dagger", "ddagger",
    "oplus", "ominus", "otimes", "oslash", "odot", "boxplus",
    "rightleftharpoons", "leftrightarrows", "angle", "measuredangle",
    "perp", "parallel", "nmid", "mid", "vert", "Vert",
    "langle", "rangle", "lfloor", "rfloor", "lceil", "rceil",
    "not", "ne", "le", "ge", "ll", "gg",
    "over", "above", "atop", "choose",
    "underset", "overset", "stackrel",
    "xleftarrow", "xrightarrow",
    "pmod", "bmod",
    "Re", "Im", "wp", "ell", "hbar", "imath", "jmath",
    "triangle", "square", "lozenge", "clubsuit", "diamondsuit",
    "heartsuit", "spadesuit", "sharp", "flat", "natural",
    "checkmark", "maltese", "dagger", "ddagger",
    "therefore", "because", "qed",
})

_KNOWN_ENVIRONMENTS: frozenset[str] = frozenset({
    "equation", "equation*", "align", "align*", "aligned",
    "gather", "gather*", "gathered", "multline", "multline*",
    "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
    "Bmatrix", "smallmatrix",
    "cases", "dcases", "split", "subequations", "flalign", "flalign*",
    "alignat", "alignat*",
})

_CMD_RE = re.compile(r"\\([a-zA-Z]+)")
_ENV_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_ENV_END_RE = re.compile(r"\\end\{([^}]+)\}")
_FRAC_RE = re.compile(r"\\(?:d|t|c)?frac")
_LEFT_RE = re.compile(r"\\left\s*(?:[({[\|.]|\\[a-zA-Z]+)")
_RIGHT_RE = re.compile(r"\\right\s*(?:[)}\[\|.]|\\[a-zA-Z]+)")


def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    depth = 0
    for ch in text:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _consume_group(latex: str, start: int) -> int | None:
    i = start
    while i < len(latex) and latex[i] == " ":
        i += 1
    if i >= len(latex) or latex[i] != "{":
        return None
    depth = 0
    while i < len(latex):
        if latex[i] == "{":
            depth += 1
        elif latex[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _check_frac_args(latex: str) -> bool:
    for match in _FRAC_RE.finditer(latex):
        pos = match.end()
        end1 = _consume_group(latex, pos)
        if end1 is None:
            return False
        end2 = _consume_group(latex, end1)
        if end2 is None:
            return False
    return True


def _check_scripts(latex: str) -> bool:
    i = 0
    while i < len(latex):
        if latex[i] in ("^", "_"):
            j = i + 1
            while j < len(latex) and latex[j] == " ":
                j += 1
            if j >= len(latex):
                return False
            # Must be followed by { or \ or an alphanumeric
            if latex[j] not in ("{", "\\") and not latex[j].isalnum():
                return False
        i += 1
    return True


def _max_depth(latex: str) -> int:
    depth = max_d = 0
    for ch in latex:
        if ch == "{":
            depth += 1
            max_d = max(max_d, depth)
        elif ch == "}":
            depth -= 1
    return max_d


_ISSUE_MESSAGES: dict[str, str] = {
    "balanced_braces":      "Unbalanced curly braces detected.",
    "balanced_brackets":    "Unbalanced square brackets detected.",
    "balanced_parens":      "Unbalanced parentheses detected.",
    "left_right_matched":   "Mismatched \\left / \\right delimiters.",
    "environments_closed":  "\\begin / \\end environment mismatch.",
    "frac_has_two_args":    "\\frac with fewer than 2 arguments detected.",
    "scripts_have_args":    "Superscript or subscript without argument.",
    "commands_are_known":   "Unknown LaTeX commands found.",
    "depth_within_limit":   "Nesting depth exceeds maximum limit.",
}


def estimate_syntax(
    *,
    latex: str,
    config: ConfidenceConfig,
) -> tuple[float, SyntaxDetails, list[Issue]]:
    issues: list[Issue] = []
    details = SyntaxDetails()

    if not latex or not latex.strip():
        issues.append(Issue("EMPTY_LATEX", "error", "LaTeX string is empty.", "syntax"))
        return 0.0, details, issues

    checks: dict[str, bool] = {}

    checks["balanced_braces"]    = _balanced(latex, "{", "}")
    checks["balanced_brackets"]  = _balanced(latex, "[", "]")
    checks["balanced_parens"]    = _balanced(latex, "(", ")")
    checks["left_right_matched"] = len(_LEFT_RE.findall(latex)) == len(_RIGHT_RE.findall(latex))

    begins = _ENV_BEGIN_RE.findall(latex)
    ends   = _ENV_END_RE.findall(latex)
    checks["environments_closed"] = begins == ends

    checks["frac_has_two_args"] = _check_frac_args(latex)
    checks["scripts_have_args"] = _check_scripts(latex)

    commands = _CMD_RE.findall(latex)
    unknown  = [c for c in commands if c not in _KNOWN_COMMANDS]
    checks["commands_are_known"] = len(unknown) == 0

    checks["depth_within_limit"] = _max_depth(latex) <= config.max_nesting_depth

    details.balanced_braces      = checks["balanced_braces"]
    details.balanced_brackets    = checks["balanced_brackets"]
    details.balanced_parens      = checks["balanced_parens"]
    details.left_right_matched   = checks["left_right_matched"]
    details.environments_closed  = checks["environments_closed"]
    details.frac_has_two_args    = checks["frac_has_two_args"]
    details.scripts_have_args    = checks["scripts_have_args"]
    details.commands_are_known   = checks["commands_are_known"]
    details.depth_within_limit   = checks["depth_within_limit"]
    details.unknown_commands     = list(set(unknown))

    for name, passed in checks.items():
        if not passed:
            msg = _ISSUE_MESSAGES.get(name, f"Check {name!r} failed.")
            if name == "commands_are_known" and unknown:
                msg = f"Unknown commands: {list(set(unknown))[:5]}."
            issues.append(Issue(name.upper(), "warning", msg, "syntax"))

    n_total  = len(checks)
    n_passed = sum(checks.values())
    score    = n_passed / n_total if n_total else 0.0
    return score, details, issues
