import re

# Precompile regexes 
_RE_FEATURE = re.compile(r"^(?P<wl>\d{3,5})(?P<desc>.*)W(?P<kind>DEP|POS)?$")
_RE_KMEANS = re.compile(r"^kmeans-(\d+)-(\d+)(CLUSTERS|INDEX)$")
_RE_MINMAP_RANGE = re.compile(
    r"^MinMap-([0-9.]+)-([0-9.]+)-(SAM|MSAM|pearson)-(.+?)(CONF|INDEX|LEGEND)$"
)
_RE_MINMAP_STD = re.compile(r"^MinMap-(SAM|MSAM|pearson)-(.+?)(CONF|INDEX|LEGEND)$")
_RE_MINMAP_MULTI = re.compile(r"^MinMapMulti-(sam)-(.+?)(CONF|INDEX|LEGEND|WINDOW)$", re.IGNORECASE)
_RE_MINMAP_DOWNHOLE = re.compile(r"^MinMap-(SAM|MSAM|pearson)-(.+?)(FRACTIONS|DOM-MIN)(-PROF)?$")


def _fmt_range_num(x: str) -> str:
    """Format '2100.0' -> '2100', keep '2100.5' -> '2100.5'."""
    try:
        f = float(x)
    except Exception:
        return x
    if f.is_integer():
        return str(int(f))
    s = f"{f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _pretty_desc(desc: str) -> str:
    """
    Normalise descriptor text for display:
    - trim
    - convert underscores / hyphens to spaces
    - collapse whitespace
    - lowercase
    """
    desc = (desc or "").strip()
    if not desc:
        return ""
    desc = re.sub(r"[_\-]+", " ", desc)
    desc = re.sub(r"\s+", " ", desc)
    return desc.lower()


def gen_display_text(key: str) -> str:
    """
    Generate a human-readable display label from an internal dataset key.
    Falls back to returning `key` when no rule applies.
    """

    # ---- Exact mappings ----
    exact = {
        "bands": "",
        "cropped": "",
        "mask": "",
        "metadata": "",
        "savgol": "",
        "savgol_cr": "",
        "segments": "",
        "stats": "",
        "DholeAverage": "Downhole image",
        "DholeDepths": "Downhole depths",
        "DholeMask": "Downhole mask",
        "AvSpectra": "Downhole Spectral Profile",
        "depths": "Per Pixel Depth",
    }
    if key in exact:
        return exact[key] or key

    
    m = _RE_FEATURE.fullmatch(key)
    if m:
        wl = m.group("wl")
        desc = _pretty_desc(m.group("desc"))
        kind = m.group("kind")

        if kind == "DEP":
            base = f"{wl}nm feature depth"
        elif kind == "POS":
            base = f"{wl}nm feature position"
        else:
            base = f"{wl}nm feature"

        if desc:
            base += f" ({desc})"
        return base

    # ---- k-means outputs ----
    m = _RE_KMEANS.fullmatch(key)
    if m:
        runs, clusters, kind = m.groups()
        return (
            f"k-means cluster centres ({runs}, {clusters})"
            if kind == "CLUSTERS"
            else f"k-means image ({runs}, {clusters})"
        )

    # ---- Range mineral maps: MinMap-0.0-2100.0-pearson-filtrosCONF ----
    m = _RE_MINMAP_RANGE.fullmatch(key)
    if m:
        lo, hi, method, collection, kind = m.groups()
        lo_s, hi_s = _fmt_range_num(lo), _fmt_range_num(hi)

        method_disp = method.upper() if method != "pearson" else "Pearson"
        kind_disp = {
            "CONF": "confidence map",
            "INDEX": "mineral map",
            "LEGEND": "legend",
        }[kind]

        return f"{lo_s}-{hi_s} range {method_disp} {kind_disp} ({collection} collection)"

    # ---- Standard mineral maps: MinMap-pearson-1aCONF ----
    m = _RE_MINMAP_STD.fullmatch(key)
    if m:
        method, collection, kind = m.groups()
        method_disp = method.upper() if method != "pearson" else "Pearson"
        kind_disp = {
            "CONF": "confidence map",
            "INDEX": "mineral map",
            "LEGEND": "legend",
        }[kind]
        return f"{method_disp} {kind_disp} ({collection} collection)"

    # ---- Multi-range SAM maps: MinMapMulti-sam-filtros-montWINDOW ----
    m = _RE_MINMAP_MULTI.fullmatch(key)
    if m:
        _sam, collection, kind = m.groups()
        kind_disp = {
            "CONF": "confidence map",
            "INDEX": "mineral map",
            "LEGEND": "legend",
            "WINDOW": "window map",
        }[kind.upper()]
        return f"SAM multi-range {kind_disp} ({collection} collection)"

    # ---- Downhole derived products: FRACTIONS / DOM-MIN (+ optional -PROF) ----
    m = _RE_MINMAP_DOWNHOLE.fullmatch(key)
    if m:
        method, collection, kind, prof = m.groups()
        method_disp = method.upper() if method != "pearson" else "Pearson"
        derived = "profile derived" if prof else "box derived"

        if kind == "FRACTIONS":
            return f"{method_disp} mineral fractions downhole ({collection} collection, {derived})"
        else:  # DOM-MIN
            return f"{method_disp} dominant mineral downhole ({collection} collection, {derived})"

    # ---- Fallback ----
    return key