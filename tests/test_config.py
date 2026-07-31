"""Regression tests for jenkins_et_al.config.

These pin the constants to the exact values transcribed from the source
notebooks at commit 2d19f3a (see config.py for per-constant provenance
comments), so an accidental edit to config.py is caught rather than silently
changing every figure that reads from it.
"""

from jenkins_et_al import config


def test_stimulus_labels():
    assert config.STIMULUS_LABELS == {
        "ade": "Ade",
        "cad_2.5mm": "Cad",
        "fex1": "Fex",
        "fex_1": "Fex",
        "kw": "Kin",
        "ph4.5": "HCl",
        "pro_2.5mm": "Pro",
        "qui_2.5mm": "Qun",
        "cad_250um": "Cad Int.",
        "fex2": "Fex Int.",
        "pro_250um": "Pro Int.",
        "qui_250um": "Qun Int.",
        "cad_25um": "Cad Low",
        "fex3": "Fex Low",
        "pro_25um": "Pro Low",
        "qui_25um": "Qun Low",
    }


def test_stimulus_colors():
    assert config.STIMULUS_COLORS == {
        "Ade": "plum",
        "Cad": "mediumorchid",
        "Fex": "limegreen",
        "Kin": "forestgreen",
        "HCl": "fuchsia",
        "Pro": "olive",
        "Qun": "deeppink",
        "Cad Int.": "violet",
        "Fex Int.": "seagreen",
        "Pro Int.": "olivedrab",
        "Qun Int.": "palevioletred",
        "Cad Low": "plum",
        "Fex Low": "springgreen",
        "Pro Low": "yellowgreen",
        "Qun Low": "pink",
    }


def test_stimulus_groups_partition_stimulus_labels():
    all_grouped = set(config.STIMULUS_GROUPS["high"]) \
        | set(config.STIMULUS_GROUPS["intermediate"]) \
        | set(config.STIMULUS_GROUPS["low"])
    non_alias_codes = {"ade", "cad_2.5mm", "fex1", "kw", "ph4.5", "pro_2.5mm",
                        "qui_2.5mm", "cad_250um", "fex2", "pro_250um", "qui_250um",
                        "cad_25um", "fex3", "pro_25um", "qui_25um"}
    assert all_grouped == non_alias_codes


def test_stimulus_order_high_conc_matches_high_group():
    assert set(config.STIMULUS_ORDER_HIGH_CONC) == set(config.STIMULUS_GROUPS["high"])
    assert len(config.STIMULUS_ORDER_HIGH_CONC) == len(set(config.STIMULUS_ORDER_HIGH_CONC))


def test_valence_colors_is_canonical_pair_b():
    assert config.VALENCE_COLORS == {"positive": "green", "negative": "magenta"}


def test_valence_labels():
    assert config.VALENCE_LABELS == {"positive": "Appetitive", "negative": "Aversive"}


def test_positive_and_negative_stimuli_are_disjoint_and_known():
    positive = set(config.POSITIVE_STIMULI)
    negative = set(config.NEGATIVE_STIMULI)
    assert positive.isdisjoint(negative)
    assert positive | negative <= set(config.STIMULUS_LABELS)


def test_forebrain_area_colors():
    assert config.FOREBRAIN_AREA_COLORS == {
        "OB": "#16327F", "OE": "#000000", "Pal": "#377DA3", "SubP": "#58C8FE",
        "PT": "#EF4066", "dHb": "#663200", "vHb": "#933264", "POA": "#8C8C00",
        "PTh": "#1E7F1D", "EmT": "#6CDA60", "HR": "#D9D900", "HI": "#EB7F25",
    }


def test_hindbrain_area_colors():
    assert config.HINDBRAIN_AREA_COLORS == {
        "OB": "#16327F", "PTec": "black", "Tect": "black", "NI": "black", "Cb": "black",
        "SGN": "#6F00F7", "sDMO": "black", "sRaphe": "black", "sVMO": "black",
        "aTriMN": "black", "itDMO": "black", "itVMO": "black", "iRaphe": "black",
        "fMN": "#8A02B7", "ifDMO": "black", "ifVMO": "black",
        "vMN": "#D9AFE7", "VSL": "#654288",
    }


def test_forebrain_and_hindbrain_agree_on_ob():
    assert config.FOREBRAIN_AREA_COLORS["OB"] == config.HINDBRAIN_AREA_COLORS["OB"]


def test_reference_area():
    assert config.REFERENCE_AREA == "olfactory_bulb"


def test_area_shorthand_has_41_entries_and_matches_reference_area():
    assert len(config.AREA_SHORTHAND) == 41
    assert config.AREA_SHORTHAND[config.REFERENCE_AREA] == "OB"
    assert config.AREA_SHORTHAND["olfactory_epithelium"] == "OE"
