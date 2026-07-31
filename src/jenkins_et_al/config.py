"""Shared presentation constants used across the paper's figures.

Every value here is transcribed verbatim from the notebooks/scripts at git
commit 2d19f3a (see the comment above each constant for the source cell).
Nothing here is a new choice except where a comment says so explicitly.
"""

# ---------------------------------------------------------------------------
# Stimulus identity: raw code -> display label -> plot color
# ---------------------------------------------------------------------------
# Raw stimulus codes as they appear in the underlying dataframes. The food
# extract odor is spelled two ways across the original codebase: "fex1" (and
# "fex2"/"fex3" for its lower concentrations) in the head-fixed behaviour
# notebooks (e.g. notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb,
# cell 3), and "fex_1" in every neural notebook and regression script (e.g.
# notebooks/neural/wholebrain_population_analysis.ipynb, cell 3). Per
# instruction, both spellings are kept as aliases mapping to the same label
# below rather than collapsed to one canonical form.
#
# Source: identical in notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb
# cell 3, notebooks/behaviour_head_fixed/bout_type_frequency_LMM.ipynb cell 1,
# and notebooks/behaviour_head_fixed/reorientation_posture_LMM.ipynb cell 1.
STIMULUS_LABELS = {
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
# Note: notebooks/neural/wholebrain_population_analysis.ipynb (cell 4) uses a
# different label pair for the lower-concentration food extract dilutions
# ("Fex 1:100" / "Fex 1:1000" for fex_2/fex_3, vs. "Fex Int." / "Fex Low"
# above). That notebook's variant is specific to its own concentration-series
# and raster plots and is deliberately not unified here -- resolve it if/when
# wholebrain_population_analysis.ipynb is ported.

# Source: identical in the same three notebooks as STIMULUS_LABELS above.
STIMULUS_COLORS = {
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

# The 7 high-concentration stimuli, grouped by concentration tier as used to
# select `key_stimuli` in the head-fixed behaviour notebooks.
# Source: notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb, cell 3.
STIMULUS_GROUPS = {
    "high": ("ade", "cad_2.5mm", "ph4.5", "qui_2.5mm", "fex1", "kw", "pro_2.5mm"),
    "intermediate": ("cad_250um", "qui_250um", "fex2", "pro_250um"),
    "low": ("cad_25um", "qui_25um", "fex3", "pro_25um"),
}

# Default display order for the 7 high-concentration stimuli. The original
# notebooks used at least 4 different orderings for this same set (see the
# survey); this one matches
# notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb cell 3 and is the
# same valence grouping (aversive-then-appetitive) used by
# bout_type_frequency_LMM.ipynb and reorientation_posture_LMM.ipynb. Treat
# this as a default, not a forced global order: plotting functions should
# accept an explicit `order` argument so a call site can reproduce a
# notebook that used a different order (e.g.
# notebooks/behaviour_head_fixed/bout_frequency_PSTH.ipynb's
# appetitive-first order).
STIMULUS_ORDER_HIGH_CONC = (
    "ade", "cad_2.5mm", "ph4.5", "qui_2.5mm", "fex1", "kw", "pro_2.5mm",
)

# ---------------------------------------------------------------------------
# Valence (appetitive/aversive) color and grouping
# ---------------------------------------------------------------------------
# The original notebooks used three different color pairs for positive vs.
# negative valence (limegreen/deeppink, green/magenta, green/red) -- flagged
# to the user in the survey. Resolved 2026-07-31: green/magenta is canonical
# going forward.
# Source: notebooks/behaviour_head_fixed/reorientation_posture_LMM.ipynb,
# cell 5; notebooks/behaviour_head_fixed/bout_frequency_PSTH.ipynb (hard-coded
# in its valence plot); notebooks/behaviour_head_fixed/bout_type_frequency_LMM.ipynb,
# cell 13.
VALENCE_COLORS = {"positive": "green", "negative": "magenta"}

# Source: notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb, cell 3
# (there keyed "POS"/"NEG"; renamed here to match VALENCE_COLORS' keys).
VALENCE_LABELS = {"positive": "Appetitive", "negative": "Aversive"}

# Raw stimulus codes grouped by valence, across all three concentration
# tiers. Source: notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb,
# cell 3 (`positive_stims` / `negative_stims`).
POSITIVE_STIMULI = ("fex1", "kw", "pro_2.5mm", "fex2", "pro_250um", "fex3", "pro_25um")
NEGATIVE_STIMULI = (
    "ade", "cad_2.5mm", "ph4.5", "qui_2.5mm",
    "cad_250um", "qui_250um", "cad_25um", "qui_25um",
)

# ---------------------------------------------------------------------------
# Brain area color palettes and shorthand codes
# ---------------------------------------------------------------------------
# Forebrain area palette, identical in both source notebooks.
# Source: notebooks/neural/lifetime_sparseness.ipynb, cell 22
# (`plot_sparseness_boxplot`'s local `palette`), and
# notebooks/neural/response_latency.ipynb, cell 4 (`PALETTE`).
FOREBRAIN_AREA_COLORS = {
    "OB": "#16327F", "OE": "#000000", "Pal": "#377DA3", "SubP": "#58C8FE",
    "PT": "#EF4066", "dHb": "#663200", "vHb": "#933264", "POA": "#8C8C00",
    "PTh": "#1E7F1D", "EmT": "#6CDA60", "HR": "#D9D900", "HI": "#EB7F25",
}

# Hindbrain area palette (18 areas; most have no assigned color in the
# original and are plotted black).
# Source: notebooks/neural/template_matching_classification.ipynb, cell 29
# (`plot_classification_accuracy_boxplot`'s local `palette`).
HINDBRAIN_AREA_COLORS = {
    "OB": "#16327F", "PTec": "black", "Tect": "black", "NI": "black", "Cb": "black",
    "SGN": "#6F00F7", "sDMO": "black", "sRaphe": "black", "sVMO": "black",
    "aTriMN": "black", "itDMO": "black", "itVMO": "black", "iRaphe": "black",
    "fMN": "#8A02B7", "ifDMO": "black", "ifVMO": "black",
    "vMN": "#D9AFE7", "VSL": "#654288",
}

# Reference area used throughout the "boxplot vs. reference area" plotting
# pattern (lifetime sparseness, response latency, classification accuracy).
# Source: notebooks/neural/response_latency.ipynb, cell 4 (`REFERENCE_AREA`).
REFERENCE_AREA = "olfactory_bulb"

# Full brain-area name -> shorthand code, mirrored verbatim from the shared
# external file every neural notebook that uses shorthand codes loads at
# runtime, so the package doesn't require that drive to be mounted just to
# look up a shorthand.
# Source: /Volumes/LaCie/larval_HuC/imaging/brain_map_values/area_shorthand_dict.json,
# as loaded by e.g. notebooks/neural/lifetime_sparseness.ipynb,
# notebooks/neural/noise_correlations_area_pair_analysis.ipynb, and
# notebooks/neural/template_matching_classification.ipynb.
AREA_SHORTHAND = {
    "anterior_trigeminal_mn": "aTriMN",
    "caudal_hypothalamus": "HC",
    "cerebellum": "Cb",
    "dorsal_habenula": "dHb",
    "dorsal_thalamus": "dTh",
    "eminentia_thalami": "EmT",
    "facial_mn": "fMN",
    "glossopharyngeal": "gPh",
    "LRN": "LRN",
    "IPN": "IPN",
    "inferior_olive": "IO",
    "inferior_raphe": "iRaphe",
    "inferior_ventral_med_ob": "ifVMO",
    "intermediate_dorsal_med_ob": "itDMO",
    "intermediate_hypothalamus": "HI",
    "intermediate_ventral_med_ob": "itVMO",
    "locus_coeruleus": "LC",
    "nucleus_isthmi": "NI",
    "olfactory_bulb": "OB",
    "olfactory_epithelium": "OE",
    "pallium": "Pal",
    "posterior_trigeminal_mn": "pTriMN",
    "posterior_tuberculum": "PT",
    "preoptic": "POA",
    "pretectum": "PTec",
    "prethalamus": "PTh",
    "rostral_hypothalamus": "HR",
    "subpallium": "SubP",
    "superior_dorsal_med_ob": "sDMO",
    "superior_raphe": "sRaphe",
    "superior_ventral_med_ob": "sVMO",
    "tectum": "Tect",
    "tegmentum": "Teg",
    "tori": "Tori",
    "trigeminal_ganglion": "TriG",
    "vagal_sensory_lobe": "VSL",
    "vagus_mn": "vMN",
    "ventral_habenula": "vHb",
    "inferior_dorsal_med_ob": "ifDMO",
    "secondary_gustatory_nucleus": "SGN",
    "nMLF": "nMLF",
}
