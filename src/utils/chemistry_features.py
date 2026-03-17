from __future__ import annotations

import re
from typing import Any

RDKIT_DESCRIPTOR_SNAPSHOT = [
    "MaxEStateIndex",
    "MinEStateIndex",
    "MaxAbsEStateIndex",
    "MinAbsEStateIndex",
    "qed",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]

RDKIT_DESCRIPTOR_LOOKUP = {name.lower(): name for name in RDKIT_DESCRIPTOR_SNAPSHOT}

COUNT_FEATURE_ALIASES = {
    "AtomCount": ["AtomCount", "atom count", "atom counts", "num atoms", "number of atoms", "atom_count"],
    "BondCount": ["BondCount", "bond count", "bond counts", "num bonds", "number of bonds", "bond_count"],
    "HeavyAtomCount": ["HeavyAtomCount", "heavy atom count", "heavy atoms", "heavyatomcount"],
    "NumHeteroatoms": ["NumHeteroatoms", "heteroatom count", "hetero atom count", "heteroatoms"],
    "RingCount": ["RingCount", "ring count", "ring counts", "num rings", "number of rings"],
    "NumRotatableBonds": ["NumRotatableBonds", "rotatable bond count", "rotatable bonds", "num rotatable bonds"],
    "NumAliphaticRings": ["NumAliphaticRings", "aliphatic ring count", "aliphatic rings"],
    "NumAromaticRings": ["NumAromaticRings", "aromatic ring count", "aromatic rings"],
    "NumSaturatedRings": ["NumSaturatedRings", "saturated ring count", "saturated rings"],
    "NumHAcceptors": ["NumHAcceptors", "h bond acceptor count", "hydrogen bond acceptors", "acceptor count"],
    "NumHDonors": ["NumHDonors", "h bond donor count", "hydrogen bond donors", "donor count"],
    "NHOHCount": ["NHOHCount", "nhoh count"],
    "NOCount": ["NOCount", "no count", "n/o count"],
}

FINGERPRINT_FAMILY_ALIASES = {
    "morgan": ["morgan", "ecfp", "extended connectivity", "extended-connectivity", "circular fingerprint", "circular fingerprints", "fcfp"],
    "maccs": ["maccs", "maccs keys"],
    "atom_pair": ["atom pair", "atom-pair"],
    "topological_torsion": ["topological torsion", "torsion fingerprint"],
    "rdkit": ["rdkit fingerprint", "rdkitfp", "daylight-like fingerprint"],
}

EXTERNAL_FEATURE_TOOL_ALIASES = {
    "Mordred": ["mordred"],
    "PaDEL": ["padel", "pa del"],
    "Dragon": ["dragon"],
    "MOE": ["moe"],
    "alvaDesc": ["alvadesc", "alva desc"],
}

DESCRIPTOR_SIGNAL_TERMS = [
    "descriptor",
    "descriptors",
    "rdkit",
    "physicochemical",
    "constitutional",
    "topological descriptor",
    "electronic descriptor",
]

EXACT_FEATURE_ALIASES = {
    "cmpdname": ["cmpdname", "compound name", "compound names"],
    "mw": ["mw", "molecular weight"],
    "mf": ["mf", "molecular formula"],
    "polararea": ["polararea", "polar area", "polar surface area"],
    "heavycnt": ["heavycnt", "heavy atom count", "heavy atom counts"],
    "hbondacc": ["hbondacc", "h bond acceptor", "h-bond acceptor", "hydrogen bond acceptor"],
    "iso smiles": ["iso smiles", "iso-smiles", "isosmiles", "canonical smiles"],
    "C number": ["c number", "carbon number", "number of carbon atoms"],
    "N number": ["n number", "nitrogen number", "number of nitrogen atoms"],
    "O number": ["o number", "oxygen number", "number of oxygen atoms"],
    "C/N/O number": ["c/n/o number", "c n o number", "c, n, o number"],
    "side chain number": ["side chain number", "number of side chains", "side-chain number"],
    "hydrocarbon": ["hydrocarbon", "hydrocarbons"],
    "alcohol": ["alcohol", "alcohols"],
    "amine": ["amine", "amines"],
}

RETAINED_INPUT_FEATURES = {
    "cmpdname",
    "mw",
    "mf",
    "polararea",
    "heavycnt",
    "hbondacc",
    "iso smiles",
}

DERIVED_FEATURE_NAMES = {
    "C number",
    "N number",
    "O number",
    "C/N/O number",
    "side chain number",
}

CLASS_LABEL_NAMES = {
    "hydrocarbon",
    "alcohol",
    "amine",
}

FINGERPRINT_SIGNAL_TERMS = ["fingerprint", "fingerprints", "ecfp", "morgan", "maccs", "atom pair", "topological torsion"]


def merge_unique(*collections: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for collection in collections:
        if collection is None:
            continue
        if isinstance(collection, str):
            items = [collection]
        else:
            items = list(collection)
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return merge_unique([str(item).strip() for item in value if str(item).strip()])
    if isinstance(value, tuple):
        return merge_unique([str(item).strip() for item in value if str(item).strip()])
    text = str(value).strip()
    if not text or text == "Not found":
        return []
    parts = re.split(r"[,;\n]+", text)
    return merge_unique([part.strip() for part in parts if part.strip()])


def _normalize_feature_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip(" .,:;()[]{}"))
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for canonical_name, aliases in EXACT_FEATURE_ALIASES.items():
        if lowered == canonical_name.lower():
            return canonical_name
        if lowered in {alias.lower() for alias in aliases}:
            return canonical_name
    if len(cleaned) < 3:
        return ""
    return cleaned


def _compact_feature_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def normalize_exact_feature_terms(value: Any) -> list[str]:
    resolved: list[str] = []
    alias_lookup: dict[str, str] = {}
    for canonical_name, aliases in EXACT_FEATURE_ALIASES.items():
        alias_lookup[_compact_feature_key(canonical_name)] = canonical_name
        for alias in aliases:
            alias_lookup[_compact_feature_key(alias)] = canonical_name
    for item in normalize_string_list(value):
        normalized = alias_lookup.get(_compact_feature_key(item))
        if normalized:
            resolved.append(normalized)
    return merge_unique(resolved)


def split_exact_feature_terms(value: Any) -> dict[str, list[str]]:
    exact_feature_terms = normalize_exact_feature_terms(value)
    retained_input_features = [name for name in exact_feature_terms if name in RETAINED_INPUT_FEATURES]
    derived_feature_names = [name for name in exact_feature_terms if name in DERIVED_FEATURE_NAMES]
    class_label_names = [name for name in exact_feature_terms if name in CLASS_LABEL_NAMES]
    return {
        "exact_feature_terms": exact_feature_terms,
        "retained_input_features": retained_input_features,
        "derived_feature_names": derived_feature_names,
        "class_label_names": class_label_names,
    }


def _dataset_feature_alias_keys(feature_name: str) -> set[str]:
    canonical_name = _normalize_feature_phrase(feature_name)
    if not canonical_name:
        return set()
    keys = {_compact_feature_key(canonical_name)}
    for alias in EXACT_FEATURE_ALIASES.get(canonical_name, []):
        keys.add(_compact_feature_key(alias))
    return keys


def match_dataset_feature_columns(columns: list[str] | tuple[str, ...], feature_names: Any) -> list[str]:
    normalized_terms = normalize_exact_feature_terms(feature_names)
    if not normalized_terms:
        return []

    keys_by_feature = {feature_name: _dataset_feature_alias_keys(feature_name) for feature_name in normalized_terms}
    matched: list[str] = []
    seen: set[str] = set()
    for column_name in columns or []:
        column_key = _compact_feature_key(column_name)
        for feature_name in normalized_terms:
            if column_key and column_key in keys_by_feature.get(feature_name, set()):
                if column_name not in seen:
                    seen.add(column_name)
                    matched.append(column_name)
                break
    return matched


def normalize_descriptor_names(value: Any) -> list[str]:
    resolved: list[str] = []
    for item in normalize_string_list(value):
        canonical = RDKIT_DESCRIPTOR_LOOKUP.get(item.lower())
        if canonical:
            resolved.append(canonical)
    return merge_unique(resolved)


def normalize_count_feature_names(value: Any) -> list[str]:
    resolved: list[str] = []
    alias_lookup: dict[str, str] = {}
    for canonical_name, aliases in COUNT_FEATURE_ALIASES.items():
        alias_lookup[canonical_name.lower()] = canonical_name
        for alias in aliases:
            alias_lookup[alias.lower()] = canonical_name
    for item in normalize_string_list(value):
        canonical = alias_lookup.get(item.lower())
        if canonical:
            resolved.append(canonical)
    return merge_unique(resolved)


def normalize_fingerprint_family(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    if not text or text == "not found":
        return None
    for family, aliases in FINGERPRINT_FAMILY_ALIASES.items():
        if text == family:
            return family
        if any(alias in text for alias in aliases):
            return family
    if "fingerprint" in text:
        return "morgan"
    return None


def _first_match_start(text: str, phrases: list[str]) -> int | None:
    best: int | None = None
    for phrase in phrases:
        lowered_phrase = phrase.lower()
        match = re.search(rf"(?<![a-z0-9_]){re.escape(lowered_phrase)}(?![a-z0-9_])", text)
        if match:
            start = match.start()
            if best is None or start < best:
                best = start
    return best


def extract_descriptor_names(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for descriptor_name in RDKIT_DESCRIPTOR_SNAPSHOT:
        start = _first_match_start(lowered, [descriptor_name])
        if start is not None:
            matches.append((start, descriptor_name))
    matches.sort(key=lambda item: item[0])
    return merge_unique([name for _, name in matches])


def extract_count_feature_names(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for canonical_name, aliases in COUNT_FEATURE_ALIASES.items():
        start = _first_match_start(lowered, [canonical_name, *aliases])
        if start is not None:
            matches.append((start, canonical_name))
    matches.sort(key=lambda item: item[0])
    return merge_unique([name for _, name in matches])


def extract_external_feature_terms(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for canonical_name, aliases in EXTERNAL_FEATURE_TOOL_ALIASES.items():
        start = _first_match_start(lowered, aliases)
        if start is not None:
            matches.append((start, canonical_name))
    matches.sort(key=lambda item: item[0])
    return merge_unique([name for _, name in matches])


def extract_exact_feature_terms(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []

    for canonical_name, aliases in EXACT_FEATURE_ALIASES.items():
        start = _first_match_start(lowered, [canonical_name, *aliases])
        if start is not None:
            matches.append((start, canonical_name))

    list_patterns = [
        r"(?:characteristics?|features?|feature set|input variables?|variables?|columns?)\s*\(([^)]{1,240})\)",
        r"(?:added|adding|transformation process.*?added)\s+\d+\s+characteristics?\s*\(([^)]{1,240})\)",
        r"(?:labeling compounds as|labeled as|classified as)\s*([a-z ,/()-]{1,200})",
    ]
    for pattern in list_patterns:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            raw_group = match.group(1)
            pieces = re.split(r",|\band\b", raw_group)
            for piece in pieces:
                normalized_piece = _normalize_feature_phrase(piece)
                if normalized_piece:
                    matches.append((match.start(1), normalized_piece))

    shorthand_patterns = [
        r"\b([cno])\s+number\b",
        r"\bc\s*/\s*n\s*/\s*o\s+number\b",
        r"\bside[- ]chain number\b",
    ]
    for pattern in shorthand_patterns:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            normalized_piece = _normalize_feature_phrase(match.group(0))
            if normalized_piece:
                matches.append((match.start(), normalized_piece))

    matches.sort(key=lambda item: item[0])
    return merge_unique([name for _, name in matches])


def extract_dataset_feature_count(text: str) -> int | None:
    lowered = text.lower()
    candidates: list[tuple[int, int, int]] = []
    for match in re.finditer(r"\b(\d+)\s+characteristics?\b", lowered):
        value = int(match.group(1))
        window_start = max(0, match.start() - 120)
        window_end = min(len(lowered), match.end() + 160)
        context = lowered[window_start:window_end]
        score = 0
        if "without boiling point" in context:
            score += 14
        if "data discretization" in context:
            score += 8
        if "data transformation" in context:
            score += 5
        if "data cleaning" in context:
            score += 5
        if "dataset contains" in context or "contains" in context:
            score += 4
        if "unknown dataset" in context:
            score += 4
        if "result" in context or "remains" in context:
            score += 3
        if "raw dataset" in context:
            score -= 8
        if "raw" in context and "dataset" in context:
            score -= 4
        if "boiling point" in context and "without boiling point" not in context:
            score -= 2
        candidates.append((score, match.start(), value))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def extract_fingerprint_details(text: str) -> dict[str, Any]:
    lowered = text.lower()
    family_matches: list[tuple[int, str]] = []
    for family, aliases in FINGERPRINT_FAMILY_ALIASES.items():
        start = _first_match_start(lowered, aliases)
        if start is not None:
            family_matches.append((start, family))
    family_matches.sort(key=lambda item: item[0])
    fingerprint_family = family_matches[0][1] if family_matches else None

    radius: int | None = None
    radius_match = re.search(r"(?:radius|rad)\s*[:=]?\s*(\d+)", lowered)
    if radius_match:
        radius = int(radius_match.group(1))
    else:
        ecfp_match = re.search(r"\becfp\s*[-_ ]?(\d+)\b", lowered)
        if ecfp_match:
            fingerprint_family = fingerprint_family or "morgan"
            radius = max(1, int(ecfp_match.group(1)) // 2)

    n_bits: int | None = None
    for pattern in [
        r"\b(\d{3,5})\s*[- ]?bits?\b",
        r"\bnbits?\s*[:=]?\s*(\d{3,5})\b",
        r"\bbit length\s*[:=]?\s*(\d{3,5})\b",
    ]:
        match = re.search(pattern, lowered)
        if match:
            n_bits = int(match.group(1))
            break

    if fingerprint_family is None and any(term in lowered for term in FINGERPRINT_SIGNAL_TERMS):
        fingerprint_family = "morgan"

    return {
        "fingerprint_family": fingerprint_family,
        "radius": radius,
        "n_bits": n_bits,
    }


def analyze_feature_text(*text_sources: Any) -> dict[str, Any]:
    combined_text = "\n".join([str(text).strip() for text in text_sources if str(text).strip()])
    lowered = combined_text.lower()
    descriptor_names = extract_descriptor_names(combined_text)
    count_feature_names = extract_count_feature_names(combined_text)
    exact_feature_terms = extract_exact_feature_terms(combined_text)
    exact_categories = split_exact_feature_terms(exact_feature_terms)
    fingerprint_details = extract_fingerprint_details(combined_text)
    unresolved_feature_terms = extract_external_feature_terms(combined_text)
    dataset_feature_count = extract_dataset_feature_count(combined_text)

    has_rdkit_descriptor_signal = bool(descriptor_names) or any(term in lowered for term in DESCRIPTOR_SIGNAL_TERMS)
    has_exact_dataset_feature_signal = bool(exact_categories["retained_input_features"] or exact_categories["derived_feature_names"] or exact_categories["class_label_names"])
    has_descriptor_signal = has_rdkit_descriptor_signal or has_exact_dataset_feature_signal
    has_count_signal = bool(count_feature_names)
    has_fingerprint_signal = fingerprint_details["fingerprint_family"] is not None

    if has_fingerprint_signal and (has_descriptor_signal or has_count_signal):
        method = "combined"
    elif has_fingerprint_signal:
        method = "morgan"
    elif has_descriptor_signal or has_count_signal:
        method = "descriptor"
    else:
        method = "Not found"

    feature_terms = merge_unique(
        exact_categories["retained_input_features"],
        exact_categories["derived_feature_names"],
        exact_categories["class_label_names"],
        descriptor_names,
        count_feature_names,
        [fingerprint_details["fingerprint_family"]] if fingerprint_details["fingerprint_family"] else [],
        unresolved_feature_terms,
    )
    return {
        "method": method,
        "descriptor_names": descriptor_names,
        "count_feature_names": count_feature_names,
        "fingerprint_family": fingerprint_details["fingerprint_family"],
        "radius": fingerprint_details["radius"],
        "n_bits": fingerprint_details["n_bits"],
        "feature_terms": feature_terms,
        "exact_feature_terms": exact_categories["exact_feature_terms"],
        "retained_input_features": exact_categories["retained_input_features"],
        "derived_feature_names": exact_categories["derived_feature_names"],
        "class_label_names": exact_categories["class_label_names"],
        "dataset_feature_count": dataset_feature_count,
        "unresolved_feature_terms": unresolved_feature_terms,
        "has_descriptor_signal": has_descriptor_signal,
        "has_rdkit_descriptor_signal": has_rdkit_descriptor_signal,
        "has_exact_dataset_feature_signal": has_exact_dataset_feature_signal,
        "has_count_signal": has_count_signal,
        "has_fingerprint_signal": has_fingerprint_signal,
    }


def _expand_grouped_exact_feature_terms(feature_names: Any) -> list[str]:
    expanded: list[str] = []
    for feature_name in normalize_exact_feature_terms(feature_names):
        if feature_name == "C/N/O number":
            expanded.extend(["C number", "N number", "O number"])
            continue
        expanded.append(feature_name)
    return merge_unique(expanded)


def build_evidence_first_feature_contract(feature_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(feature_payload or {})
    analysis = analyze_feature_text(
        payload.get("summary", ""),
        payload.get("key_values", ""),
        payload.get("evidence_snippet", ""),
    )

    retained_input_features = _expand_grouped_exact_feature_terms(analysis["retained_input_features"])
    derived_feature_names = _expand_grouped_exact_feature_terms(analysis["derived_feature_names"])
    class_label_names = _expand_grouped_exact_feature_terms(analysis["class_label_names"])
    descriptor_names = list(analysis["descriptor_names"])
    count_feature_names = list(analysis["count_feature_names"])
    fingerprint_family = analysis["fingerprint_family"]
    unresolved_feature_terms = list(analysis["unresolved_feature_terms"])
    feature_terms = merge_unique(
        retained_input_features,
        derived_feature_names,
        class_label_names,
        descriptor_names,
        count_feature_names,
        [fingerprint_family] if fingerprint_family else [],
        unresolved_feature_terms,
    )

    has_tabular_features = bool(retained_input_features or derived_feature_names)
    has_descriptor_features = bool(descriptor_names)
    has_count_features = bool(count_feature_names)
    has_fingerprint_features = bool(fingerprint_family)
    if has_fingerprint_features and (has_tabular_features or has_descriptor_features or has_count_features):
        method = "combined"
    elif has_fingerprint_features:
        method = "morgan"
    elif has_tabular_features or has_descriptor_features or has_count_features or class_label_names:
        method = "descriptor"
    else:
        method = "Not found"

    return {
        "summary": payload.get("summary", "Not found"),
        "key_values": payload.get("key_values", "Not found"),
        "evidence_chunks": payload.get("evidence_chunks", []),
        "evidence_snippet": payload.get("evidence_snippet", "Not found"),
        "method": method,
        "descriptor_names": descriptor_names,
        "count_feature_names": count_feature_names,
        "fingerprint_family": fingerprint_family,
        "radius": analysis["radius"] if fingerprint_family else None,
        "n_bits": analysis["n_bits"] if fingerprint_family else None,
        "use_rdkit_descriptors": bool(descriptor_names),
        "retained_input_features": retained_input_features,
        "derived_feature_names": derived_feature_names,
        "class_label_names": class_label_names,
        "dataset_feature_count": analysis["dataset_feature_count"],
        "feature_terms": feature_terms,
        "unresolved_feature_terms": unresolved_feature_terms,
    }


def augment_feature_payload(feature_payload: dict[str, Any] | None, text_sources: list[str] | None = None) -> dict[str, Any]:
    payload = dict(feature_payload or {})
    analysis = analyze_feature_text(
        payload.get("summary", ""),
        payload.get("key_values", ""),
        payload.get("evidence_snippet", ""),
        *(text_sources or []),
    )

    descriptor_names = merge_unique(
        normalize_descriptor_names(payload.get("descriptor_names")),
        analysis["descriptor_names"],
    )
    count_feature_names = merge_unique(
        normalize_count_feature_names(payload.get("count_feature_names")),
        analysis["count_feature_names"],
    )
    unresolved_feature_terms = merge_unique(
        normalize_string_list(payload.get("unresolved_feature_terms")),
        analysis["unresolved_feature_terms"],
    )
    retained_input_features = merge_unique(
        split_exact_feature_terms(payload.get("retained_input_features")).get("retained_input_features", []),
        analysis["retained_input_features"],
    )
    derived_feature_names = merge_unique(
        split_exact_feature_terms(payload.get("derived_feature_names")).get("derived_feature_names", []),
        analysis["derived_feature_names"],
    )
    class_label_names = merge_unique(
        split_exact_feature_terms(payload.get("class_label_names")).get("class_label_names", []),
        analysis["class_label_names"],
    )
    fingerprint_family = normalize_fingerprint_family(payload.get("fingerprint_family")) or analysis["fingerprint_family"]
    feature_terms = merge_unique(
        normalize_string_list(payload.get("feature_terms")),
        retained_input_features,
        derived_feature_names,
        class_label_names,
        descriptor_names,
        count_feature_names,
        [fingerprint_family] if fingerprint_family else [],
        unresolved_feature_terms,
        analysis["feature_terms"],
    )

    method = str(payload.get("method", "")).strip() or analysis["method"]
    if method == "Not found" and analysis["method"] != "Not found":
        method = analysis["method"]

    use_rdkit_descriptors = payload.get("use_rdkit_descriptors")
    if use_rdkit_descriptors is None:
        if descriptor_names or analysis["has_rdkit_descriptor_signal"]:
            use_rdkit_descriptors = True
        elif count_feature_names or retained_input_features or derived_feature_names or class_label_names:
            use_rdkit_descriptors = False

    dataset_feature_count = payload.get("dataset_feature_count")
    if dataset_feature_count in {None, "", "Not found"} and analysis["dataset_feature_count"] is not None:
        dataset_feature_count = analysis["dataset_feature_count"]
    elif dataset_feature_count not in {None, "", "Not found"}:
        try:
            dataset_feature_count = int(dataset_feature_count)
        except (TypeError, ValueError):
            dataset_feature_count = analysis["dataset_feature_count"]

    payload["method"] = method
    payload["descriptor_names"] = descriptor_names
    payload["count_feature_names"] = count_feature_names
    payload["fingerprint_family"] = fingerprint_family
    payload["retained_input_features"] = retained_input_features
    payload["derived_feature_names"] = derived_feature_names
    payload["class_label_names"] = class_label_names
    payload["dataset_feature_count"] = dataset_feature_count
    payload["feature_terms"] = feature_terms
    payload["unresolved_feature_terms"] = unresolved_feature_terms
    if payload.get("radius") in {None, "", "Not found"} and analysis["radius"] is not None:
        payload["radius"] = analysis["radius"]
    if payload.get("n_bits") in {None, "", "Not found"} and analysis["n_bits"] is not None:
        payload["n_bits"] = analysis["n_bits"]
    if use_rdkit_descriptors is not None:
        payload["use_rdkit_descriptors"] = use_rdkit_descriptors
    return payload
