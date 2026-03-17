from __future__ import annotations

CANONICAL_SCAFFOLD_GUIDANCE = """
Always generate code with this top-level structure and helper order:
1) imports
2) SPEC = {...}
3) ASSUMPTIONS = [...]
4) parse_args()
5) load_dataframe()
6) mol_from_smiles()
7) build_descriptor_matrix()
8) build_count_feature_matrix()
9) build_fingerprint_matrix()
10) assemble_feature_matrix()
11) build_model()
12) train_and_evaluate()
13) main()

Rules for this scaffold:
- Keep helper names exactly as written above.
- Use SPEC["feature_pipeline"] to drive which feature blocks run.
- Use only the evidence-backed feature fields already present in SPEC["feature_pipeline"].
- If smiles_feature_names is non-empty, generate those exact features from the SMILES column with RDKit.
- If dataset_required_columns is non-empty, use only those remaining tabular columns through allowed_feature_columns.
- If descriptor_names is non-empty, compute only that descriptor subset.
- If count_feature_names is non-empty, compute only those count features in build_count_feature_matrix().
- If fingerprint_family is present, route build_fingerprint_matrix() by family.
- Do not fall back to full RDKit descriptors, implicit fingerprints, or extra count features when the evidence-backed feature contract does not require them.
- class_label_names are metadata only unless they are also explicitly allowed as model input columns.
""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "title": "Evidence-backed exact SMILES features only",
        "description": "Boiling point regression using only exact paper-backed features that are generated from the SMILES column.",
        "snippet": """
SPEC = {
    "dataset": {"smiles_column": "smiles", "target_column": "bp"},
    "feature_pipeline": {
        "method": "descriptor",
        "descriptor_names": [],
        "count_feature_names": [],
        "fingerprint_family": None,
        "radius": None,
        "n_bits": None,
        "use_rdkit_descriptors": False,
        "retained_input_features": ["mw", "mf", "polararea", "heavycnt", "hbondacc"],
        "derived_feature_names": ["C number", "N number", "O number", "side chain number"],
        "class_label_names": ["hydrocarbon", "alcohol", "amine"],
        "required_model_columns": ["mw", "mf", "polararea", "heavycnt", "hbondacc", "C number", "N number", "O number", "side chain number"],
        "smiles_feature_names": ["mw", "mf", "polararea", "heavycnt", "hbondacc", "C number", "N number", "O number", "side chain number"],
        "dataset_required_columns": [],
        "missing_required_feature_columns": [],
        "allowed_feature_columns": [],
        "unresolved_feature_terms": [],
    },
}

def build_descriptor_matrix(mols, descriptor_names, use_all_descriptors=False):
    return np.zeros((len(mols), 0), dtype=float), []

def build_count_feature_matrix(mols, count_feature_names):
    return np.zeros((len(mols), 0), dtype=float), []

def build_fingerprint_matrix(mols, family, radius, n_bits):
    return np.zeros((len(mols), 0), dtype=float)

def build_smiles_feature_matrix(mols, smiles_feature_names):
    formula_values = [rdMolDescriptors.CalcMolFormula(mol) for mol in mols]
    formula_feature_names = ["mf::" + formula for formula in sorted(set(formula_values))]
    rows = []
    for mol, formula_value in zip(mols, formula_values):
        row = [
            float(Descriptors.MolWt(mol)),
            *[1.0 if formula_value == formula_name.split("::", 1)[1] else 0.0 for formula_name in formula_feature_names],
            float(Descriptors.TPSA(mol)),
            float(Descriptors.HeavyAtomCount(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)),
            float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)),
            float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)),
            float(sum(max(atom.GetDegree() - 2, 0) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)),
        ]
        rows.append(row)
    return np.asarray(rows, dtype=float), ["mw", *formula_feature_names, "polararea", "heavycnt", "hbondacc", "C number", "N number", "O number", "side chain number"]

def assemble_feature_matrix(df, mols):
    if SPEC["feature_pipeline"]["missing_required_feature_columns"]:
        raise ValueError("논문에 명시된 필수 feature를 사용 가능한 입력에서 해결하지 못했습니다.")
    smiles_matrix, smiles_columns = build_smiles_feature_matrix(
        mols,
        SPEC["feature_pipeline"]["smiles_feature_names"],
    )
    return smiles_matrix, {
        "smiles_feature_names": SPEC["feature_pipeline"]["smiles_feature_names"],
        "smiles_feature_columns": smiles_columns,
        "count_feature_names": [],
        "fingerprint_family": None,
    }
""".strip(),
    },
    {
        "title": "Explicit RDKit count features only",
        "description": "Boiling point regression using only count-style chemistry features that were explicitly present in the evidence.",
        "snippet": """
SPEC = {
    "dataset": {"smiles_column": "smiles", "target_column": "bp"},
    "feature_pipeline": {
        "method": "descriptor",
        "descriptor_names": [],
        "count_feature_names": ["AtomCount", "BondCount", "HeavyAtomCount", "RingCount", "NumRotatableBonds"],
        "fingerprint_family": None,
        "radius": None,
        "n_bits": None,
        "use_rdkit_descriptors": False,
        "unresolved_feature_terms": [],
    },
}

def build_count_feature_matrix(mols, count_feature_names):
    rows = []
    for mol in mols:
        row = []
        for feature_name in count_feature_names:
            if feature_name == "AtomCount":
                row.append(float(mol.GetNumAtoms()))
            elif feature_name == "BondCount":
                row.append(float(mol.GetNumBonds()))
            elif feature_name == "HeavyAtomCount":
                row.append(float(Descriptors.HeavyAtomCount(mol)))
            elif feature_name == "RingCount":
                row.append(float(Descriptors.RingCount(mol)))
            elif feature_name == "NumRotatableBonds":
                row.append(float(Descriptors.NumRotatableBonds(mol)))
        rows.append(row)
    return np.asarray(rows, dtype=float), count_feature_names

def assemble_feature_matrix(mols):
    count_matrix, count_names = build_count_feature_matrix(
        mols,
        SPEC["feature_pipeline"]["count_feature_names"],
    )
    return count_matrix, {
        "descriptor_names": [],
        "count_feature_names": count_names,
        "fingerprint_family": None,
    }
""".strip(),
    },
    {
        "title": "Explicit fingerprint plus descriptor and count features",
        "description": "Boiling point regression using only descriptor/count/fingerprint features that were explicitly present in the evidence.",
        "snippet": """
SPEC = {
    "dataset": {"smiles_column": "smiles", "target_column": "bp"},
    "feature_pipeline": {
        "method": "combined",
        "descriptor_names": ["MolWt", "TPSA"],
        "count_feature_names": ["RingCount", "NumRotatableBonds"],
        "fingerprint_family": "morgan",
        "radius": 2,
        "n_bits": 2048,
        "use_rdkit_descriptors": True,
        "unresolved_feature_terms": [],
    },
}

def build_fingerprint_matrix(mols, family, radius, n_bits):
    rows = []
    for mol in mols:
        vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=float)
        DataStructs.ConvertToNumpyArray(vect, arr)
        rows.append(arr)
    return np.vstack(rows)

def assemble_feature_matrix(mols):
    descriptor_matrix, descriptor_names = build_descriptor_matrix(
        mols,
        SPEC["feature_pipeline"]["descriptor_names"],
        use_all_descriptors=False,
    )
    count_matrix, count_names = build_count_feature_matrix(
        mols,
        SPEC["feature_pipeline"]["count_feature_names"],
    )
    fingerprint_matrix = build_fingerprint_matrix(
        mols,
        family=SPEC["feature_pipeline"]["fingerprint_family"],
        radius=SPEC["feature_pipeline"]["radius"],
        n_bits=SPEC["feature_pipeline"]["n_bits"],
    )
    feature_blocks = [block for block in [descriptor_matrix, count_matrix, fingerprint_matrix] if block.size]
    return np.hstack(feature_blocks), {
        "descriptor_names": descriptor_names,
        "count_feature_names": count_names,
        "fingerprint_family": SPEC["feature_pipeline"]["fingerprint_family"],
    }
""".strip(),
    },
]


def build_few_shot_prompt_block() -> str:
    parts = [CANONICAL_SCAFFOLD_GUIDANCE, ""]
    for index, example in enumerate(FEW_SHOT_EXAMPLES, start=1):
        parts.append(f"Few-shot example {index}: {example['title']}")
        parts.append(example["description"])
        parts.append("```python")
        parts.append(example["snippet"])
        parts.append("```")
        parts.append("")
    return "\n".join(parts).strip()
