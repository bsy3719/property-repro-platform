"""
FeatureResolver: spec의 feature명 → 실제 사용할 RDKit descriptor 결정.
코드 생성 전에 실행해서 LLM 프롬프트 및 code_spec에 매핑 결과를 주입한다.
"""

from __future__ import annotations

import difflib
import re

from src.utils import RDKIT_DESCRIPTOR_SNAPSHOT, create_openai_client, run_text_response

from .resources.rdkit_descriptor_reference import FEATURE_FALLBACK_MAP

# ---------------------------------------------------------------------------
# 내부 상수
# ---------------------------------------------------------------------------

# 논문/데이터셋에서 자주 쓰이는 비표준 feature명 → canonical RDKit descriptor
_DIRECT_ALIAS_MAP: dict[str, str] = {
    # exact SMILES feature names used in this project
    "mw": "MolWt",
    "polararea": "TPSA",
    "heavycnt": "HeavyAtomCount",
    "hbondacc": "NumHAcceptors",
    "hbonddon": "NumHDonors",
    # weight / size variants
    "molwt": "MolWt",
    "molweight": "MolWt",
    "molecular weight": "MolWt",
    "mol weight": "MolWt",
    "exactmolwt": "ExactMolWt",
    "exact mol wt": "ExactMolWt",
    # logP / refractivity
    "logp": "MolLogP",
    "alogp": "MolLogP",
    "clogp": "MolLogP",
    "xlogp": "MolLogP",
    "log p": "MolLogP",
    "mr": "MolMR",
    "molar refractivity": "MolMR",
    # polar surface area
    "tpsa": "TPSA",
    "psa": "TPSA",
    "polar surface area": "TPSA",
    "topological polar surface area": "TPSA",
    # H-bond
    "hba": "NumHAcceptors",
    "hbd": "NumHDonors",
    "h bond acceptors": "NumHAcceptors",
    "h bond donors": "NumHDonors",
    "hydrogen bond acceptors": "NumHAcceptors",
    "hydrogen bond donors": "NumHDonors",
    "num hba": "NumHAcceptors",
    "num hbd": "NumHDonors",
    # rotatable bonds
    "rotatable bonds": "NumRotatableBonds",
    "num rotatable bonds": "NumRotatableBonds",
    # ring counts
    "ring count": "RingCount",
    "num rings": "RingCount",
    "total rings": "RingCount",
    "aromatic rings": "NumAromaticRings",
    "num aromatic rings": "NumAromaticRings",
    "aliphatic rings": "NumAliphaticRings",
    "num aliphatic rings": "NumAliphaticRings",
    "saturated rings": "NumSaturatedRings",
    # atom counts
    "heavy atoms": "HeavyAtomCount",
    "heavy atom count": "HeavyAtomCount",
    "heteroatoms": "NumHeteroatoms",
    "num heteroatoms": "NumHeteroatoms",
    # complexity / topology
    "bertz": "BertzCT",
    "complexity": "BertzCT",
    "balaban j": "BalabanJ",
    "ipc": "Ipc",
    # surface area
    "asa": "LabuteASA",
    "labute asa": "LabuteASA",
    "approx sa": "LabuteASA",
    # saturation
    "fsp3": "FractionCSP3",
    "fraction csp3": "FractionCSP3",
    "sp3 fraction": "FractionCSP3",
}

_COUNT_FEATURE_ALIAS_MAP: dict[str, list[str]] = {
    "c number": ["C_count"],
    "c numbers": ["C_count"],
    "carbon number": ["C_count"],
    "carbon count": ["C_count"],
    "n number": ["N_count"],
    "n numbers": ["N_count"],
    "nitrogen number": ["N_count"],
    "nitrogen count": ["N_count"],
    "o number": ["O_count"],
    "o numbers": ["O_count"],
    "oxygen number": ["O_count"],
    "oxygen count": ["O_count"],
    "c n o number": ["C_count", "N_count", "O_count"],
    "c n o numbers": ["C_count", "N_count", "O_count"],
    "c/n/o number": ["C_count", "N_count", "O_count"],
    "c/n/o numbers": ["C_count", "N_count", "O_count"],
    "c, n, o number": ["C_count", "N_count", "O_count"],
    "c, n, o numbers": ["C_count", "N_count", "O_count"],
    "cno number": ["C_count", "N_count", "O_count"],
    "cno numbers": ["C_count", "N_count", "O_count"],
    "c n o count": ["C_count", "N_count", "O_count"],
    "atom count": ["AtomCount"],
    "bond count": ["BondCount"],
    "ring count": ["RingCount"],
}

# 유효한 RDKit descriptor 이름 집합 (런타임 확인용)
_RDKIT_VALID_SET: frozenset[str] = frozenset(RDKIT_DESCRIPTOR_SNAPSHOT)

# 식별자/레이블 컬럼으로 판단할 정확 매칭 목록
_IDENTIFIER_EXACT: frozenset[str] = frozenset({
    "smiles", "canonical_smiles", "isomeric_smiles", "iso smiles", "iso-smiles",
    "compound", "compound_name", "compound name", "cmpdname", "cmpd_name",
    "molecule", "molecule_name", "molecule name",
    "name", "inchi", "inchikey", "inchi_key",
    "id", "mol_id", "compound_id", "cmpd_id",
    "cas", "cas_no", "cas number", "cas_number",
    "index", "entry",
})

# 식별자 패턴 정규식 (낮은 false-positive를 위해 보수적으로 설정)
_IDENTIFIER_REGEX = re.compile(
    r"^("
    r"smiles|canonical|isomeric|inchi|"
    r"cas[_\s-]?(no|number)?|"
    r"cmpd[_\s]?(name|id)?|compound[_\s]?(name|id)?|"
    r"molecule?[_\s]?(name|id)?|mol[_\s]id|"
    r".+[_\s]id|.+[_\s]name|name|^id$"
    r")$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# FeatureResolver
# ---------------------------------------------------------------------------


class FeatureResolver:
    """
    spec의 feature명 → 실제 사용할 RDKit descriptor 결정.
    코드 생성 전에 실행해서 LLM 프롬프트에 매핑 결과를 주입한다.
    """

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        feature_names: list[str],
        *,
        class_label_names: list[str] | None = None,
        target_columns: list[str] | None = None,
    ) -> dict:
        """
        반환 형태:
        {
          "resolved": {
            "polararea": "TPSA",            # 직접 매핑
            "side chain number": "BertzCT", # fallback map + LLM 선택
            "mf": "HeavyAtomCount",         # fallback map + LLM 선택
          },
          "resolved_counts": {
            "C/N/O number": ["C_count", "N_count", "O_count"],
            "O number": ["O_count"],
          },
          "excluded": ["cmpdname", "iso-smiles"],  # 식별자/레이블 제외
          "llm_resolved": ["side chain number"],   # LLM이 판단한 항목
          "assumptions": [
            "side chain number → BertzCT (FEATURE_FALLBACK_MAP 후보 중 LLM 선택)",
            "mf → HeavyAtomCount (FEATURE_FALLBACK_MAP 후보 중 LLM 선택)",
            "cmpdname 제외 (식별자 컬럼)",
          ]
        }
        """
        resolved: dict[str, str] = {}
        resolved_counts: dict[str, list[str]] = {}
        excluded: list[str] = []
        llm_resolved: list[str] = []
        assumptions: list[str] = []

        class_label_names = [str(name).strip() for name in (class_label_names or []) if str(name).strip()]
        target_columns = [str(name).strip() for name in (target_columns or []) if str(name).strip()]

        for raw_name in feature_names:
            if not raw_name or not str(raw_name).strip():
                continue
            name = str(raw_name).strip()

            # 1. 식별자/레이블 컬럼 → 제외
            if self._is_identifier(name):
                excluded.append(name)
                assumptions.append(f"{name} 제외 (식별자 컬럼)")
                continue
            if self._is_class_label(name, class_label_names):
                excluded.append(name)
                assumptions.append(f"{name} 제외 (class label)")
                continue
            if self._is_target_column(name, target_columns):
                excluded.append(name)
                assumptions.append(f"{name} 제외 (target 컬럼)")
                continue

            # 2. atom/bond count feature 직접 매핑
            count_direct = self._direct_count_map(name)
            if count_direct:
                resolved_counts[name] = count_direct
                assumptions.append(f"{name} → {', '.join(count_direct)} (atom/bond count 직접 매핑)")
                continue

            # 3. 직접 매핑 (정확한 RDKit 이름 또는 alias)
            direct = self._direct_map(name)
            if direct:
                resolved[name] = direct
                if direct != name:
                    assumptions.append(f"{name} → {direct} (직접 매핑)")
                continue

            # 4. fallback map → 후보 있으면 LLM 선택
            candidates = self._fallback_map(name)
            if candidates:
                selected = self._llm_select(name, candidates)
                resolved[name] = selected
                llm_resolved.append(name)
                if selected == candidates[0]:
                    assumptions.append(
                        f"{name} → {selected} (FEATURE_FALLBACK_MAP 후보 중 LLM 선택)"
                    )
                else:
                    assumptions.append(
                        f"{name} → {selected} (FEATURE_FALLBACK_MAP 후보 중 LLM 선택)"
                    )
                continue

            # 5. 후보 없음 → LLM이 RDKit에서 가장 유사한 것을 탐색
            selected = self._llm_select(name, None)
            if selected:
                resolved[name] = selected
                llm_resolved.append(name)
                assumptions.append(
                    f"{name} → {selected} (LLM이 RDKit에서 가장 유사한 descriptor 선택)"
                )
            else:
                excluded.append(name)
                assumptions.append(f"{name} 제외 (RDKit 대응 descriptor 없음)")

        return {
            "resolved": resolved,
            "resolved_counts": resolved_counts,
            "excluded": excluded,
            "llm_resolved": llm_resolved,
            "assumptions": assumptions,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_identifier(self, name: str) -> bool:
        """smiles, name, id, cmpdname 등 식별자/레이블 컬럼 패턴 감지."""
        normalized = name.strip().lower()
        if normalized in _IDENTIFIER_EXACT:
            return True
        return bool(_IDENTIFIER_REGEX.match(normalized))

    def _is_class_label(self, feature_name: str, class_label_names: list[str]) -> bool:
        normalized = self._normalize(feature_name)
        return normalized in {self._normalize(name) for name in class_label_names if name}

    def _is_target_column(self, feature_name: str, target_columns: list[str]) -> bool:
        normalized = self._normalize(feature_name)
        return normalized in {self._normalize(name) for name in target_columns if name}

    def _direct_map(self, name: str) -> str | None:
        """
        RDKIT_DESCRIPTOR_SNAPSHOT에서 정확/유사 매핑 시도.
        1) 원본 이름이 유효한 RDKit descriptor → 그대로 반환
        2) case-insensitive 매칭
        3) _DIRECT_ALIAS_MAP 조회
        4) difflib.get_close_matches 활용
        """
        # 정확 매칭
        if name in _RDKIT_VALID_SET:
            return name
        # case-insensitive
        lower = name.strip().lower()
        for rdkit_name in _RDKIT_VALID_SET:
            if rdkit_name.lower() == lower:
                return rdkit_name
        # alias 맵
        alias_hit = _DIRECT_ALIAS_MAP.get(lower)
        if alias_hit:
            return alias_hit

        normalized_to_descriptor = {
            self._normalize(alias): descriptor
            for alias, descriptor in _DIRECT_ALIAS_MAP.items()
        }
        descriptor_lookup = {
            self._normalize(rdkit_name): rdkit_name
            for rdkit_name in _RDKIT_VALID_SET
        }
        normalized_name = self._normalize(name)
        close_alias = difflib.get_close_matches(
            normalized_name,
            list(normalized_to_descriptor.keys()),
            n=1,
            cutoff=0.75,
        )
        if close_alias:
            return normalized_to_descriptor[close_alias[0]]
        close_descriptor = difflib.get_close_matches(
            normalized_name,
            list(descriptor_lookup.keys()),
            n=1,
            cutoff=0.8,
        )
        if close_descriptor:
            return descriptor_lookup[close_descriptor[0]]
        return None

    def _direct_count_map(self, name: str) -> list[str] | None:
        normalized_name = self._normalize(name)
        direct = _COUNT_FEATURE_ALIAS_MAP.get(normalized_name)
        if direct:
            return list(direct)
        close_alias = difflib.get_close_matches(
            normalized_name,
            list(_COUNT_FEATURE_ALIAS_MAP.keys()),
            n=1,
            cutoff=0.82,
        )
        if close_alias:
            return list(_COUNT_FEATURE_ALIAS_MAP[close_alias[0]])
        return None

    def _fallback_map(self, name: str) -> list[str] | None:
        """
        FEATURE_FALLBACK_MAP에서 후보 반환.
        정확 키 매칭 후, 없으면 부분 문자열 매칭 시도.
        """
        lower = name.strip().lower()
        if lower in FEATURE_FALLBACK_MAP:
            return FEATURE_FALLBACK_MAP[lower]
        # 부분 매칭 (키가 name에 포함되거나 name이 키에 포함)
        for key, candidates in FEATURE_FALLBACK_MAP.items():
            if key in lower or lower in key:
                return candidates
        return None

    def _llm_select(self, feature_name: str, candidates: list[str] | None) -> str | None:
        """
        candidates가 있으면 "이 중에서 골라라",
        candidates가 None이면 "RDKit에서 가장 유사한 걸 찾아라".
        LLM 실패 시 candidates[0] 반환 (없으면 None).
        """
        try:
            if candidates:
                prompt = (
                    f"Select the single best RDKit descriptor for the chemistry feature below.\n"
                    f"Feature name (from paper): '{feature_name}'\n"
                    f"Candidates (all are valid RDKit descriptors): {candidates}\n\n"
                    f"Return ONLY the exact name of the chosen descriptor — "
                    f"no explanation, no quotes, no extra text."
                )
            else:
                valid_examples = (
                    "MolWt, TPSA, MolLogP, HeavyAtomCount, NumHDonors, NumHAcceptors, "
                    "NumRotatableBonds, RingCount, BertzCT, Chi0v, MaxPartialCharge, "
                    "LabuteASA, NumAromaticRings, FractionCSP3, Ipc, NumHeteroatoms, "
                    "NHOHCount, NOCount, MolMR, BalabanJ, Kappa1, Kappa2, Kappa3, HallKierAlpha"
                )
                prompt = (
                    f"Map the chemistry feature name below to the closest valid RDKit descriptor.\n"
                    f"Feature name (from paper): '{feature_name}'\n\n"
                    f"Some valid RDKit descriptors: {valid_examples}, and many more.\n\n"
                    f"If a reasonable match exists, return ONLY the exact RDKit descriptor name.\n"
                    f"If no reasonable match exists, return exactly: NOT_FOUND\n"
                    f"No explanation, no quotes, just the name."
                )
            raw = run_text_response(
                self.client,
                self.model_name,
                prompt,
                f"feature resolution: {feature_name}",
            )
            result = raw.strip().strip('"').strip("'")
            if result.upper() == "NOT_FOUND":
                return None
            # 유효성 검증
            if candidates and result in candidates:
                return result
            if result in _RDKIT_VALID_SET:
                return result
            # case-insensitive 재확인
            for rdkit_name in _RDKIT_VALID_SET:
                if rdkit_name.lower() == result.lower():
                    return rdkit_name
            # LLM이 알 수 없는 이름을 반환 → fallback
            return candidates[0] if candidates else None
        except Exception:
            return candidates[0] if candidates else None

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()
