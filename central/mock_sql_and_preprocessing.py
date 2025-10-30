# CDA batch extractor – fixed CEDIS parsing and CSV saved next to this script

from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd


# Path to the folder where this script itself is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Path to cda files
ROOT = (SCRIPT_DIR / "../dwh").resolve()

NAMESPACES = {"hl7": "urn:hl7-org:v3"}


def parse_dt(val: str):
    if not val:
        return None
    for fmt in ("%Y", "%Y%m", "%Y%m%d", "%Y%m%d%H%M", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(val, fmt)
        except Exception:
            pass
    return None


def find_section_by_display_name(root, display_name):
    for sec in root.findall(".//hl7:section", NAMESPACES):
        code = sec.find("./hl7:code", NAMESPACES)
        if code is not None and code.get("displayName") == display_name:
            return sec
    return None


def get_triage_value_elem(root):
    sec = find_section_by_display_name(root, "Acuity assessment")
    if sec is None:
        return None
    return sec.find(".//hl7:observation//hl7:value", NAMESPACES)


def get_triage_score(root):
    val = get_triage_value_elem(root)
    return val.get("code") if val is not None else None


def get_triage_times(root):
    sec = find_section_by_display_name(root, "Acuity assessment")
    if sec is None:
        return None, None
    obs = sec.find(".//hl7:observation", NAMESPACES)
    if obs is None:
        return None, None

    low = obs.find("./hl7:effectiveTime/hl7:low", NAMESPACES)
    high = obs.find("./hl7:effectiveTime/hl7:high", NAMESPACES)

    low_dt = (
        parse_dt(low.get("value")) if low is not None and low.get("value") else None
    )
    high_dt = (
        parse_dt(high.get("value")) if high is not None and high.get("value") else None
    )

    return low_dt, high_dt


def get_birth_date(root):
    node = root.find(
        ".//hl7:recordTarget/hl7:patientRole/hl7:patient/hl7:birthTime", NAMESPACES
    )
    return (
        parse_dt(node.get("value")) if node is not None and node.get("value") else None
    )


def get_encounter_start(root):
    node = root.find(
        ".//hl7:componentOf/hl7:encompassingEncounter/hl7:effectiveTime/hl7:low",
        NAMESPACES,
    )
    return (
        parse_dt(node.get("value")) if node is not None and node.get("value") else None
    )


def years_between(birth_dt, ref_dt):
    if not birth_dt or not ref_dt:
        return None
    return (
        ref_dt.year
        - birth_dt.year
        - ((ref_dt.month, ref_dt.day) < (birth_dt.month, birth_dt.day))
    )


# --- Case-insensitive, namespace-safe attribute getter ---
def _get_attr_ci(elem, attr_name: str):
    if elem is None:
        return None
    target = attr_name.lower()
    for k, v in elem.attrib.items():
        local = k.split("}", 1)[-1].lower()
        if local == target:
            return v
    return None


# --- Robust CEDIS extraction ---
def get_cedis_from_triage_value(root):
    val = get_triage_value_elem(root)
    if val is None:
        return {}
    return {
        "code": _get_attr_ci(val, "code"),
        "codeSystem": _get_attr_ci(val, "codeSystem"),
        "codeSystemName": _get_attr_ci(val, "codeSystemName"),
        "displayName": _get_attr_ci(val, "displayName"),
    }


def get_stationaere_aufnahme(root):
    node = root.find(
        ".//hl7:componentOf/hl7:encompassingEncounter/hl7:dischargeDispositionCode",
        NAMESPACES,
    )
    if node is None:
        return {}
    return {"code": node.get("code")}


def bucket(dt):
    if not dt:
        return None
    h = dt.hour
    if 5 <= h < 12:
        return "Morgen"
    elif 12 <= h < 17:
        return "Nachmittag"
    elif 17 <= h < 22:
        return "Abend"
    else:
        return "Nacht"


def extract_one(xml_path: Path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        triage_score = get_triage_score(root)
        triage_low, _ = get_triage_times(root)
        birth_dt = get_birth_date(root)
        encounter_start = get_encounter_start(root)
        alter = years_between(birth_dt, encounter_start)
        cedis = get_cedis_from_triage_value(root)
        stationaer = get_stationaere_aufnahme(root)

        return {
            "file": xml_path.name,
            "Triage-Score": triage_score,
            "Tageszeit": bucket(triage_low),
            "Alter": alter,
            "CEDIS": cedis.get("code"),
            "stationäre Aufnahme (code)": stationaer.get("code"),
        }

    except Exception as e:
        return {"file": xml_path.name, "error": str(e)}


def main():
    xml_files = sorted(ROOT.rglob("*.xml"))
    rows = [extract_one(p) for p in xml_files]
    df = pd.DataFrame(rows)

    out_csv = SCRIPT_DIR / "cda_summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
