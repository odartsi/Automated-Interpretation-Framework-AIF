from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.analysis.magnetism.analyzer import Ordering
from emmet.core.symmetry import CrystalSystem
API_KEY = "QVYHs5rzUOrOUoq6HOTikMAJkp1VFs5r" 



from mp_api.client import MPRester

def has_stable_polymorph(formula: str, api_key: API_KEY) -> bool:
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(formula=formula)

        for doc in docs:
            if doc.is_stable:
                print(f"✅ Stable polymorph found: {formula}_{doc.symmetry.number}")
                return True

        print(f"❌ No stable polymorph found for {formula}.")
        return False
def get_stable_polymorph_spacegroup(formula: str, api_key: str = API_KEY):
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(formula=formula)

        for doc in docs:
            if doc.is_stable:
                print(f"✅ Stable polymorph found: {formula}_{doc.symmetry.number}")
                return doc.symmetry.number 
        return None