import re
from collections import defaultdict



def parse_alloy_formula(formula: str):
    """
    Parse alloy composition like (Cu50Zr50)92Al7Gd1
    For missing percentages: assign count 1 to unspecified elements,
    
    Returns dictionary of element -> normalized fraction.
    Returns None if formula cannot be parsed (e.g., contains non-alloy components).
    """
    try:
        comp = defaultdict(float)

        # Remove all whitespace to be robust to inputs like "(Cu50 Zr50) 92 Al7"
        formula = re.sub(r"\s+", "", formula)

        # Handle grouped parentheses like (Cu50Zr50)92
        pattern = r"\(([A-Za-z0-9\.]+)\)(\d+)"
        while re.search(pattern, formula):
            for block, mult in re.findall(pattern, formula):
                subcomp = parse_alloy_formula(block)
                total_frac = sum(subcomp.values())
                for el, frac in subcomp.items():
                    comp[el] += frac * float(mult) / total_frac
            # Remove all processed parenthetical groups to avoid double-counting
            formula = re.sub(pattern, "", formula)

        # Handle element parts possibly missing explicit percentages.
        # If some elements have numbers and others do not, assign count 1 to unspecified elements.
        tokens = re.findall(r"([A-Z][a-z]*)(\d*(?:\.\d+)?)", formula)
        if tokens:
            for el, n in tokens:
                if n == "":
                    comp[el] += 1.0
                else:
                    comp[el] += float(n)

        # Normalize
        total = sum(comp.values())
        if total == 0:  # No valid elements found
            return None
            
        result = {el: val / total for el, val in comp.items()}
        
        # Validate that all elements follow proper chemical element patterns
        # Elements should be 1-2 characters: single capital letter or capital + lowercase
        for element in result.keys():
            if not (len(element) == 1 and element.isupper()) and not (len(element) == 2 and element[0].isupper() and element[1].islower()):
                return None
        
        return result
    except Exception as e:
        # Return None for any parsing errors or non-alloy components
        print(f"cannot parse formula: {formula}")
        print(e)
        return None

