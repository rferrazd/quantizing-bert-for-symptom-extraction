
if __name__ == "__main__":
    import re
    PLACEHOLDER_RE = re.compile(r'\{(SYMPTOM[^}]*)\}')
    SINGLE_SYMPTOM_GROUPS = {"affirmed", "negated", "distractor"}
    # =============
    # HELPERS 
    # =============
    def extract_placeholder(template:str):
        """Return unique placeholder names in template, preserving order.Don't use set() it won't preserve the order"""
        return list(dict.fromkeys(PLACEHOLDER_RE.findall(template)))

    template = "On ROS, positive for SYMPTOM_POS_1 and {SYMPTOM_POS_1}, negative for SYMPTOM_NEG."
    # print(PLACEHOLDER_RE.findall(template))
    # print(list(set((PLACEHOLDER_RE.findall(template)))))
    # print(extract_placeholder(template))

    ph = template
    index_match = re.match(r'^(SYMPTOM_(?:POS|NEG|O))_(\d+)$', ph)
    print(index_match)
    print(index_match.group(1))