from features_extraction import dependency_path


def rule_retired(person, org, sent):
    tokens = sent.split()
    if 'retired' in tokens:
        p_idx = tokens.index(person.split()[-1])
        o_idx = tokens.index(org.split()[-1])
        r_idx = tokens.index('retired')
        if (p_idx < r_idx < o_idx) or (o_idx < r_idx < p_idx):
          if r_idx + 2 >= o_idx:
            return False
    return True


def rule_org_s(person, org, sent, lex_loc):
    tokens = sent.split()
    if "'s" in tokens:
        p_idx = tokens.index(person.split()[-1])
        o_idx = tokens.index(org.split()[-1])
        s_idx = tokens.index("'s")
        if o_idx < s_idx < p_idx:
            if s_idx - 1 == o_idx:
                if org in lex_loc:
                    return False
    return True


def root_propn(sent, per, org):
    for p in [ent for ent in sent.ents if ent.label_ == 'PER']:
        if p.text == per:
            per = p
            break
    for o in [ent for ent in sent.ents if ent.label_ == 'ORG']:
        if o.text == org:
            org = o
            break

    dep_path, root_pos = dependency_path(per, org, root_pos=True)
    if root_pos == 'PROPN':
        return False
    return True
