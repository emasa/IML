
def meta_info(arff_meta):
    # auxiliary function to parse (feature name, feature type[, range of values (for categorical data)])
    meta_str = str(arff_meta)
    parsed_lines = [l.strip().split("'s type is ") for l in meta_str.split('\n') if 'type' in l]
    cleaned_lines = [(name, type_range.split(', range is')) for name, type_range in parsed_lines]
    final_lines = [(name, t_r[0]) if len(t_r) == 1 else  (name, t_r[0], t_r[1].translate(None, "(){}' ").split(','))
                   for name, t_r in cleaned_lines]
    final_mapping = [n_t_r if len(n_t_r) == 2 else (n_t_r[0], n_t_r[1], {val: k for k, val in enumerate(n_t_r[2])})
                     for n_t_r in final_lines]

    return final_mapping