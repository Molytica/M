from placeholder import get_trained_conversion_model


model = get_trained_conversion_model()

def normalize(t):
    # Normalize the transcriptome
    
    t = t / t.sum(axis=0)

    return t

def get_target_modulome(initial_t, target_t):
    # Get the target modulome from the initial and target transcriptome
    diff_t = normalize(target_t) - normalize(initial_t)
    diff_t = normalize(diff_t)

    target_modulome = model(diff_t)

    return target_modulome



