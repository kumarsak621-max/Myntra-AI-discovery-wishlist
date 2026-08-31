from app.pipeline.quantification import label_distribution


def barrier_distribution(db, myntra_only: bool = True):
    return label_distribution(db, "barriers", myntra_only=myntra_only)
