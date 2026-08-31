from app.pipeline.quantification import label_distribution


def intent_distribution(db, myntra_only: bool = True):
    return label_distribution(db, "intent", myntra_only=myntra_only)
