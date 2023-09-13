"""a script for calculating metrics."""
from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs) -> MetricCollection:
    """Get metrics.

    Args:
        **kwargs: remaining keyword arguments are passed to bar.  # noqa:RST210

    Returns:
        MetricCollection: get f1 score, precision and recall
    """
    # не понимаю как решить
    # RST210 Inline strong start-string without end-string.
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
        },
    )
