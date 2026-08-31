from mjr_am_backend.routes.route_catalog import (
    CORE_ROUTE_REGISTRATIONS,
    OPTIONAL_ROUTE_REGISTRATIONS,
)


def test_route_catalog_labels_are_unique():
    labels = [item.label for item in (*CORE_ROUTE_REGISTRATIONS, *OPTIONAL_ROUTE_REGISTRATIONS)]
    assert len(labels) == len(set(labels))


def test_route_catalog_tracks_recent_extractions():
    core_labels = {item.label for item in CORE_ROUTE_REGISTRATIONS}
    optional_labels = {item.label for item in OPTIONAL_ROUTE_REGISTRATIONS}

    assert "asset docs" in core_labels
    assert "download+duplicates" in optional_labels
